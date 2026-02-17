use anyhow::{Context, Result};
use std::env;

use tokio::sync::mpsc;
use whisper_rs::{
    convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters,
};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // --- Whisper init
    let model = env::var("WHISPER_MODEL").unwrap_or("/models/ggml-tiny.en.bin".into());
    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default())
        .with_context(|| format!("loading model {model}"))?;
    let state = ctx.create_state().context("create whisper state")?;

    // --- Runtime config from env (defaults match your prior config)
    let sr_fallback: usize = env::var("SR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16_000);
    let gap_ms: i64 = env::var("GAP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let min_utter_ms: i64 = env::var("MIN_UTTER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);
    let max_utter_ms: i64 = env::var("MAX_UTTER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);
    let vad_rms_th: f32 = env::var("VAD_RMS_TH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.010);
    let best_of: i32 = env::var("BEST_OF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    // Greedy for lowest latency
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of });
    params.set_detect_language(false);
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_no_context(true);
    params.set_temperature(0.0);
    params.set_n_threads(std::cmp::min(num_cpus::get(), 4) as i32);
    params.set_print_realtime(false);
    params.set_print_progress(false);

    // --- Zenoh config from env
    // Examples:
    //   ZENOH_CONNECT_ENDPOINTS="controller:7447,lerobot-drive:7447"
    //   ZENOH_AUDIO_IN_KEY="AUDIO_IN"
    //   ZENOH_TEXT_OUT_KEY="TRANSCRIPT_TEXT"
    let endpoints_raw = env::var("ZENOH_CONNECT_ENDPOINTS").unwrap_or_default();
    let connect_endpoints = parse_zenoh_connect_endpoints(&endpoints_raw)?;
    if connect_endpoints.is_empty() {
        anyhow::bail!("Missing/empty ZENOH_CONNECT_ENDPOINTS");
    }

    let audio_in_key = env::var("ZENOH_AUDIO_IN_KEY").unwrap_or("AUDIO_IN".into());
    let text_out_key = env::var("ZENOH_TEXT_OUT_KEY").unwrap_or("TRANSCRIPT_TEXT".into());

    let session = open_zenoh(&connect_endpoints).await.context("open zenoh")?;
    let sub = session
        .declare_subscriber(audio_in_key.clone())
        .with(zenoh::handlers::RingChannel::new(16))
        .await
        .map_err(|e| anyhow::anyhow!("declare subscriber: {}", e))?;
    let pub_text = session
        .declare_publisher(text_out_key.clone())
        .await
        .map_err(|e| anyhow::anyhow!("declare publisher: {}", e))?;

    // --- worker channel (don’t block subscriber) — pass (samples_i16, input_sr)
    let (tx, mut rx) = mpsc::channel::<(Vec<i16>, usize)>(8);

    // Whisper worker
    let mut worker_state = state;
    let worker_params = params.clone();
    let (pub_tx, mut pub_rx) = mpsc::channel::<String>(100);

    tokio::spawn(async move {
        while let Some(s) = pub_rx.recv().await {
            let s = s.trim();
            if s.is_empty() || s == "[BLANK_AUDIO]" {
                continue;
            }
            println!("Detected: {s}");
            if let Err(e) = pub_text.put(s.as_bytes()).await {
                eprintln!("zenoh publish error: {e:?}");
            }
        }
    });

    tokio::task::spawn_blocking(move || -> Result<()> {
        while let Some((mono_i16, in_sr)) = rx.blocking_recv() {
            if mono_i16.is_empty() {
                continue;
            }

            // Convert to f32 [-1,1]
            let mut audio_f32 = vec![0.0f32; mono_i16.len()];
            if let Err(e) = convert_integer_to_float_audio(&mono_i16, &mut audio_f32) {
                eprintln!("convert_integer_to_float_audio error: {e:?}");
                continue;
            }

            // Resample to 16 kHz for Whisper
            let audio_f32 = resample_linear_to_16k(&audio_f32, in_sr);

            // Inference
            worker_state
                .full(worker_params.clone(), &audio_f32)
                .context("whisper full")?;

            // Collect segments
            let n = worker_state.full_n_segments();
            let mut out = String::new();
            for i in 0..n {
                if let Some(s) = worker_state.get_segment(i) {
                    out.push_str(s.to_str().unwrap_or(""));
                }
            }
            let out = out.trim();
            let _ = pub_tx.try_send(out.to_string());
        }
        Ok(())
    });

    // --- low-latency gap/VAD aggregator (assumes incoming audio is PCM s16le mono @ SR env)
    let input_sr: usize = env::var("AUDIO_SR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(sr_fallback);

    let mut buf: Vec<i16> = Vec::with_capacity(sr_fallback * 3);
    let mut utter_start_ms: Option<i64> = None;
    let mut last_rx_ms: Option<i64> = None;

    while let Ok(sample) = sub.recv_async().await {
        let now_ms = now_ms();

        // bytes -> i16 mono
        let data = sample.payload().to_bytes();
        let i16s: Vec<i16> = data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // gap detection based on wall clock (since raw bytes have no pts/time_base)
        let mut flush = false;
        if let Some(prev) = last_rx_ms {
            if now_ms - prev >= gap_ms {
                flush = true;
            }
        }
        if let Some(start) = utter_start_ms {
            if now_ms - start >= max_utter_ms {
                flush = true;
            }
        }

        buf.extend_from_slice(&i16s);

        let dur_ms = ((buf.len() as f32) * 1000.0 / input_sr as f32) as i64;
        let quiet = rms_i16(&buf) < vad_rms_th;

        if dur_ms >= 2000 || (dur_ms >= 800 && quiet) {
            flush = true;
        }

        if flush {
            if let (Some(start), Some(_prev)) = (utter_start_ms, last_rx_ms) {
                let span_ms = now_ms - start;
                if span_ms >= min_utter_ms && !buf.is_empty() {
                    let _ = tx.try_send((std::mem::take(&mut buf), input_sr));
                } else {
                    buf.clear();
                }
            }
            utter_start_ms = Some(now_ms);
        }

        if utter_start_ms.is_none() {
            utter_start_ms = Some(now_ms);
        }
        last_rx_ms = Some(now_ms);
    }

    Ok(())
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    d.as_millis() as i64
}

fn parse_zenoh_connect_endpoints(s: &str) -> Result<Vec<String>> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }

    // JSON list
    if s.starts_with('[') {
        let arr: Vec<String> =
            serde_json::from_str(s).context("parse ZENOH_CONNECT_ENDPOINTS json")?;
        let out = arr
            .into_iter()
            .filter_map(|x| {
                let x = x.trim().to_string();
                if x.is_empty() {
                    None
                } else if x.starts_with("tcp/") {
                    Some(x)
                } else {
                    Some(format!("tcp/{x}"))
                }
            })
            .collect();
        return Ok(out);
    }

    // CSV
    let out = s
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| {
            if p.starts_with("tcp/") {
                p.to_string()
            } else {
                format!("tcp/{p}")
            }
        })
        .collect();
    Ok(out)
}

async fn open_zenoh(connect_endpoints: &[String]) -> Result<zenoh::Session> {
    let mut cfg = zenoh::Config::default();
    let _ = cfg
        .insert_json5(
            "connect/endpoints",
            serde_json::to_string(connect_endpoints)?.as_str(),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(zenoh::open(cfg)
        .await
        .map_err(|e| anyhow::anyhow!("zenoh open error: {e:?}"))?)
}

/// Linear resampler to 16 kHz.
fn resample_linear_to_16k(input: &[f32], in_sr: usize) -> Vec<f32> {
    if in_sr == 16_000 {
        return input.to_vec();
    }
    if in_sr == 0 || input.is_empty() {
        return Vec::new();
    }
    let ratio = 16_000.0 / in_sr as f32;
    let out_len = ((input.len() as f32) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = i as f32 / ratio;
        let j = t.floor() as usize;
        let a = t - j as f32;
        let s0 = *input.get(j).unwrap_or(&0.0);
        let s1 = *input.get(j + 1).unwrap_or(&s0);
        out.push(s0 + a * (s1 - s0));
    }
    out
}

/// Simple RMS on i16 buffer scaled to [-1,1]
fn rms_i16(a: &[i16]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .map(|&x| {
            let v = x as f64 / 32768.0;
            v * v
        })
        .sum();
    (sum / (a.len() as f64)).sqrt() as f32
}
