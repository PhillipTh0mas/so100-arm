# SO-100 Agent Stack (Controller + LeRobot + Whisper + TTS)

## Demo

<p align="center">
  <a href="https://app.vidzflow.com/v/7PCM7xIxul">
    <img src="./assets/demo-thumbnail.png" width="800" alt="Watch Demo">
  </a>
</p>

## Architecture

```text

┌──────────────────────────────────────────────────────────┐
│ ollama/                                                  │
│ - LLM endpoint                                           │
│ - VLA for image description                              │
│   http://ollama:11434                                    │
└──────────────────────────────────────────────────────────┘
                          ▲
                          │  HTTP (OpenAI-compat)
                          │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ controller/                                                                            │
│ - Web UI (:8080) + WebSocket /ws                                                       │
│ - LLM orchestration (Ollama + image-description MCP)                                   │
│ - Robot control (robot-mcp)                                                            │
│ - Audio bridge (Zenoh mic/speaker)                                                     │
└─────────┬───────────────────────────┬──────────────────────────────┬───────────────────┘
          │                           │                              │
          │ HTTP (MCP / RERUN)        │ Zenoh                        │ Zenoh
          │                           │                              │
          ▼                           ▼                             ▼

┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ lerobot-drive/           │   │ whisper/                 │   │ tts/                     │
│ - SO-100 control         │   │ - sub AUDIO_IN           │   │ - sub LLM_OUTPUT_TEXT    │
│ - MCP server (:9988)     │   │ - pub TRANSCRIPT_TEXT    │   │ - pub AUDIO_OUT          │
│ - Rerun server (:9877)   │   └──────────────────────────┘   └──────────────────────────┘
└──────────────────────────┘


```

### Zenoh topics (current)

- `AUDIO_IN` (bytes): raw PCM s16le mono (browser microphone)
- `TRANSCRIPT_TEXT` (utf-8): ASR output (whisper)
- `AUDIO_OUT` (bytes): raw PCM s16le mono (TTS output for browser playback)

Optional (if enabled in drive):

- `CAMERA_1_IMAGE`, `CAMERA_2_IMAGE` (bytes): raw JPEG bytes

---

## Docker Compose setup

### Prereqs

- Docker + Docker Compose (v2)
- Optional camera device on host (e.g. `/dev/video0`)
- Optional LeRobot gRPC server address reachable by `lerobot-drive` (`LEROBOT_SERVER_ADDRESS`) if you run policy control

### Files

- `docker-compose.yml` at repo root
- `lerobot-drive/calibration.json` (mounted into the drive container)

### Start

```bash
docker compose build
docker compose up -d
docker compose logs -f robot-orchestrator
```

### Open the UI

- Web UI: `http://localhost:8080`
- It embeds the rerun viewer via `RERUN_URL` (defaults to `http://lerobot-drive:9877/` inside the compose network).

### Optional: enable camera

```bash
export CAMERA_DEV=/dev/video0
docker compose up -d --build lerobot-drive
```

### Stop

```bash
docker compose down
```

### Useful debugging

```bash
docker compose logs -f whisper
docker compose logs -f tts
docker compose logs -f lerobot-drive
```

---

## make87 setup

This repo can be run under make87 as a system deployment. The typical workflow is:

### 1) Pick device and ensure it’s online

```bash
m87 device list
```

### 2) Deploy with docker-compose on the device

From this repo root:

```bash
m87 device <DEVICE_NAME> docker compose up -d --build
```

Follow logs:

```bash
m87 device <DEVICE_NAME> docker compose logs -f robot-orchestrator
```

### 3) Update / redeploy

```bash
m87 device <DEVICE_NAME> docker compose build
m87 device <DEVICE_NAME> docker compose up -d
```

### 4) Tear down

```bash
m87 device <DEVICE_NAME> docker compose down
```

Notes:

- `LEROBOT_SERVER_ADDRESS`, `ZENOH_CONNECT_ENDPOINTS`, and `RERUN_URL` should be set in the device environment for the deployment.
- `lerobot-drive/calibration.json` should be present on the device (or provided as a mounted config file path).

```

```
