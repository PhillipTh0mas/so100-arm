import io
import json
import logging
import os
import socket
import tempfile
import time
import wave

import numpy as np
import pyttsx3
import zenoh

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tts")


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def parse_zenoh_connect_endpoints(s: str):
    s = (s or "").strip()
    if not s:
        return []
    if s.startswith("["):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("ZENOH_CONNECT_ENDPOINTS JSON must be a list.")
        out = []
        for x in arr:
            if isinstance(x, str) and x.strip():
                x = x.strip()
                out.append(x if x.startswith("tcp/") else f"tcp/{x}")
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [p if p.startswith("tcp/") else f"tcp/{p}" for p in parts]


def make_zenoh_session(connect_endpoints):
    cfg = zenoh.Config()
    if not is_port_in_use(7447):
        cfg.insert_json5("listen/endpoints", json.dumps(["tcp/0.0.0.0:7447"]))
    if connect_endpoints:
        cfg.insert_json5("connect/endpoints", json.dumps(connect_endpoints))
    return zenoh.open(cfg)


class TextToPcmS16le:
    """Text -> PCM s16le mono bytes (resampled)."""

    def __init__(self, target_sr: int):
        self.target_sr = target_sr
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", env_int("TTS_RATE", 150))
        self.engine.setProperty("volume", env_float("TTS_VOLUME", 1.0))

        voice = env_str("TTS_VOICE", "").strip()
        if voice:
            self.engine.setProperty("voice", voice)
        else:
            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)

    def _text_to_wav_bytes(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _wav_to_i16(self, wav_data: bytes):
        wav_io = io.BytesIO(wav_data)
        with wave.open(wav_io, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()

        if sw == 1:
            a = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            a = ((a - 128.0) * 256.0).astype(np.int16)
        elif sw == 2:
            a = np.frombuffer(frames, dtype=np.int16)
        elif sw == 4:
            a = np.frombuffer(frames, dtype=np.int32)
            a = (a // 65536).astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sw}")

        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
            ch = 1

        return a, sr, ch

    def _resample_i16(self, a: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
        if in_sr == out_sr or a.size == 0:
            return a.astype(np.int16)

        ratio = out_sr / float(in_sr)
        out_len = int(round(a.size * ratio))
        x_old = np.arange(a.size, dtype=np.float32)
        x_new = np.linspace(0, a.size - 1, out_len, dtype=np.float32)
        out = np.interp(x_new, x_old, a.astype(np.float32)).astype(np.int16)
        return out

    def synth(self, text: str) -> bytes:
        wav = self._text_to_wav_bytes(text)
        a, in_sr, _ch = self._wav_to_i16(wav)
        a = self._resample_i16(a, in_sr, self.target_sr)
        return a.astype("<i2").tobytes()


def main():
    endpoints = parse_zenoh_connect_endpoints(env_str("ZENOH_CONNECT_ENDPOINTS", ""))
    if not endpoints:
        raise ValueError("Missing/empty ZENOH_CONNECT_ENDPOINTS")

    text_key = env_str("ZENOH_TTS_TEXT_KEY", "TRANSCRIPT_TEXT")
    audio_key = env_str("ZENOH_TTS_AUDIO_KEY", "AUDIO_OUT")
    out_sr = env_int("TTS_SR", 24000)

    z = make_zenoh_session(endpoints)

    sub = z.declare_subscriber(
        text_key, handler=zenoh.handlers.RingChannel(capacity=100)
    )
    pub = z.declare_publisher(audio_key)

    tts = TextToPcmS16le(target_sr=out_sr)

    logger.info(f"Subscribing text: {text_key}")
    logger.info(f"Publishing audio: {audio_key} (sr={out_sr})")

    while True:
        s = sub.recv()
        if not s or not s.payload:
            continue
        text = s.payload.to_bytes().decode("utf-8", errors="ignore").strip()
        if not text:
            continue

        try:
            pcm = tts.synth(text)
            if pcm:
                pub.put(pcm)
            logger.info(f"TTS: {text}")
        except Exception as e:
            logger.exception(f"TTS failed: {e}")


if __name__ == "__main__":
    main()
