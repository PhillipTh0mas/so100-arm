import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Set

import zenoh
from aiohttp import WSMsgType, web


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None else default


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


class AudioBridge:
    def __init__(self):
        endpoints = parse_zenoh_connect_endpoints(
            env_str("ZENOH_CONNECT_ENDPOINTS", "")
        )
        if not endpoints:
            raise ValueError("Missing/empty ZENOH_CONNECT_ENDPOINTS")

        self._z = make_zenoh_session(endpoints)

        # From browser -> zenoh
        self._audio_in_key = env_str("ZENOH_KEY_AUDIO_IN", "AUDIO_IN")
        self._pub_in = self._z.declare_publisher(self._audio_in_key)

        # From zenoh -> browser
        self._audio_out_key = env_str("ZENOH_KEY_AUDIO_OUT", "AUDIO_OUT")
        self._sub_out = self._z.declare_subscriber(
            self._audio_out_key,
            handler=zenoh.handlers.RingChannel(capacity=1024),
        )

        self._clients: Set[web.WebSocketResponse] = set()
        self._fanout_task: asyncio.Task | None = None

    async def start(self):
        if self._fanout_task is None:
            self._fanout_task = asyncio.create_task(self._fanout_loop())

    async def _fanout_loop(self):
        while True:
            sample = self._sub_out.recv()  # blocking call in zenoh, run in thread
            payload = sample.payload.to_bytes()

            # aiohttp websockets want async sends -> bounce into loop thread safely
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_bytes(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # raw int16 PCM bytes
                    self._pub_in.put(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._clients.discard(ws)
            await ws.close()

        return ws


def create_app(static_dir: Path) -> web.Application:
    bridge = AudioBridge()

    app = web.Application()
    app["bridge"] = bridge

    async def on_startup(_app: web.Application):
        await bridge.start()

    app.on_startup.append(on_startup)

    async def config_json(_req: web.Request) -> web.Response:
        rerun_url = env_str("RERUN_URL", "about:blank")
        return web.json_response({"rerun_url": rerun_url})

    async def index(_req: web.Request) -> web.FileResponse:
        return web.FileResponse(static_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_get("/config.json", config_json)
    app.router.add_get("/ws", bridge.handle_ws)

    return app


async def serve_webui() -> None:
    port = int(env_str("WEB_PORT", "8080"))
    static_dir = Path(__file__).resolve().parent.parent / "static"

    app = create_app(static_dir)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # keep running
    await asyncio.Event().wait()
