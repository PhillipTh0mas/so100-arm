# app/main.py
import asyncio
import json
import logging
import os
import socket
import urllib
from typing import List

import zenoh
from fast_agent import FastAgent
from ollama import Client

from app.image_analyzer import ImageAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info("Port %s is already in use on %s.", port, host)
            return True


def parse_zenoh_connect_endpoints(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []

    if s.startswith("["):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("ZENOH_CONNECT_ENDPOINTS JSON must be a list.")
        out: List[str] = []
        for x in arr:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
            out.append(x if x.startswith("tcp/") else f"tcp/{x}")
        return out

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [p if p.startswith("tcp/") else f"tcp/{p}" for p in parts]


def make_zenoh_session(connect_endpoints: List[str]) -> zenoh.Session:
    cfg = zenoh.Config()

    if not is_port_in_use(7447):
        cfg.insert_json5("listen/endpoints", json.dumps(["tcp/0.0.0.0:7447"]))

    if connect_endpoints:
        cfg.insert_json5("connect/endpoints", json.dumps(connect_endpoints))

    return zenoh.open(cfg)


async def wait_http(url: str, timeout_s: float = 30.0, interval_s: float = 0.5) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s

    def _probe() -> int:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code

    while True:
        try:
            status = await asyncio.to_thread(_probe)
            if 200 <= status < 500:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass

        if asyncio.get_event_loop().time() > deadline:
            raise RuntimeError(f"Timeout waiting for {url}")

        await asyncio.sleep(interval_s)


def warm_ollama_chat(ollama_url: str, model: str) -> None:
    base = model.replace("generic.", "")
    base_url = (
        ollama_url.rstrip("/") + "/v1"
        if not ollama_url.endswith("/v1")
        else ollama_url.rstrip("/")
    )

    body = json.dumps(
        {
            "model": base,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        _ = resp.read()


async def run_agent() -> None:
    model_name = env_str("MODEL_NAME", "generic.qwen3:4b-instruct")
    agent_instruction_topic = env_str("AGENT_INSTRUCTION_TOPIC", "AGENT_INSTRUCTION")
    agent_response_topic = env_str("AGENT_RESPONSE_TOPIC", "AGENT_RESPONSE")

    ollama_url = env_str("OLLAMA_URL", "").strip()
    if not ollama_url:
        raise ValueError("Missing env var: OLLAMA_URL (e.g. http://<ip>:11434)")

    # fast-agent generic provider env
    os.environ["GENERIC_BASE_URL"] = env_str(
        "GENERIC_BASE_URL", (ollama_url.rstrip("/") + "/v1")
    ).strip()
    os.environ["GENERIC_API_KEY"] = env_str("GENERIC_API_KEY", "ollama")
    logger.info("GENERIC_BASE_URL=%s", os.environ["GENERIC_BASE_URL"])

    robot_mcp_base = env_str("ROBOT_MCP_URL", "").strip()
    if not robot_mcp_base:
        raise ValueError("Missing env var: ROBOT_MCP_URL (e.g. http://<ip>:<port>)")

    zenoh_endpoints_raw = env_str("ZENOH_CONNECT_ENDPOINTS", "").strip()
    zenoh_connect_endpoints = parse_zenoh_connect_endpoints(zenoh_endpoints_raw)
    if not zenoh_connect_endpoints:
        raise ValueError(
            "Missing/empty env var: ZENOH_CONNECT_ENDPOINTS "
            "(e.g. '10.0.0.2:7447' or '10.0.0.2:7447,10.0.0.3:7447')"
        )

    # Zenoh session + pub/sub
    z = make_zenoh_session(zenoh_connect_endpoints)
    sub = z.declare_subscriber(
        agent_instruction_topic, handler=zenoh.handlers.RingChannel(capacity=100)
    )
    pub = z.declare_publisher(agent_response_topic)

    # Local image MCP server
    image_analyzer = ImageAnalyzer(
        ollama_url=ollama_url,
        model=env_str("IMAGE_MODEL_NAME", "ministral-3"),
        image_topic=env_str("IMAGE_TOPIC", "IMAGE"),
        mcp_host="0.0.0.0",
        mcp_port=int(env_str("IMAGE_MCP_PORT", "9989")),
        pull_model=True,
    )
    image_analyzer.start_background(z=z)

    # Ensure ollama model exists if using generic.*
    client = Client(host=ollama_url)
    if model_name.startswith("generic."):
        base = model_name.replace("generic.", "")
        logger.info("Ensuring model present: %s", base)
        for e in client.pull(model=base, stream=True):
            logger.info("Model download: %s", e)
        logger.info("Model ready: %s", base)
        logger.info("Warming Ollama chat endpoint...")
        await asyncio.to_thread(warm_ollama_chat, ollama_url, model_name)
        logger.info("Ollama warm.")

    # Wait for MCP endpoints
    await wait_http(f"{robot_mcp_base.rstrip('/')}/mcp")
    await wait_http(f"http://127.0.0.1:{int(env_str('IMAGE_MCP_PORT', '9989'))}/mcp")

    # fast-agent auto-loads fastagent.config.yaml from CWD/parents
    fast = FastAgent("RobotOrchestrator")

    @fast.agent(
        name="robot_agent",
        model=model_name,
        servers=["image-description-mcp", "robot-mcp"],
        instruction=(
            "You are the robot agent. You control the robot and can also describe the scene.\n"
            "You have two tools:\n"
            "- image-description-mcp: set a prompt and ask what the camera sees.\n"
            "- robot-mcp: execute movements, rotations, or manipulations.\n\n"
            "Tool calling contract:\n"
            "- When you need to use a tool, you MUST respond with a structured tool call, not natural language.\n"
            "- While calling tools: output ONLY tool call(s) with valid JSON arguments.\n"
            "- After tool results: output the final user-facing message.\n\n"
            "Rules:\n"
            "1) ACTION requests: ALWAYS call robot-mcp (use image-description-mcp first if location is needed).\n"
            "2) OBSERVATION requests: ALWAYS call image-description-mcp then summarize.\n\n"
            "User-facing output (after tool results only):\n"
            "- Exactly 1 sentence; prefer <= 8 words unless describing the scene.\n"
            "- Never mention tools/JSON.\n"
            "- Never claim an action happened unless robot-mcp actually ran.\n"
        ),
        use_history=True,
        human_input=False,
    )
    async def mcp_agent_loop() -> None:
        stacked_message = ""
        while True:
            try:
                async with fast.run() as agent:
                    while True:
                        sample = sub.try_recv()
                        if sample and sample.payload:
                            msg = sample.payload.to_bytes().decode("utf-8")
                            if msg.lower().strip(".") == "restart":
                                pub.put(payload=b"Restarting controller!")
                                return

                            if (msg.startswith("(") and msg.endswith(")")) or (
                                msg.startswith("[") and msg.endswith("]")
                            ):
                                await asyncio.sleep(0.1)
                            else:
                                stacked_message = (
                                    (stacked_message + "\n" + msg)
                                    if stacked_message
                                    else msg
                                )
                                await asyncio.sleep(0.1)

                        elif stacked_message:
                            try:
                                result = await asyncio.wait_for(
                                    agent.robot_agent.send(stacked_message),
                                    timeout=160.0,
                                )
                                logger.info(
                                    "agent result type=%s repr=%r", type(result), result
                                )
                                if result:
                                    pub.put(payload=result.encode("utf-8"))
                                stacked_message = ""
                            except Exception as e:
                                pub.put(payload=f"Publish failed: {e}".encode("utf-8"))
                                raise
                        else:
                            await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                logger.warning(
                    "Agent call exceeded 160s; restarting session (message retained)."
                )
                pub.put(payload=b"Restarting controller!")
                raise
            except BaseException as e:
                logger.error("Session error caught; will restart: %s", e, exc_info=True)
                pub.put(payload=b"Restarting controller!")
                return

            await asyncio.sleep(0.5)

    await mcp_agent_loop()


if __name__ == "__main__":
    asyncio.run(run_agent())
