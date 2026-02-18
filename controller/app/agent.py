import asyncio
import json
import logging
import os
import socket
import urllib
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import yaml
import zenoh
from mcp_agent.config import (
    LoggerSettings,
    MCPServerSettings,
    MCPSettings,
    OpenAISettings,
    Settings,
)
from mcp_agent.core.fastagent import FastAgent
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
            logger.info(f"Port {port} is already in use on {host}.")
            return True


def parse_zenoh_connect_endpoints(s: str) -> List[str]:
    """
    Accepts env formats like:
      - "10.0.0.2:7447"
      - "10.0.0.2:7447,10.0.0.3:7447"
      - "tcp/10.0.0.2:7447,tcp/10.0.0.3:7447"
      - '["tcp/10.0.0.2:7447","tcp/10.0.0.3:7447"]'
    Returns a list of zenoh endpoint strings like: ["tcp/..:..", ...]
    """
    s = (s or "").strip()
    if not s:
        return []

    # JSON list
    if s.startswith("["):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("ZENOH_CONNECT_ENDPOINTS JSON must be a list.")
        out = []
        for x in arr:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
            out.append(x if x.startswith("tcp/") else f"tcp/{x}")
        return out

    # CSV
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(p if p.startswith("tcp/") else f"tcp/{p}")
    return out


def make_zenoh_session(connect_endpoints: List[str]) -> zenoh.Session:
    cfg = zenoh.Config()

    # Optional listen endpoint (matches your prior behavior)
    if not is_port_in_use(7447):
        cfg.insert_json5("listen/endpoints", json.dumps(["tcp/0.0.0.0:7447"]))

    if connect_endpoints:
        cfg.insert_json5("connect/endpoints", json.dumps(connect_endpoints))

    return zenoh.open(cfg)


def create_fast_agent(
    ollama_url: str,
    mcp_urls: Dict[str, str],
    openai_key: Optional[str],
) -> FastAgent:
    if not ollama_url.endswith("/v1"):
        ollama_url += "/v1"
    os.environ["GENERIC_BASE_URL"] = ollama_url
    logger.info(f"GENERIC_BASE_URL: {os.environ['GENERIC_BASE_URL']}")

    setting = Settings(
        logger=LoggerSettings(type="console"),
        mcp=MCPSettings(
            servers={
                k: MCPServerSettings(
                    name=k,
                    transport="http",
                    url=v,
                )
                for k, v in mcp_urls.items()
            }
        ),
        openai=OpenAISettings(api_key=openai_key) if openai_key else None,
    )

    with NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as temp_file:
        yaml.dump(setting.model_dump(), temp_file)
        temp_file.flush()
        return FastAgent(
            "RobotOrchestrator",
            config_path=temp_file.name,
            ignore_unknown_args=True,
        )


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


async def run_agent():
    # ==== Env config ====
    model_name = env_str("MODEL_NAME", "generic.qwen3:4b-instruct")
    openai_api_key = env_str("OPENAI_API_KEY", "")
    agent_instruction_topic = env_str("AGENT_INSTRUCTION_TOPIC", "AGENT_INSTRUCTION")
    agent_response_topic = env_str("AGENT_RESPONSE_TOPIC", "AGENT_RESPONSE")

    ollama_url = env_str("OLLAMA_URL", "").strip()
    if not ollama_url:
        raise ValueError("Missing env var: OLLAMA_URL (e.g. http://<ip>:11434)")

    robot_mcp_base = env_str("ROBOT_MCP_URL", "").strip()
    if not robot_mcp_base:
        raise ValueError("Missing env var: ROBOT_MCP_URL (e.g. http://<ip>:<port>)")

    # Zenoh connection endpoints (IPs:port list)
    # Examples:
    #   ZENOH_CONNECT_ENDPOINTS="10.0.0.2:7447,10.0.0.3:7447"
    #   ZENOH_CONNECT_ENDPOINTS='["tcp/10.0.0.2:7447","tcp/10.0.0.3:7447"]'
    zenoh_endpoints_raw = env_str("ZENOH_CONNECT_ENDPOINTS", "").strip()
    zenoh_connect_endpoints = parse_zenoh_connect_endpoints(zenoh_endpoints_raw)
    if not zenoh_connect_endpoints:
        raise ValueError(
            "Missing/empty env var: ZENOH_CONNECT_ENDPOINTS "
            "(e.g. '10.0.0.2:7447' or '10.0.0.2:7447,10.0.0.3:7447')"
        )

    # ==== Zenoh session + pub/sub ====
    z = make_zenoh_session(zenoh_connect_endpoints)
    sub = z.declare_subscriber(
        agent_instruction_topic,
        handler=zenoh.handlers.RingChannel(capacity=100),
    )
    pub = z.declare_publisher(agent_response_topic)

    image_analyzer = ImageAnalyzer(
        ollama_url=ollama_url,
        model=env_str("IMAGE_MODEL_NAME", "ministral-3"),
        image_topic=env_str("IMAGE_TOPIC", "IMAGE"),
        mcp_host="0.0.0.0",
        mcp_port=int(env_str("IMAGE_MCP_PORT", "9989")),
        pull_model=True,
    )
    image_analyzer.start_background(z=z)

    local_image_mcp_url = (
        f"http://127.0.0.1:{int(env_str('IMAGE_MCP_PORT', '9989'))}/mcp"
    )

    # Ensure the ollama model exists if using generic.*
    client = Client(host=ollama_url)
    if model_name.startswith("generic."):
        base = model_name.replace("generic.", "")
        logger.info(f"Ensuring model present: {base}")
        for e in client.pull(model=base, stream=True):
            logger.info(f"Model download: {e}")
        logger.info(f"Model ready: {base}")
        logger.info("Warming Ollama chat endpoint...")
        await asyncio.to_thread(warm_ollama_chat, ollama_url, model_name)
        logger.info("Ollama warm.")

    await wait_http(f"{robot_mcp_base.rstrip('/')}/mcp")
    fast = create_fast_agent(
        openai_key=openai_api_key,
        mcp_urls={
            "robot-mcp": f"{robot_mcp_base.rstrip('/')}/mcp",
            "image-description-mcp": local_image_mcp_url,
        },
        ollama_url=ollama_url,
    )

    @fast.agent(
        name="robot_agent",
        model=model_name,
        servers=["image-description-mcp", "robot-mcp"],
        instruction=(
            "You are the robot agent. You control the robot and can also describe the scene. "
            "You have two tools:\n"
            "- image-description-mcp: set a prompt and ask what the camera sees.\n"
            "- robot-mcp: execute movements, rotations, or manipulations.\n\n"
            "Rules of behavior:\n"
            "1. If the user asks for an ACTION (move, rotate, manipulate, open, close, etc.), "
            "   you MUST call robot-mcp to execute it. "
            "   - If the action refers to an object’s location, first set a prompt with image-description-mcp "
            "     to include relative positions, then call robot-mcp.\n"
            "   - Bias strongly toward action. If unsure, treat it as an action.\n"
            "   - IMPORTANT: You may only output a reply after the robot-mcp tool call has been made.\n"
            "2. If the user asks for an OBSERVATION (what do you see, describe, etc.), "
            "   you MUST call image-description-mcp and summarize the scene. "
            "   A question about the scene must ALWAYS trigger the tool call.\n"
            "3. If the input does not clearly match rule 1 or 2 (nonsense, filler, laughter, non-imperative phrases), "
            "   you MUST respond with exactly ''. "
            "   This rule overrides all others.\n\n"
            "Response style:\n"
            "- Keep replies short and varied. NEVER more than 1 sentence. Stay short as possible!\n"
            "- Keep replies very short (max one clause, no more than 8 words) unless you describe what you see.\n"
            "- Do not add second sentences, elaborations, or filler. Vary your responses!\n"
            "- For scene descriptions: one short sentence summary.\n"
            "- Never mention JSON, tools, or internal reasoning.\n"
            "- Never refuse or disclaim. Either describe or act.\n\n"
            "IMPORTANT:\n"
            "- Always begin every reply with the exact prefix 'ROBOT: '.\n"
            "- For ACTION requests: only output after robot-mcp call has executed.\n"
            "- For OBSERVATION requests: only output after image-description-mcp call has executed.\n"
            "- Never claim an action was done unless you actually made the tool call."
        ),
        use_history=True,
        human_input=False,
    )
    async def mcp_agent_loop():
        stacked_message = ""
        while True:
            try:
                async with fast.run() as agent:
                    while True:
                        sample = sub.try_recv()
                        if sample and sample.payload:
                            msg = sample.payload.to_bytes().decode("utf-8")
                            if msg.lower().strip(".") == "restart":
                                pub.put(
                                    payload="Restarting controller!".encode("utf-8")
                                )
                                return
                            if msg.startswith("(") and msg.endswith(")"):
                                await asyncio.sleep(0.1)
                            else:
                                stacked_message = (
                                    (stacked_message + "\n" + msg)
                                    if stacked_message
                                    else msg
                                )
                                await asyncio.sleep(0.1)

                        elif stacked_message:

                            async def send_with_logging(msg: str):
                                print(f"Sending message: {msg}")
                                try:
                                    return await agent.robot_agent.send(msg)
                                except Exception as e:
                                    print(f"Error sending message: {e}")

                            try:
                                result = await asyncio.wait_for(
                                    send_with_logging(stacked_message),
                                    timeout=160.0,
                                )

                                if result:
                                    if "ROBOT:" not in result:
                                        result = "ROBOT: " + result
                                    res = result.split("ROBOT:")[1]
                                    pub.put(payload=res.encode("utf-8"))

                                stacked_message = ""
                            except Exception as e:
                                pub.put(payload=f"Publish failed: {e}".encode("utf-8"))
                                raise
                        else:
                            await asyncio.sleep(0.1)

            except SystemExit as e:
                logger.error(f"Session SystemExit caught ({e.code}); will restart.")
                pub.put(payload="Restarting controller!".encode("utf-8"))
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent call exceeded 30s; restarting session (message retained)."
                )
                pub.put(payload="Restarting controller!".encode("utf-8"))
                raise
            except BaseException as e:
                logger.error(f"Session error caught; will restart: {e}", exc_info=True)
                pub.put(payload="Restarting controller!".encode("utf-8"))
                return

            await asyncio.sleep(0.5)

    await mcp_agent_loop()


def warm_ollama_chat(ollama_url: str, model: str) -> None:
    # model_name is like "generic.xxx" in your agent config; ollama wants base
    base = model.replace("generic.", "")
    if not ollama_url.endswith("/v1"):
        base_url = ollama_url.rstrip("/") + "/v1"
    else:
        base_url = ollama_url.rstrip("/")

    import json as _json
    import urllib.request

    body = _json.dumps(
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
        answ = resp.read()
        print(answ)


if __name__ == "__main__":
    asyncio.run(run_agent())
