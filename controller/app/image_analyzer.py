import logging
import threading
from typing import Optional

import mcp
import zenoh
import zenoh.handlers
from ollama import Client, Image, Message

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe the content of the image in detail. "
    "Also describe positions of objects and their distance relative to the image center."
)


class ImageAnalyzer:
    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        image_topic: str = "IMAGE",
        mcp_host: str = "0.0.0.0",
        mcp_port: int = 9988,
        pull_model: bool = True,
    ):
        self.client = Client(host=ollama_url)
        self.model = model
        self.image_topic = image_topic

        self._last_image: Optional[bytes] = None
        self._lock = threading.RLock()

        if pull_model:
            logger.info("Ensuring model present: %s", model)
            self.client.pull(model=model, stream=False)

        server = mcp.server.FastMCP(
            name="image_describer", host=mcp_host, port=mcp_port
        )
        self.server = server

        @server.tool(
            description="Describe the latest camera image with an optional prompt."
        )
        def get_camera_image_description(prompt: str = DEFAULT_PROMPT) -> str:
            with self._lock:
                img = self._last_image
            if not img:
                return "no image seen yet."
            return self.describe_image(image=img, prompt=prompt)

    def describe_image(self, *, image: bytes, prompt: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                Message(
                    role="user",
                    content=prompt,
                    images=[Image(value=image)],
                )
            ],
            options={"temperature": 0},
        )
        return resp.message.content or ""

    # --- run loops (blocking) ---

    def run_mcp_server(self) -> None:
        # blocking call
        self.server.run(transport="streamable-http")

    def run_image_subscription(self, *, z: zenoh.Session) -> None:
        # blocking call
        sub = z.declare_subscriber(
            self.image_topic,
            handler=zenoh.handlers.RingChannel(capacity=1),
        )
        while True:
            sample = sub.recv()
            if not sample or not sample.payload:
                continue
            try:
                jpeg_bytes = sample.payload.to_bytes()  # raw JPEG payload
                with self._lock:
                    self._last_image = bytes(jpeg_bytes)
            except Exception as e:
                logger.error("Failed to update last image: %s", e, exc_info=True)

    def start_background(self, *, z: zenoh.Session) -> None:
        # convenience: start both in background threads
        threading.Thread(target=self.run_mcp_server, daemon=True).start()
        threading.Thread(
            target=self.run_image_subscription, args=(), kwargs={"z": z}, daemon=True
        ).start()
        logger.info("Local MCP image_describer + image subscriber running.")
