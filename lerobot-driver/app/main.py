import json
import logging
import os
import socket
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import zenoh

from app.viewer import init_rerun_viewer_server

logger = logging.getLogger(__name__)


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


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            logger.info(f"Port {port} is already in use on {host}.")
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
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
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


@dataclass
class Camera:
    name: str
    dev_path: str


def discover_cameras_from_env() -> Dict[str, Camera]:
    cams: Dict[str, Camera] = {}

    cam1_dev = env_str("CAMERA_1_DEV", "").strip()
    if cam1_dev:
        cam1_name = env_str("CAMERA_1_NAME", "front").strip() or "front"
        cams[cam1_name] = Camera(name=cam1_name, dev_path=cam1_dev)

    cam2_dev = env_str("CAMERA_2_DEV", "").strip()
    if cam2_dev:
        cam2_name = env_str("CAMERA_2_NAME", "wrist").strip() or "wrist"
        cams[cam2_name] = Camera(name=cam2_name, dev_path=cam2_dev)

    return cams


def load_calibration() -> Optional[dict]:
    """
    Prefer a mounted file over env (cleaner for large nested configs).

    Env:
      - CALIBRATION_PATH=/config/calibration.json  (recommended)
      - CALIBRATION_JSON='{"shoulder_pan":{...}}'  (fallback)
    """
    path = env_str("CALIBRATION_PATH", "/config/calibration.json").strip()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raw = env_str("CALIBRATION_JSON", "").strip()
    if raw:
        return json.loads(raw)

    return None


class ImageChangePublisher:
    def __init__(
        self,
        image_change_threshold: float,
        only_publish_image_on_change: bool,
        publisher: zenoh.Publisher,
    ):
        self._last_image: Optional[np.ndarray] = None
        self.only_publish_image_on_change = only_publish_image_on_change
        self.image_change_threshold = image_change_threshold
        self.publisher = publisher

    def _frame_changed(
        self, current: np.ndarray, last: np.ndarray, threshold: float
    ) -> bool:
        if current.shape != last.shape:
            return True
        diff = cv2.absdiff(current, last)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_ratio = float(np.count_nonzero(gray)) / float(gray.size)
        return changed_ratio >= threshold

    def on_new_image(self, img_bgr: np.ndarray):
        try:
            if self.only_publish_image_on_change and self._last_image is not None:
                if not self._frame_changed(
                    img_bgr, self._last_image, self.image_change_threshold
                ):
                    return

            self._last_image = img_bgr

            ret, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ret:
                return
            # Publish raw JPEG bytes (no protobuf wrapper)
            self.publisher.put(jpeg.tobytes())
        except Exception as e:
            logger.error(f"Error sending image: {e}")
            return


def run_policy_controlled():
    from lerobot.scripts.server.helpers import Observation

    from app.robot_client import CustomRobotClient, get_so100_policy_config

    zenoh_endpoints = parse_zenoh_connect_endpoints(
        env_str("ZENOH_CONNECT_ENDPOINTS", "")
    )
    if not zenoh_endpoints:
        raise ValueError("Missing/empty ZENOH_CONNECT_ENDPOINTS")

    z = make_zenoh_session(zenoh_endpoints)

    key_agent_logs = env_str("ZENOH_KEY_AGENT_LOGS", "AGENT_LOGS")
    key_cam1 = env_str("ZENOH_KEY_CAMERA_1_IMAGE", "CAMERA_1_IMAGE")
    key_cam2 = env_str("ZENOH_KEY_CAMERA_2_IMAGE", "CAMERA_2_IMAGE")
    key_agent_chat = env_str("ZENOH_KEY_AGENT_CHAT", "AGENT_CHAT")

    agent_logs_pub = z.declare_publisher(key_agent_logs)
    cam1_pub = z.declare_publisher(key_cam1)
    cam2_pub = z.declare_publisher(key_cam2)

    agent_chat_q = z.declare_queryable(
        key_expr=key_agent_chat,
        handler=zenoh.handlers.RingChannel(capacity=100),
    )

    robot_index = env_int("ROBOT_INDEX", 0)
    actions_per_chunk = env_int("ACTIONS_PER_CHUNK", 10)
    pretrained_name_or_path = env_str(
        "PRETRAINED_NAME_OR_PATH", "rtsmc/smolvla_box_in_bin_so101_test"
    )
    policy_type = env_str("POLICY_TYPE", "smolvla")
    camera_1_name = env_str("CAMERA_1_NAME", "front")
    camera_2_name = env_str("CAMERA_2_NAME", "wrist")
    calibration = load_calibration()

    server_address = env_str("LEROBOT_SERVER_ADDRESS", "").strip()
    if not server_address:
        host = env_str("LEROBOT_HOST", "").strip()
        port = env_str("LEROBOT_PORT", "").strip()
        if host and port:
            server_address = f"{host}:{port}"
    if not server_address:
        raise ValueError(
            "Missing LEROBOT_SERVER_ADDRESS (or LEROBOT_HOST + LEROBOT_PORT)"
        )

    cameras = discover_cameras_from_env()

    def on_observation_callback(observation: Observation):
        if not observation:
            return

        def publish_frame(pub: zenoh.Publisher, frame_rgb: np.ndarray):
            try:
                # lerobot Observation frames are typically RGB; encode wants BGR.
                bgr = frame_rgb[..., ::-1]
                ret, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ret:
                    return
                pub.put(jpeg.tobytes())
            except Exception as e:
                logger.error(f"Error sending image: {e}")

        if camera_1_name in observation:
            publish_frame(cam1_pub, observation[camera_1_name])
        if camera_2_name in observation:
            publish_frame(cam2_pub, observation[camera_2_name])

    def on_action_callback(action):
        pass

    def get_next_task() -> str:
        try:
            q = agent_chat_q.recv()
            return q.payload.to_bytes().decode("utf-8")
        except Exception as e:
            logger.error(f"Error receiving task: {e}")
            return ""

    camera_paths = {}
    if camera_1_name in cameras:
        camera_paths[camera_1_name] = cameras[camera_1_name].dev_path
    if camera_2_name in cameras:
        camera_paths[camera_2_name] = cameras[camera_2_name].dev_path

    robot_config = get_so100_policy_config(
        server_address=server_address,
        actions_per_chunk=actions_per_chunk,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        index=robot_index,
        camera_paths=camera_paths,
    )

    CustomRobotClient.run_robot_client(
        get_next_task=get_next_task,
        robot_config=robot_config,
        on_action_callback=on_action_callback,
        on_observation_callback=on_observation_callback,
        calibration=calibration,
    )


def run_teleop():
    from app.teleop import teleoperate

    rec = init_rerun_viewer_server()
    zenoh_endpoints = parse_zenoh_connect_endpoints(
        env_str("ZENOH_CONNECT_ENDPOINTS", "")
    )
    if not zenoh_endpoints:
        raise ValueError("Missing/empty ZENOH_CONNECT_ENDPOINTS")

    z = make_zenoh_session(zenoh_endpoints)

    key_image = env_str("ZENOH_KEY_IMAGE", "IMAGE")
    image_pub = z.declare_publisher(key_image)

    robot_index = env_int("ROBOT_INDEX", 0)
    calibration = load_calibration()
    image_change_threshold = env_float("IMAGE_CHANGE_THRESHOLD", 0.05)
    only_publish_image_on_change = env_bool("ONLY_PUBLISH_IMAGE_ON_CHANGE", True)

    cam1_dev = env_str("CAMERA_1_DEV", "").strip()
    if not cam1_dev:
        raise ValueError("Missing CAMERA_1_DEV for teleop mode")

    change_publisher = ImageChangePublisher(
        image_change_threshold=image_change_threshold,
        only_publish_image_on_change=only_publish_image_on_change,
        publisher=image_pub,
    )

    teleoperate(
        camera_paths={"gripper": cam1_dev},
        index=robot_index,
        calibration=calibration,
        on_new_image=change_publisher.on_new_image,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("TELEOP", None) is None:
        run_policy_controlled()
    else:
        run_teleop()
