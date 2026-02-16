import base64
import enum
import json
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import mcp
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.motors import MotorCalibration
from lerobot.robots import Robot
from lerobot.robots.so100_follower import (
    SO100FollowerEndEffector,
    SO100FollowerEndEffectorConfig,
)
from lerobot.teleoperate import TeleoperateConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from mcp.server.fastmcp.utilities.types import Image

from app.so100_autodetect import find_so100_port, get_camera_info
from app.so100_fpv_follower import SO100FPVFollower


@TeleoperatorConfig.register_subclass("mcp")
@dataclass
class MCPTeleopConfig(TeleoperatorConfig):
    port: int = 9988


class ArmPreset(enum.Enum):
    rest = "rest"
    top_down = "top_down"
    work_hover = "work_hover"


class MCPEndEffectorTeleop(Teleoperator):
    """
    Teleop class to use mcp inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = MCPTeleopConfig
    name = "mcp"

    def __init__(self, config: MCPTeleopConfig, robot: Robot):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.robot = robot
        self.delta_queue = Queue()
        self.current_pressed = []
        self.logs = {}
        server = mcp.server.FastMCP(
            name="image_describer", host="0.0.0.0", port=config.port
        )
        self._server_thread = None
        self.server = server

        self._goal_lock = threading.Lock()
        self._goal_xyz: Optional[Tuple[float, float, float]] = None
        self._pos_tol = 0.005  # 5 mm
        self._max_step_per_tick = 0.25  # maps error meters → [-1, 1] deltas

        self.last_delta = None
        self.random_move_time = time.time()

        @server.tool(
            description=(
                "Move the robotic arm using normalized delta values in the range [-1, 1].\n\n"
                "Translation parameters:\n"
                "- delta_forward > 0 moves forward.\n"
                "- delta_backward > 0 moves backward.\n"
                "- delta_left > 0 moves left.\n"
                "- delta_right > 0 moves right.\n"
                "- delta_up > 0 moves up.\n"
                "- delta_down > 0 moves down.\n\n"
                "Rotation parameters (end effector position stays the same, only orientation changes):\n"
                "- rotate_left > 0 rotates left (yaw).\n"
                "- rotate_right > 0 rotates right (yaw).\n"
                "- rotate_up > 0 tilts upward (pitch).\n"
                "- rotate_down > 0 tilts downward (pitch).\n"
                "- roll_left > 0 rolls counter-clockwise.\n"
                "- roll_right > 0 rolls clockwise.\n\n"
                "Gripper parameter:\n"
                "- gripper_open ∈ [0, 1], 1 to open it!\n"
                "- gripper_close ∈ [0, 1], 1 to close it!\n\n"
                "Examples:\n"
                "- To move right, set delta_right=0.5.\n"
                "- To rotate the wrist 90° to the right in place, set rotate_right=1.0.\n"
                "- To tilt the gripper upward without moving, set rotate_up=1.0.\n"
                "- To roll the camera clockwise, set roll_right=1.0.\n\n"
                "If all parameters are 0.0, nothing happens."
            )
        )
        def move_arm_vector(
            # translations
            delta_forward: float = 0.0,
            delta_backward: float = 0.0,
            delta_left: float = 0.0,
            delta_right: float = 0.0,
            delta_up: float = 0.0,
            delta_down: float = 0.0,
            # gripper
            gripper_open: float = 0.0,
            gripper_close: float = 0.0,
            # rotations
            rotate_up: float = 0.0,
            rotate_down: float = 0.0,
            rotate_left: float = 0.0,
            rotate_right: float = 0.0,
            roll_left: float = 0.0,
            roll_right: float = 0.0,
        ) -> str:
            # normalize to internal deltas
            delta_x = delta_forward - delta_backward
            delta_y = delta_left - delta_right
            delta_z = delta_up - delta_down
            delta_pitch = rotate_up - rotate_down
            delta_yaw = rotate_right - rotate_left
            delta_roll = roll_right - roll_left
            gripper = gripper_open - gripper_close

            if all(
                v == 0.0
                for v in [
                    delta_x,
                    delta_y,
                    delta_z,
                    gripper,
                    delta_pitch,
                    delta_yaw,
                    delta_roll,
                ]
            ):
                return "No values passed, nothing happened."

            self.delta_queue.put(
                {
                    "delta_x": delta_x,
                    "delta_y": delta_y,
                    "delta_z": delta_z,
                    "gripper": gripper + 1,  # gripper in [0, 2]
                    "delta_pitch": delta_pitch * 90,
                    "delta_yaw": delta_yaw * 90,
                    "delta_roll": delta_roll * 90,
                }
            )
            time.sleep(1)

            return "Move arm command successfully sent"

        # @server.tool(description="Get the current image from the robot's gripper camera as jpeg bytes.")
        # def get_gripper_image() -> Image:
        #    observation = self.robot.get_observation()
        #    if "gripper" in observation:
        #        image = observation["gripper"]
        #        jpegimg = cv2.imencode(".jpg", image[...,::-1])[1].tobytes()
        #    else:
        #        jpegimg = b""
        #    return Image(data=jpegimg, format="jpeg")

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        # self.server.run(transport="streamable-http")
        # run server in thread
        if not self.is_connected:
            thr = threading.Thread(
                target=self.server.run,
                kwargs={"transport": "streamable-http"},
                daemon=True,
            )
            thr.start()
            self._server_thread = thr

    def calibrate(self) -> None:
        pass

    def _drain_pressed_keys(self):
        while not self.delta_queue.empty():
            vals = self.delta_queue.get_nowait()
            self.current_pressed.append(vals)

        if time.time() - self.random_move_time > 1.5 and False:
            if self.last_delta:
                self.delta_queue.put(
                    {
                        "delta_x": -self.last_delta["delta_x"],
                        "delta_y": -self.last_delta["delta_y"],
                        "delta_z": -self.last_delta["delta_z"],
                        "gripper": 0.0,
                        "delta_pitch": -self.last_delta.get("delta_pitch", 0.0),
                        "delta_yaw": -self.last_delta.get("delta_yaw", 0.0),
                        "delta_roll": -self.last_delta.get("delta_roll", 0.0),
                    }
                )
            delta = {
                "delta_x": random.random() * 0.01,
                "delta_y": random.random() * 0.01,
                "delta_z": random.random() * 0.01,
                "gripper": 0.0,
                "delta_pitch": random.random() * 90,
                "delta_yaw": random.random() * 90,
                "delta_roll": random.random() * 15,
            }
            self.delta_queue.put(delta)
            self.last_delta = delta
            self.random_move_time = time.time()

    def configure(self):
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self.is_connected:
            if self._server_thread is not None:
                self._server_thread.join(timeout=1)
                self._server_thread = None

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {
                "delta_x": 0,
                "delta_y": 1,
                "delta_z": 2,
                "gripper": 3,
                "delta_pitch": 4,
                "delta_yaw": 5,
                "delta_roll": 5,
            },
        }

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        action_dict = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
            "delta_roll": 0.0,
        }
        for val in self.current_pressed:
            action_dict["delta_x"] += val["delta_x"]
            action_dict["delta_y"] += val["delta_y"]
            action_dict["delta_z"] += val["delta_z"]
            action_dict["gripper"] = val["gripper"]
            action_dict["delta_pitch"] += val["delta_pitch"]
            action_dict["delta_yaw"] += val["delta_yaw"]
            action_dict["delta_roll"] += val["delta_roll"]
        self.current_pressed.clear()
        return action_dict


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    on_new_image: Optional[Callable[[np.ndarray], None]] = None,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()

        if on_new_image:
            try:
                observation = robot.get_observation()
                if "gripper" in observation:
                    image = observation["gripper"]
                    on_new_image(image[..., ::-1])
            except Exception as e:
                print(e)
                pass

        try:
            robot.send_action(action)
        except Exception as e:
            print(e)
            pass
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        # print("\n" + "-" * (display_len + 10))
        # print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        # for motor, value in action.items():
        #    print(f"{motor:<{display_len}} | {value:>7.2f}")
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


def teleoperate(
    camera_paths: Dict[str, str],
    index: int = 0,
    calibration: Optional[Dict] = None,
    on_new_image: Optional[Callable[[np.ndarray], None]] = None,
):
    port = find_so100_port(index=index)

    cameras = dict()
    for name, path in camera_paths.items():
        w, h, fps = get_camera_info(index_or_path=path)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(path), width=w, height=h, fps=fps
        )

    robot_cfg = SO100FollowerEndEffectorConfig(
        port=port,
        id=f"so100-{index}",
        cameras=cameras,
        end_effector_step_sizes={
            "x": 0.05,
            "y": 0.05,
            "z": 0.05,
        },
        end_effector_bounds={
            "min": [
                -0.40,
                -0.50,
                0.02,
            ],  # X/Y limits over your table # 2 cm above the surface to avoid scraping
            "max": [0.40, 0.60, 0.45],  # and a sane Z-max
        },
        urdf_path=str(Path(__file__).parent.absolute() / "so101_new_calib.urdf"),
    )
    if calibration:
        calibration_file = (
            HF_LEROBOT_CALIBRATION / ROBOTS / "so100_follower" / f"{robot_cfg.id}.json"
        )
        robot_cfg.calibration_dir = calibration_file.parent
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with calibration_file.open("w") as f:
            json.dump(calibration, f, indent=4)

    robot = SO100FPVFollower(config=robot_cfg)
    teleop = MCPEndEffectorTeleop(config=MCPTeleopConfig(), robot=robot)

    teleop.connect()
    robot.connect(calibrate=False)
    # just use default
    cfg = TeleoperateConfig(robot=robot_cfg, teleop=teleop.config)

    try:
        teleop_loop(
            teleop,
            robot,
            cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            on_new_image=on_new_image,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # if cfg.display_data:
        #     rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
