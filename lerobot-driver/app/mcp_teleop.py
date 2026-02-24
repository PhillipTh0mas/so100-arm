import enum
import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict

import rerun as rr
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from mcp.server import FastMCP

from app.idle_behavior import IdleBehaviorConfig, IdleBehaviorController

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("mcp")
@dataclass
class MCPTeleopConfig(TeleoperatorConfig):
    port: int = 9988


class ArmPreset(enum.Enum):
    rest = "rest"
    top_down = "top_down"
    work_hover = "work_hover"


class MCPEndEffectorTeleop(Teleoperator):
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
        server = FastMCP(name="image_describer", host="0.0.0.0", port=config.port)
        self._server_thread = None
        self.server = server

        # idle behavior
        self.idle = IdleBehaviorController(IdleBehaviorConfig())
        self._last_manual_ts = time.time()

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
            logger.info("move_arm_vector")

            # mark activity
            self._last_manual_ts = time.time()
            self.idle.on_user_activity()

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
                    "gripper": gripper + 1,
                    "delta_pitch": delta_pitch * 90,
                    "delta_yaw": delta_yaw * 90,
                    "delta_roll": delta_roll * 90,
                }
            )
            time.sleep(1)
            return "Move arm command successfully sent"

    def _drain_pressed_keys(self):
        while not self.delta_queue.empty():
            vals = self.delta_queue.get_nowait()
            self.current_pressed.append(vals)

    def update_idle_image(self):
        try:
            observation = self.robot.get_observation()
            if "gripper" in observation:
                # your observation seems RGB; convert to BGR for cv2/deepface pipeline
                rgb = observation["gripper"]
                bgr = rgb[..., ::-1].copy()
                self.idle.update_image(bgr)
                box = self.idle.get_face_box()
                if box is not None:
                    x, y, w, h = box
                    # Rerun Boxes2D wants [xmin, ymin, xmax, ymax]
                    # # todo.. do this in tobot visualizer
                    rr.log(
                        "robot/wrist_roll_follower_so101_v1.stl/camera/face",
                        rr.Boxes2D(mins=[x, y], sizes=[w, h]),
                    )
                else:
                    rr.log(
                        "robot/wrist_roll_follower_so101_v1.stl/camera/face",
                        rr.Boxes2D(mins=[0, 0], sizes=[0, 0]),
                    )
        except Exception:
            pass

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

        # keep the latest image flowing to idle controller (cheap; deepface runs throttled)
        self.update_idle_image()

        # manual commands take priority
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

        if self.current_pressed:
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

        # if no input, idle behavior may generate subtle motion
        idle_action = self.idle.tick()
        return idle_action

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def run_mcp_server(self, **kwargs):
        try:
            self.server.run(**kwargs)
        except Exception as e:
            print(f"Error running MCP server: {e}")

    def connect(self, calibrate: bool = True) -> None:
        # self.server.run(transport="streamable-http")
        # run server in thread
        if not self.is_connected:
            thr = threading.Thread(
                target=self.run_mcp_server,
                kwargs={"transport": "streamable-http"},
                daemon=True,
            )
            thr.start()
            self._server_thread = thr

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self.is_connected:
            if self._server_thread is not None:
                self._server_thread.join(timeout=1)
                self._server_thread = None
