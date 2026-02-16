import logging
from typing import Any, Callable
import time
import math
import numpy as np
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.so100_follower import SO100FollowerEndEffector, SO100FollowerEndEffectorConfig
from app.robot_logging import SO101Visualizer


logger = logging.getLogger(__name__)

class SO100FPVFollower(SO100FollowerEndEffector):
    def __init__(self, config: SO100FollowerEndEffectorConfig):
        super().__init__(config=config)
        self.visualizer = SO101Visualizer(urdf_path=config.urdf_path)
        self.last_log_time = time.time()


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        step = self.config.end_effector_step_sizes
        sx, sy, sz = step.get("x", 1.0), step.get("y", 1.0), step.get("z", 1.0)
        sp, syaw, sr = step.get("pitch", 0.5), step.get("yaw", 0.5), step.get("roll", 0.5)  # deg per unit


        # Parse angular deltas (local yaw then local pitch)
        delta_pitch = float(action.get("delta_pitch", 0.0)) * sp * -1
        delta_roll = float(action.get("delta_roll", 0.0)) * sr
        delta_yaw   = float(action.get("delta_yaw", 0.0)) * syaw

        # Gripper: accept scalar or array
        grip_raw = action.get("gripper", 1.0)
        try:
            grip_val = float(np.asarray(grip_raw).ravel()[-1])
        except Exception:
            logger.warning(f"Invalid gripper value {grip_raw!r}; defaulting to 1.0")
            grip_val = 1.0

        # Initialize state if needed
        if self.current_joint_pos is None:
            present = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([present[name] for name in self.bus.motors])
        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos)


        grip_delta = np.eye(4)
        grip_delta[:3, 3] = np.array([-action["delta_z"]*sx, action["delta_y"]*sy, action["delta_x"]*sz], dtype=np.float32)
        grip_delta[:3, :3] = _rot_x(delta_yaw) @ _rot_y(delta_pitch) @ _rot_z(delta_roll)
        desired_ee = np.eye(4, dtype=np.float32)
        desired_ee = self.current_ee_pos @ grip_delta

        #R = _rot_z(delta_yaw) @ _rot_y(delta_pitch) @ _rot_x(delta_roll)
        #grip_delta = np.eye(4, dtype=np.float32)
        #grip_delta[:3, 3] = np.array([-action["delta_z"]*sx, action["delta_y"]*sy, -action["delta_x"]*sz], dtype=np.float32)
        #if self.end_effector_bounds is not None:
        #    grip_delta[:3, 3] = np.clip(grip_delta[:3, 3], self.end_effector_bounds["min"], self.end_effector_bounds["max"])
        #grip_delta[:3, :3] = R
        #desired_ee = self.current_ee_pos @ grip_delta

        if self.end_effector_bounds is not None:
            grip_delta[:3, 3] = np.clip(grip_delta[:3, 3], self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Local orientation update: R_des = R_cur @ Rz(yaw) @ Ry(pitch)
        

        target_deg = self.kinematics.inverse_kinematics(self.current_joint_pos, desired_ee)

        joint_action = {
            f"{name}.pos": target_deg[i]
            for i, name in enumerate(self.bus.motors.keys())
        }
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (grip_val - 1.0) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        self.current_ee_pos = desired_ee.copy()
        self.current_joint_pos = target_deg.copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        return super(SO100FollowerEndEffector, self).send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        observation = super().get_observation()
        if "gripper" in observation and time.time() - self.last_log_time > 1.0:
            image = observation["gripper"]
            joint_positions={k.replace(".pos", ""): math.radians(v) for k, v in observation.items() if k != "gripper"}
            self.visualizer.update_joint_positions(joint_positions)
            self.visualizer.log_camera_at_gripper(image)
            self.last_log_time = time.time()
        return observation

def _rot_y(pitch: float) -> np.ndarray:
    pitch = np.deg2rad(pitch)
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float32)

def _rot_x(yaw: float) -> np.ndarray:
    yaw = np.deg2rad(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[1, 0, 0],
                    [0, c,-s],
                    [0, s, c]], dtype=np.float32)

def _rot_z(a: float) -> np.ndarray:
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=np.float32)