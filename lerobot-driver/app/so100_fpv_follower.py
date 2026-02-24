# app/so100_fpv_follower.py
import logging
import math
import time
from typing import Any

import numpy as np
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.so100_follower import (
    SO100FollowerEndEffector,
    SO100FollowerEndEffectorConfig,
)

from app.robot_logging import SO101Visualizer

logger = logging.getLogger(__name__)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n)


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = float(math.cos(float(angle_rad)))
    s = float(math.sin(float(angle_rad)))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if float(np.linalg.norm(v)) < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-12:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _normalize(axis)
        return _axis_angle(axis, math.pi)

    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


class SO100FPVFollower(SO100FollowerEndEffector):
    """
    Safety in world/base coordinates:
      - clamp position to end_effector_bounds with margin
      - clamp orientation so camera view direction (EE +Z) stays:
          * near horizontal ("cylinder wall")
          * within a forward cone (not looking behind)
    """

    def __init__(self, config: SO100FollowerEndEffectorConfig):
        super().__init__(config=config)
        self.visualizer = SO101Visualizer(urdf_path=config.urdf_path)
        self.visualizer._log_meshes_once()
        self.last_log_time = time.time()

        self.bounds_margin_xyz = np.array([0.03, 0.03, 0.02], dtype=np.float64)
        self.view_max_tilt_deg = 12.0
        self.view_max_yaw_deg = 35.0

        # Camera looks along EE +Z in your setup
        self.view_axis_ee = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # base +Z
        self.world_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # base +X

    def get_ee_pose(self) -> np.ndarray | None:
        return (
            None
            if self.current_ee_pos is None
            else np.asarray(self.current_ee_pos, dtype=np.float64).copy()
        )

    def _clamp_desired_position(self, desired_ee: np.ndarray) -> np.ndarray:
        if self.end_effector_bounds is None:
            return desired_ee

        bmin = (
            np.asarray(self.end_effector_bounds["min"], dtype=np.float64)
            + self.bounds_margin_xyz
        )
        bmax = (
            np.asarray(self.end_effector_bounds["max"], dtype=np.float64)
            - self.bounds_margin_xyz
        )

        desired_ee = np.asarray(desired_ee, dtype=np.float64)
        desired_ee[:3, 3] = np.clip(desired_ee[:3, 3], bmin, bmax)
        return desired_ee

    def _apply_view_constraints(self, desired_ee: np.ndarray) -> np.ndarray:
        desired_ee = np.asarray(desired_ee, dtype=np.float64)
        R = desired_ee[:3, :3].copy()

        up = _normalize(self.world_up)
        fwd = _normalize(self.world_forward)
        view_dir = _normalize(R @ self.view_axis_ee)

        # 1) cylinder-wall band: keep view close to horizontal (dot(view, up) near 0)
        max_tilt = math.radians(float(self.view_max_tilt_deg))
        dot_up = float(np.clip(np.dot(view_dir, up), -1.0, 1.0))

        cos_min = math.cos((math.pi / 2) + max_tilt)
        cos_max = math.cos((math.pi / 2) - max_tilt)
        dot_up_clamped = float(np.clip(dot_up, cos_min, cos_max))

        if abs(dot_up - dot_up_clamped) > 1e-6:
            horiz = view_dir - dot_up * up
            if float(np.linalg.norm(horiz)) < 1e-12:
                horiz = np.cross(up, fwd)
            horiz = _normalize(horiz)

            target = _normalize(
                horiz * math.sqrt(max(0.0, 1.0 - dot_up_clamped * dot_up_clamped))
                + up * dot_up_clamped
            )

            R_fix = _rotation_from_a_to_b(view_dir, target)
            R = R_fix @ R
            view_dir = _normalize(R @ self.view_axis_ee)

        # 2) forward cone: keep view within max yaw around world_forward
        min_dot_fwd = math.cos(math.radians(float(self.view_max_yaw_deg)))
        if float(np.dot(view_dir, fwd)) < min_dot_fwd:
            v_h = view_dir - float(np.dot(view_dir, up)) * up
            f_h = fwd - float(np.dot(fwd, up)) * up

            if (
                float(np.linalg.norm(v_h)) > 1e-12
                and float(np.linalg.norm(f_h)) > 1e-12
            ):
                v_h = _normalize(v_h)
                f_h = _normalize(f_h)

                cross = np.cross(v_h, f_h)
                sign = 1.0 if float(np.dot(cross, up)) >= 0.0 else -1.0
                ang = math.acos(float(np.clip(np.dot(v_h, f_h), -1.0, 1.0)))
                ang_to_boundary = ang - math.radians(float(self.view_max_yaw_deg))

                R_fix = _axis_angle(up, sign * ang_to_boundary)
                R = R_fix @ R

        desired_ee[:3, :3] = R
        return desired_ee

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        step = self.config.end_effector_step_sizes
        sx, sy, sz = step.get("x", 1.0), step.get("y", 1.0), step.get("z", 1.0)
        sp, syaw, sr = (
            step.get("pitch", 0.5),
            step.get("yaw", 0.5),
            step.get("roll", 0.5),
        )

        # Keep your existing sign for pitch; yaw sign may need flipping depending on camera
        delta_pitch = float(action.get("delta_pitch", 0.0)) * sp * -1.0
        delta_yaw = float(action.get("delta_yaw", 0.0)) * syaw
        delta_roll = float(action.get("delta_roll", 0.0)) * sr

        grip_raw = action.get("gripper", 1.0)
        try:
            grip_val = float(np.asarray(grip_raw).ravel()[-1])
        except Exception:
            logger.warning(f"Invalid gripper value {grip_raw!r}; defaulting to 1.0")
            grip_val = 1.0

        if self.current_joint_pos is None:
            present = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array(
                [present[name] for name in self.bus.motors], dtype=np.float64
            )

        if self.current_ee_pos is None:
            self.current_ee_pos = np.asarray(
                self.kinematics.forward_kinematics(self.current_joint_pos),
                dtype=np.float64,
            )

        cur = np.asarray(self.current_ee_pos, dtype=np.float64)

        # translation in local frame (keep as you had)
        grip_delta = np.eye(4, dtype=np.float64)
        grip_delta[:3, 3] = np.array(
            [
                -float(action.get("delta_z", 0.0)) * sx,
                float(action.get("delta_y", 0.0)) * sy,
                float(action.get("delta_x", 0.0)) * sz,
            ],
            dtype=np.float64,
        )

        # --- World-correct view rotations (recommended) --------------------------
        R_cur = cur[:3, :3].copy()
        up = _normalize(self.world_up)
        view_dir = _normalize(R_cur @ self.view_axis_ee)

        right = np.cross(up, view_dir)
        if float(np.linalg.norm(right)) < 1e-12:
            right = np.cross(up, _normalize(self.world_forward))
        right = _normalize(right)

        yaw_rad = math.radians(float(delta_yaw))
        pitch_rad = math.radians(float(delta_pitch))
        roll_rad = math.radians(float(delta_roll))

        R_delta_world = (
            _axis_angle(up, yaw_rad)
            @ _axis_angle(right, pitch_rad)
            @ _axis_angle(view_dir, roll_rad)
        )

        desired_ee = cur.copy()
        desired_ee[:3, :3] = R_delta_world @ R_cur
        desired_ee[:3, 3] = (cur @ grip_delta)[:3, 3]

        desired_ee = self._clamp_desired_position(desired_ee)
        desired_ee = self._apply_view_constraints(desired_ee)

        target_deg = self.kinematics.inverse_kinematics(
            self.current_joint_pos, desired_ee
        )

        joint_action = {
            f"{name}.pos": float(target_deg[i])
            for i, name in enumerate(self.bus.motors.keys())
        }
        joint_action["gripper.pos"] = float(
            np.clip(
                float(self.current_joint_pos[-1])
                + (grip_val - 1.0) * float(self.config.max_gripper_pos),
                5.0,
                float(self.config.max_gripper_pos),
            )
        )

        self.current_ee_pos = desired_ee.copy()
        self.current_joint_pos = np.asarray(target_deg, dtype=np.float64).copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        return super(SO100FollowerEndEffector, self).send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        observation = super().get_observation()
        if "gripper" in observation and time.time() - self.last_log_time > 1.0:
            image = observation["gripper"]
            joint_positions = {
                k.replace(".pos", ""): float(math.radians(float(v)))
                for k, v in observation.items()
                if k != "gripper"
            }
            self.visualizer.update_joint_positions(joint_positions)
            self.visualizer.log_camera_at_gripper(image)
            self.last_log_time = time.time()
        return observation
