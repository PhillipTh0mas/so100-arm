#!/usr/bin/env python3
"""
SO-101 ARM URDF Visualizer with Rerun - class-based, yourdfpy only.
Loads URDF once, logs meshes once, then only updates transforms.
"""

import rerun as rr
import numpy as np
import time
from typing import Dict, Optional, Union
from copy import deepcopy
from pathlib import Path
import logging
import yourdfpy as urdf_lib

logger = logging.getLogger(__name__)


class SO101Visualizer:
    def __init__(self, urdf_path: str, entity_prefix: str = "robot"):
        self.urdf_path = urdf_path
        self.entity_prefix = entity_prefix

        # Load URDF once
        self.robot = urdf_lib.URDF.load(
            urdf_path, build_scene_graph=True, load_meshes=True
        )
        self.scene = self.robot.scene
        self.logged_meshes = False
        self.last_mesh_log = 0.0  # never logged
        self.last_image = None
        self.last_joint_pos = None

    def _log_meshes_once(self):
        """Log all meshes once with default transforms (identity).
        Will re-log at most once every 30s.
        """
        # throttle: only log again if 30s have passed
        if self.logged_meshes and (time.time() - self.last_mesh_log < 120):
            return

        for node in self.scene.graph.nodes:
            if node in self.scene.geometry:
                mesh = self.scene.geometry[node]
                if mesh is None:
                    continue

                entity_path = f"{self.entity_prefix}/{node}/mesh"

                try:
                    vertices = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    mesh_component = rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_normals=mesh.vertex_normals
                        if hasattr(mesh, "vertex_normals")
                        else None,
                    )

                    if (
                        hasattr(mesh.visual, "material")
                        and hasattr(mesh.visual.material, "baseColorFactor")
                    ):
                        mesh_component.albedo_factor = mesh.visual.material.baseColorFactor

                    rr.log(entity_path, mesh_component)
                    self.logged_meshes = True
                except Exception as e:
                    logger.error(f"Could not log mesh for {node}: {e}")
        self.last_mesh_log = time.time()

    def update_joint_positions(self, joint_positions: Dict[str, float]):
        """Update transforms for the robot based on new joint positions."""
        self._log_meshes_once()

        og_joints = deepcopy(joint_positions)

        try:
            if "wrist_roll" in joint_positions:
                joint_positions["wrist_roll"] += np.pi / 2  # 90 degrees adjustment for visualization
            self.robot.update_cfg(joint_positions)
        except Exception as e:
            logger.error(f"Error updating joint positions: {e}")
            return

        if self.last_joint_pos is not None:
            delta = max([abs(v - self.last_joint_pos.get(k, 0)) for k, v in joint_positions.items()])
            if delta < 0.01:
                return

        self.log_joint_states(joint_positions=og_joints)

        self.last_joint_pos = joint_positions

        # Log updated transforms only
        for node in self.scene.graph.nodes:
            transform = (
                self.scene.graph.get(node)[0].copy()
                if self.scene.graph.get(node)
                else np.eye(4)
            )

            rr.log(
                f"{self.entity_prefix}/{node}",
                rr.Transform3D(
                    translation=transform[:3, 3],
                    mat3x3=transform[:3, :3],
                ),
            )

    def log_joint_states(self, joint_positions: Dict[str, float]):
        """Log joint states as scalars."""        
        for joint_name, position in joint_positions.items():
            rr.log(
                f"{self.entity_prefix}/joint_states/{joint_name}/position",
                rr.Scalars(position),
            )
            rr.log(
                f"{self.entity_prefix}/joint_states/{joint_name}/position_deg",
                rr.Scalars(np.degrees(position)),
            )

    def log_camera_at_gripper(
        self,
        camera_image: Optional[Union[str, np.ndarray]] = None,
        image_width: int = 640,
        image_height: int = 480,
    ):
        """Log a virtual camera at the gripper link."""

        camera_entity = f"{self.entity_prefix}/wrist_roll_follower_so101_v1.stl/camera"

        # Camera intrinsics
        focal_length = min(image_width, image_height) * 0.8
        intrinsics = rr.Pinhole(
            width=image_width, height=image_height, focal_length=focal_length
        )

        # Estimated offset of the camera mount
        translation = [0.0, 0.05, -0.02]
        pitch_rad = np.radians(-10)  # Pitch down 10°
        roll_rad = np.radians(180)   # Roll 180°

        R_pitch = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)],
            ]
        )

        R_roll = np.array(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad),  np.cos(roll_rad), 0],
                [0,                0,                1],
            ]
        )

        rotation = R_roll @ R_pitch

        if camera_image is not None:
            if self.last_image is not None:
                if not frame_changed(last=self.last_image, current=camera_image):
                    return
            self.last_image = camera_image
            rr.log(
                camera_entity,
                rr.Transform3D(translation=translation, mat3x3=rotation, from_parent=True),
            )
            rr.log(camera_entity, intrinsics)
            rr.log(camera_entity, rr.Image(camera_image))


def frame_changed(last, current, threshold=0.05):
    # shrink to 16x16 grayscale
    gray_last = last.mean(axis=2)[::last.shape[0]//16, ::last.shape[1]//16]
    gray_curr = current.mean(axis=2)[::current.shape[0]//16, ::current.shape[1]//16]
    
    gray_last = gray_last / 255.0
    gray_curr = gray_curr / 255.0

    diff = np.abs(gray_last - gray_curr).mean()
    return diff > threshold
