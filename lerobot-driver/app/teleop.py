import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.robots import Robot
from lerobot.robots.so100_follower import (
    SO100FollowerEndEffectorConfig,
)
from lerobot.teleoperate import TeleoperateConfig
from lerobot.teleoperators import Teleoperator

from app.mcp_teleop import MCPEndEffectorTeleop, MCPTeleopConfig
from app.so100_autodetect import find_so100_port, get_camera_info
from app.so100_fpv_follower import SO100FPVFollower

logger = logging.getLogger(__name__)


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    on_new_image: Optional[Callable[[np.ndarray], None]] = None,
):
    logger.info("Starting teleop loop")
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
        # busy_wait(1 / fps - dt_s)
        # actual sleep
        time.sleep(max(0.1, (1 / fps) - dt_s))

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
    port = find_so100_port()

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
    cfg = TeleoperateConfig(robot=robot_cfg, teleop=teleop.config, fps=60)

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
