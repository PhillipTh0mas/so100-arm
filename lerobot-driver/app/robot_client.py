import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Dict, Callable, Optional

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.helpers import (
    Action,
    Observation,
)

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig

from lerobot.scripts.server.robot_client import RobotClient
import pickle
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
import grpc

from app.so100_autodetect import find_so100_port, get_camera_info


class CustomRobotClient(RobotClient):

    def __init__(self, config: RobotClientConfig,
                 on_observation_callback: Callable[[Observation], None],
                 on_action_callback: Callable[[Dict], None],
                 calibration: Optional[Dict] = None,
                 ):
        if calibration:
            calibration_file = (
                config.robot.calibration_dir if config.robot.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / "so100_follower" / (config.robot.id + ".json")
            )
            calibration_file.parent.mkdir(parents=True, exist_ok=True)
            with calibration_file.open("w") as f:
                json.dump(calibration, f, indent=4)

        super().__init__(config)
        self.on_observation_callback = on_observation_callback
        self.on_action_callback = on_action_callback
        self.pause = threading.Event()
        self.current_task = config.task

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                if not self.pause.is_set():
                    self.logger.info("Control loop paused, waiting for resume...")
                    self.pause.wait()

                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def custom_control_loop(self, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            if not self.pause.is_set():
                self.logger.info("Control loop paused, waiting for resume...")
                self.pause.wait()

            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                self.on_action_callback(_performed_action)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(self.current_task, verbose)
                self.on_observation_callback(_captured_observation)

            #self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action

    def pause_control_loop(self):
        """Pause the control loop"""
        self.pause.clear()
        self.logger.info("Control loop paused")

    def continue_control_loop(self):
        """Resume the control loop"""
        self.pause.set()
        self.logger.info("Control loop resumed")

    @staticmethod
    def run_robot_client(
            robot_config: RobotClientConfig,
            on_observation_callback: Callable[[Observation], None],
            on_action_callback: Callable[[Dict], None],
            get_next_task: Callable[[], str],
            calibration: Optional[Dict] = None,
    ):

        logging.basicConfig(level=logging.INFO)
        logging.info("Starting RobotClient with config:\n%s", pformat(asdict(robot_config)))

        # 5) Instantiate and start
        client = CustomRobotClient(
            config=robot_config,
            on_observation_callback=on_observation_callback,
            on_action_callback=on_action_callback,
            calibration=calibration
        )
        client.pause_control_loop()
        #  client.policy_config.lerobot_features = {k.replace("observation.images.image", "observation.image"): v for k, v in client.policy_config.lerobot_features.items()}
        if not client.start():
            raise RuntimeError("Failed to start RobotClient!")

        # 6) Start thread to receive actions
        recv_thread = threading.Thread(target=client.receive_actions, daemon=True)
        recv_thread.start()

        control_loop = threading.Thread(
            target=client.custom_control_loop,
            daemon=True
        )
        control_loop.start()

        run = True
        try:
            while run:
                # Check for new task.. blocking wait
                new_task = get_next_task()
                if new_task == "exit" or new_task == "quit" or new_task == "stop":
                    client.pause_control_loop()
                else:
                    client.current_task = new_task
                    client.continue_control_loop()
        finally:
            client.stop()
            recv_thread.join()
            control_loop.join()
            logging.info("Robot client stopped.")


def get_so100_policy_config(
        server_address: str,
        camera_paths: Dict[str, str],
        task: str = "",
        index: int = 0,
        policy_type: str = "smolvla",
        pretrained_name_or_path: str = "helper2424/smolvla_rtx_movet",
        policy_device: str = "cpu",
        actions_per_chunk: int = 10,
) -> "RobotClientConfig":
    from lerobot.scripts.server.configs import RobotClientConfig
    # 1) Build the robot & camera config
    port = find_so100_port(index=index)

    cameras = dict()
    for name, path in camera_paths.items():
        w, h, fps = get_camera_info(index_or_path=path)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(path),
            width=w, height=h, fps=fps
        )

    robot_cfg = SO100FollowerConfig(
        port=port,
        id=f"so100-{index}",
        cameras=cameras
    )

    # 4) Build the full client config
    cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=server_address,
        task=task,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        policy_device=policy_device,
        actions_per_chunk=actions_per_chunk,
        debug_visualize_queue_size=False,
        verify_robot_cameras=False,
    )
    return cfg
