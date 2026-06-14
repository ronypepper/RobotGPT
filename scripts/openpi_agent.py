# Based on code from the Isaac Lab project:
# https://github.com/isaac-sim/IsaacLab
#
# Original work:
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# Modifications:
# Copyright (c) 2026 ronypepper.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="OpenPi client for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--robot_type", type=str, default=None, help="Robot type - must be an option in OPENPI_INTERFACE_FCTS of source/RobotGPT/RobotGPT/utils/robots/openpi_interfaces.py.")
parser.add_argument("--record_scene", action="store_true", default=False, help="Record videos of the scene.")
parser.add_argument("--record_table", action="store_true", default=False, help="Record videos of the table camera observations.")
parser.add_argument("--record_wrists", action="store_true", default=False, help="Record videos of the wrist camera observations.")
parser.add_argument("--annotate", action="store_true", default=False, help="Create an annotations.yaml file with success/fail and episode length information. An episode termination means success, while a truncation (time-out) a failure.")
parser.add_argument("--output_dir", type=str, default=None, help="Videos and annotations will be saved to this path.")

# openpi-specific arguments
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to perform.")
parser.add_argument("--max_duration", type=float, default=0.0, help="Overrides default maximum episode length in seconds.")
parser.add_argument("--open_loop_horizon", type=int, default=16, help="Number of actions to execute from a prediction before re-querying the policy.")
parser.add_argument("--remote_host", type=str, default="0.0.0.0", help="IP address of the policy server.")
parser.add_argument("--remote_port", type=int, default=8000, help="Port of the policy server.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# enable_cameras needs to be on for the camera sensors
args_cli.enable_cameras = True

# arguments check
if args_cli.task is None:
    raise ValueError("task must be set.")
if args_cli.robot_type is None:
    raise ValueError("robot_type must be set.")
if (args_cli.record_scene or args_cli.record_table or args_cli.record_wrists or args_cli.annotate) and args_cli.output_dir is None:
    raise ValueError("output_dir must be specified when record_scene, record_table, record_wrists or annotate is set.")
if args_cli.num_rollouts <= 0:
    raise ValueError("num_rollouts must be greater than 0.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import signal

import gymnasium as gym
import numpy as np
import RobotGPT.tasks  # noqa: F401
import torch
import tqdm
import yaml
from moviepy import ImageSequenceClip
from openpi_client import websocket_client_policy
from RobotGPT.utils.robots.openpi_interfaces import get_openpi_interface_fcts

import omni.ui as ui

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Get openpi observation/action interface functions
process_observation_for_openpi_fct, process_openpi_action_fct = get_openpi_interface_fcts(args_cli.robot_type)


# UI window providing rollout controls
class RolloutControlsUI:
    def __init__(self):
        self.rollouts_stopped = False
        self.rollout_paused = False
        self.rollout_skipped = False

        self._window = ui.Window(
            "Rollout Controls",
            width=300,
            height=200
        )

        with self._window.frame:
            with ui.VStack(spacing=5):
                ui.Button("Pause / Resume", clicked_fn=self._toggle_pause)
                ui.Button("Skip Rollout", clicked_fn=self._skip_rollout)
                ui.Button("Stop Rollouts", clicked_fn=self._stop_rollouts)

    def _toggle_pause(self):
        self.rollout_paused = not self.rollout_paused
        print("[INFO] Rollout paused." if self.rollout_paused else "[INFO] Rollout resumed.")

    def _skip_rollout(self):
        self.rollout_skipped = True
        print("[INFO] Rollout skipped.")

    def _stop_rollouts(self):
        self.rollouts_stopped = True
        print("[INFO] Rollouts stopped.")


# Create UI window
rollout_controls_ui = RolloutControlsUI()


def main():
    """"OpenPi client for Isaac Lab environment."""
    # Create output directory
    if args_cli.output_dir:
        os.makedirs(args_cli.output_dir, exist_ok=True)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.max_duration > 0.0:
        env_cfg.episode_length_s = args_cli.max_duration
    env_fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)

    # create environment
    print("[INFO]: Creating environment.")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record_scene else None)

    # wrap environment for video recording of scene
    if args_cli.record_scene:
        video_kwargs = {
            "video_folder": args_cli.output_dir,
            "name_prefix": "scene",
            "episode_trigger": lambda episode: True # record every episode
        }
        print("[INFO] Recording scene videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Check if prompt has been specified
    if not hasattr(env_cfg, "prompt"):
        raise ValueError("No prompt has been specified in the environment.")
    print(f"[INFO]: Prompt: {env_cfg.prompt}")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Connect to the policy server
    print(f"[INFO]: Attempting to connect to policy server at {args_cli.remote_host}:{args_cli.remote_port}...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(args_cli.remote_host, args_cli.remote_port)
    print("[INFO]: Connected to policy server.")

    # simulate environment
    num_success, num_failure, num_invalid, durations = 0, 0, 0, []
    rollout_num = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # reset environment (also after last episode to save its video)
            env_obs, _ = env.reset()
            print("[INFO]: Episode was reset.")

            # Rollout parameters
            actions_from_chunk_completed = 0
            pred_action_chunk = None

            # Prepare to (potentially) save videos of camera observations
            wrist_cam_video, table_cam_video = [], []

            bar = tqdm.tqdm(range(int(env_cfg.episode_length_s / (env_cfg.sim.dt * float(env_cfg.decimation)))))
            rollout_num += 1
            print(f"[INFO]: Starting rollout {rollout_num}/{args_cli.num_rollouts}...")
            for t_step in bar:
                while rollout_controls_ui.rollout_paused and simulation_app.is_running():
                    simulation_app.update()

                # Check if app has been closed
                if not simulation_app.is_running():
                    break

                obs = extract_numpy_observation(env_obs)

                # Save camera observations for video recordings
                if args_cli.record_table:
                    table_cam_video.append(obs["table_img"])
                if args_cli.record_wrists:
                    wrist_cam_video.append(obs["wrist_img"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args_cli.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # Transform observation data to format expected by policy server
                    policy_server_obs = process_observation_for_openpi_fct(obs, env_cfg.prompt)

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(policy_server_obs)["actions"]

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Transform action data to format expected by environment
                action = process_openpi_action_fct(action)
                action = torch.tensor(action[np.newaxis], dtype=torch.float32, device=args_cli.device)

                # Step environment
                env_obs, _, terminated, truncated, _ = env.step(action)

                # Check if episode has ended
                if terminated[0]:
                    print("[INFO]: Episode has terminated.")
                    num_success += 1
                    durations.append((t_step + 1) * env_cfg.sim.dt * env_cfg.decimation)
                    break
                if truncated[0]:
                    print("[INFO]: Episode timed out.")
                    num_failure += 1
                    break
                if rollout_controls_ui.rollout_skipped:
                    rollout_controls_ui.rollout_skipped = False
                    num_invalid += 1
                    break
                if rollout_controls_ui.rollouts_stopped:
                    break

            # Save camera observation videos to disk
            if args_cli.record_table:
                filename = os.path.join(args_cli.output_dir, "table-cam-episode-" + str(rollout_num - 1)) + ".mp4"
                ImageSequenceClip(table_cam_video, fps=env_fps).write_videofile(filename, codec="libx264", logger=None)
            if args_cli.record_wrists:
                filename = os.path.join(args_cli.output_dir, "wrist-cam-episode-" + str(rollout_num - 1)) + ".mp4"
                ImageSequenceClip(wrist_cam_video, fps=env_fps).write_videofile(filename, codec="libx264", logger=None)

            if rollout_num >= args_cli.num_rollouts or rollout_controls_ui.rollouts_stopped:
                break

    # Save annotations to disk
    if args_cli.annotate:
        avg_duration = 0.0
        if num_success > 0:
            avg_duration = sum(durations) / num_success

        success_percentage = 0.0
        if num_success + num_failure > 0:
            success_percentage = num_success / (num_success + num_failure) * 100.0

        annotations = {
            "num_demos_total": num_success + num_failure + num_invalid,
            "num_success": num_success,
            "num_failure": num_failure,
            "num_invalid": num_invalid,
            "success_percentage": success_percentage,
            "average_duration_s": avg_duration,
        }

        with open(os.path.join(args_cli.output_dir, "annotations.yaml"), "w") as f:
            yaml.dump(annotations, f, sort_keys=False)

    # close the simulator
    env.close()


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def extract_numpy_observation(env_obs: dict):
    if "wrist_img" in env_obs["policy"]:
        obs = {
            "joint_pos": env_obs["policy"]["joint_pos"][0].detach().cpu().numpy(),
            "table_img": env_obs["policy"]["table_img"][0].detach().cpu().numpy(),
            "wrist_img": env_obs["policy"]["wrist_img"][0].detach().cpu().numpy(),
        }
    else:
        obs = {
            "left_joint_pos": env_obs["policy"]["left_joint_pos"][0].detach().cpu().numpy(),
            "right_joint_pos": env_obs["policy"]["right_joint_pos"][0].detach().cpu().numpy(),
            "table_img": env_obs["policy"]["table_img"][0].detach().cpu().numpy(),
            "left_wrist_img": env_obs["policy"]["left_wrist_img"][0].detach().cpu().numpy(),
            "right_wrist_img": env_obs["policy"]["right_wrist_img"][0].detach().cpu().numpy(),
        }
    return obs


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
