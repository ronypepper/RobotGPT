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
parser.add_argument("--video", action="store_true", default=False, help="Record videos of the rollouts.")
parser.add_argument("--video_obs", action="store_true", default=False, help="Record videos of the table and wrist camera observations.")

# openpi-specific arguments
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to perform.")
parser.add_argument("--max_timesteps", type=int, default=0, help="Maximum number of timesteps to take. 0 means infinite.")
parser.add_argument("--open_loop_horizon", type=int, default=16, help="Number of actions to execute from a prediction before re-querying the policy.")
parser.add_argument("--remote_host", type=str, default="0.0.0.0", help="IP address of the policy server.")
parser.add_argument("--remote_port", type=int, default=8000, help="Port of the policy server.")

# logging-specific arguments
parser.add_argument("--policy_name", type=str, default=None, help="Name of the policy.")
parser.add_argument("--policy_checkpoint", type=int, default=None, help="Checkpoint number of the policy (== training steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# enable_cameras needs to be on for the camera sensors
args_cli.enable_cameras = True

# arguments check
if args_cli.task is None:
    raise ValueError("task must be set.")
if args_cli.policy_name is None:
    raise ValueError("policy_name must be set.")
if args_cli.policy_checkpoint is None:
    raise ValueError("policy_name must be set.")
if args_cli.num_rollouts <= 0:
    raise ValueError("num_rollouts must be greater than 0.")



# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import signal
from itertools import count

import gymnasium as gym
import numpy as np
import RobotGPT.tasks  # noqa: F401
import torch
import tqdm
from moviepy import ImageSequenceClip
from openpi_client import image_tools, websocket_client_policy

import omni.ui as ui

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# UI window providing rollout controls
class RolloutControlsUI:
    def __init__(self):
        self.rollouts_stopped = False
        self.rollout_paused = False
        self.rollout_completed = False

        self._window = ui.Window(
            "Rollout Controls",
            width=300,
            height=200
        )
        # self._window.

        with self._window.frame:
            with ui.VStack(spacing=5):
                ui.Button("Pause / Resume", clicked_fn=self._toggle_pause)
                ui.Button("Complete Rollout", clicked_fn=self._complete_rollout)
                ui.Button("Stop Rollouts", clicked_fn=self._stop_rollouts)

    def _toggle_pause(self):
        self.rollout_paused = not self.rollout_paused
        print("[INFO] Rollout paused." if self.rollout_paused else "[INFO] Rollout resumed.")

    def _complete_rollout(self):
        self.rollout_completed = True
        print("[INFO] Rollout completed.")

    def _stop_rollouts(self):
        self.rollouts_stopped = True
        print("[INFO] Rollouts stopped.")


# Create UI window
rollout_controls_ui = RolloutControlsUI()


# import subprocess

# venv_python = "/path/to/other/venv/bin/python"
# script_path = "/path/to/target_script.py"

# p = subprocess.Popen([
#     venv_python,
#     script_path,
#     "arg1",
#     "arg2"
# ])

# print("Started process:", p.pid)

# # main script continues immediately

# p.terminate()  # ask it to stop gracefully
# p.wait()

# p.kill()       # force kill
# p.wait()


def main():
    """"OpenPi client for Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )

    video_dir = os.path.join("robotgpt_videos", args_cli.task, args_cli.policy_name, str(args_cli.policy_checkpoint))

    # create environment
    print("[INFO]: Creating environment.")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap environment for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": video_dir,
            "name_prefix": "video",
            "episode_trigger": lambda episode: True # record every episode
        }
        print("[INFO] Recording videos during training.")
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

            # Prepare to save videos of camera observations
            wrist_cam_video, table_cam_video = [], []

            if args_cli.max_timesteps > 0:
                bar = tqdm.tqdm(range(args_cli.max_timesteps))
            else:
                bar = tqdm.tqdm(count(), total=None)
            rollout_num += 1
            print(f"[INFO]: Starting rollout {rollout_num}/{args_cli.num_rollouts}...")
            for t_step in bar:
                while rollout_controls_ui.rollout_paused and simulation_app.is_running():
                    simulation_app.update()

                # Check if app has been closed
                if not simulation_app.is_running():
                    break

                obs = extract_numpy_observation(env_obs)

                # Save camera observations for video
                if args_cli.video_obs:
                    wrist_cam_video.append(obs["wrist_img"])
                    table_cam_video.append(obs["table_img"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args_cli.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # Transform observation data to format expected by policy server
                    policy_server_obs = franka_to_droid_obs(obs, env_cfg.prompt)

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(policy_server_obs)["actions"]

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Transform action data to format expected by environment
                action = droid_to_franka_action(action)
                action = torch.tensor(action[np.newaxis], dtype=torch.float32, device=args_cli.device)

                # Step environment
                env_obs, _, terminated, truncated, _ = env.step(action)

                # Check if episode has ended
                if terminated:
                    print("[INFO]: Episode was terminated.")
                    break
                if truncated:
                    print("[INFO]: Episode timed out.")
                    break
                if rollout_controls_ui.rollout_completed:
                    rollout_controls_ui.rollout_completed = False
                    break
                if rollout_controls_ui.rollouts_stopped:
                    break

            # Save camera observation videos to disk
            if args_cli.video_obs:
                env_fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
                filename = os.path.join(video_dir, "wrist-cam-episode-" + str(rollout_num - 1)) + ".mp4"
                ImageSequenceClip(wrist_cam_video, fps=env_fps).write_videofile(filename, codec="libx264", logger=None)
                filename = os.path.join(video_dir, "table-cam-episode-" + str(rollout_num - 1)) + ".mp4"
                ImageSequenceClip(table_cam_video, fps=env_fps).write_videofile(filename, codec="libx264", logger=None)

            if rollout_num >= args_cli.num_rollouts or rollout_controls_ui.rollouts_stopped:
                break

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
    obs = {
        "table_img": env_obs["policy"]["table_img"][0].detach().cpu().numpy(),
        "wrist_img": env_obs["policy"]["wrist_img"][0].detach().cpu().numpy(),
        "joint_pos": env_obs["policy"]["joint_pos"][0].detach().cpu().numpy()
    }
    return obs

def franka_to_droid_obs(obs: dict, prompt: str):
    # Pi0 models are trained for gripper positions in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
    # Observations in the dataset are in [0.0, 0.04], with 0.0 corresponding to fully closed and 0.04 corresponding to fully open.
    # Therefore we adjust the gripper observation to fit the Pi0 models' format.
    # For received actions (later), we don't need to do the reverse transformation since the environment expects this format for the gripper action as well.
    # Proprioceptive state normalization is handled on the server side.
    state_obs = obs["joint_pos"][:8] # 7 joints + 1 gripper
    joint_pos_obs = state_obs[0:7]
    gripper_obs = (state_obs[7:8] - 0.04) / 0.04
    state_obs = np.concatenate((joint_pos_obs, gripper_obs))
    policy_server_obs = {
        "observation/table_img": obs["table_img"],
        "observation/wrist_img": obs["wrist_img"],
        "observation/joint_pos": state_obs,
        "prompt": prompt,
    }
    return policy_server_obs

# def franka_to_droid_obs(obs: dict, prompt: str):
#     # Resize images here to minimize the amount of data sent to the policy server and improve latency.
#     # Proprioceptive state normalization is handled on the server side.
#     policy_server_obs = {
#         "observation/table_img": image_tools.convert_to_uint8(
#             image_tools.resize_with_pad(obs["table_img"], 224, 224)
#         ),
#         "observation/wrist_img": image_tools.convert_to_uint8(
#             image_tools.resize_with_pad(obs["wrist_img"], 224, 224)
#         ),
#         "observation/joint_pos": obs["joint_pos"][:8], # 7 joints + 1 gripper
#         "prompt": prompt,
#     }
#     return policy_server_obs

def polaris_franka_to_droid_obs(obs: dict, prompt: str):
    # Resize images here to minimize the amount of data sent to the policy server and improve latency.
    # Proprioceptive state normalization is handled on the server side.
    policy_server_obs = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["table_img"], 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["wrist_img"], 224, 224)
        ),
        "observation/joint_position": obs["joint_pos"][:7], # 7 joints
        "observation/gripper_position": obs["joint_pos"][7],# 1 gripper
        "prompt": prompt,
    }
    return policy_server_obs

def droid_to_franka_action(action: np.array):
    return np.append(action, action[-1])


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
