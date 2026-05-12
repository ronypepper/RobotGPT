# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
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

# openpi-specific arguments
parser.add_argument("--prompt", type=str, default=None, help="Prompt for the policy.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to perform.")
parser.add_argument("--max_timesteps", type=int, default=0, help="Maximum number of timesteps to take. 0 means infinite.")
parser.add_argument("--open_loop_horizon", type=int, default=8, help="Number of actions to execute from a prediction before re-querying the policy.")
parser.add_argument("--remote_host", type=str, default="0.0.0.0", help="IP address of the policy server.")
parser.add_argument("--remote_port", type=int, default=8000, help="Port of the policy server.")
parser.add_argument("--save_video", type=bool, default=False, help="If the table cam video should be saved to disk.")
parser.add_argument("--save_stats", type=bool, default=False, help="If results statistics should be saved to disk.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import datetime
import os
import signal
from itertools import count

import gymnasium as gym
import numpy as np
import pandas as pd
import RobotGPT.tasks  # noqa: F401
import torch
import tqdm
from moviepy import ImageSequenceClip
from openpi_client import image_tools, websocket_client_policy

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """"OpenPi client for Isaac Lab environment."""
    if args_cli.prompt is None:
        raise ValueError("No prompt has been specified.")
    if args_cli.num_rollouts <= 0:
        raise ValueError("num_rollouts must be greater than 0.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    print("[INFO]: Creating environment.")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Connect to the policy server
    print(f"[INFO]: Attempting to connect to policy server at {args_cli.remote_host}:{args_cli.remote_port}...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(args_cli.remote_host, args_cli.remote_port)
    print("[INFO]: Connected to policy server.")

    # Data frame for result info
    if args_cli.save_stats:
        stats = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # simulate environment
    rollout_num = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # # compute zero actions
            # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # # apply actions
            # env.step(actions)

            # reset environment
            env_obs, _ = env.reset()

            # Rollout parameters
            actions_from_chunk_completed = 0
            pred_action_chunk = None

            # Prepare to save video of rollout
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            video = []

            if args_cli.max_timesteps > 0:
                bar = tqdm.tqdm(range(args_cli.max_timesteps))
            else:
                bar = tqdm.tqdm(count(), total=None)
            rollout_num += 1
            print(f"[INFO]: Starting rollout {rollout_num}/{args_cli.num_rollouts}... press Ctrl+C to stop early.")
            for t_step in bar:
                # Check if app has been closed
                if not simulation_app.is_running():
                    break

                try:
                    obs = extract_numpy_observation(env_obs)

                    # Save video frame
                    if args_cli.save_video:
                        video.append(obs["table_img"])

                    # Send websocket request to policy server if it's time to predict a new chunk
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args_cli.open_loop_horizon:
                        actions_from_chunk_completed = 0

                        # Transform observation data to format expected by policy server
                        policy_server_obs = franka_to_droid_obs(obs)

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
                    env_obs, _, _, _, _ = env.step(action)

                    # # Sleep to match DROID data collection frequency
                    # elapsed_time = time.time() - start_time
                    # if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    #     time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                except KeyboardInterrupt:
                    break

            # Save video to disk
            if args_cli.save_video and simulation_app.is_running():
                video = np.stack(video)
                save_filename = "video_" + timestamp
                ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

            if args_cli.save_stats and simulation_app.is_running():
                # Query user for rollout success rating
                success: str | float | None = None
                while not isinstance(success, float):
                    success = input(
                        "[INPUT]: Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
                    )
                    if success == "y":
                        success = 1.0
                    elif success == "n":
                        success = 0.0
                    else:
                        success = float(success) / 100

                    if not (0 <= success <= 1):
                        print(f"[INPUT]: Success must be a number in [0, 100] but got: {success * 100}")
                        success = None

                # Save statistics
                stats = stats.append(
                    {
                        "success": success,
                        "duration": t_step,
                        "video_filename": save_filename,
                    },
                    ignore_index=True,
                )

            if rollout_num >= args_cli.num_rollouts:
                break

    # Store result statistics
    if args_cli.save_stats:
        os.makedirs("openpi/results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
        csv_filename = os.path.join("openpi/results", f"eval_{timestamp}.csv")
        stats.to_csv(csv_filename)
        print(f"[INFO]: Results saved to {csv_filename}")

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

def franka_to_droid_obs(obs: dict):
    # Resize images here to minimize the amount of data sent to the policy server and improve latency.
    # Proprioceptive state normalization is handled on the server side.
    policy_server_obs = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["table_img"], 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["wrist_img"], 224, 224)
        ),
        "observation/joint_position": obs["joint_pos"][:7],
        "observation/gripper_position": obs["joint_pos"][8],
        "prompt": args_cli.prompt,
    }
    return policy_server_obs

def droid_to_franka_action(action: np.array):
    return np.append(action, action[-1])


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
