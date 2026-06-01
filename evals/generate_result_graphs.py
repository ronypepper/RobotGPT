"""
Copyright (c) 2026 ronypepper.
License: BSD-3-Clause
"""

import dataclasses
import os

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import tyro
import yaml

# Set IEEE matplotlib format
plt.style.use(['science', 'ieee', 'no-latex'])

import matplotlib.pyplot as plt


@dataclasses.dataclass
class Args:
    """Arguments for the result graph generation script."""

    # Path to root directory containing evaluation directories (designated by prefix)
    root_dir: str = "robotgpt_output/evaluation/robotgpt_franka_single_arm_pi05_lora"

    # Prefix of directories containing evaluation metrics for one experiment over multiple datasets and checkpoints
    prefix: str = "place_cube_in_bin_franka"

    # Batch size used for training
    batch_size: int = 4


def parse_evaluation_directory(dir_name: str, args: Args):
    eval_dir = os.path.join(args.root_dir, dir_name)
    if not os.path.isdir(eval_dir):
        return None
    if not dir_name.startswith(args.prefix + "_") or not dir_name.endswith("_demos"):
        return None
    num_demos = dir_name[len(args.prefix) + 1:][:-len("_demos")]
    if not str.isnumeric(num_demos):
        return None
    num_demos = int(num_demos)

    eval_dir_entries = os.listdir(eval_dir)
    checkpoints, yaml_paths = [], []
    for entry in eval_dir_entries:
        checkpoint_dir = os.path.join(eval_dir, entry)
        if os.path.isdir(checkpoint_dir) and str.isnumeric(entry):
            yaml_path = os.path.join(checkpoint_dir, "annotations.yaml")
            if not os.path.exists(yaml_path):
                print(f"Warning: Checkpoint directory {checkpoint_dir} does not contain an \"annotations.yaml\" file.")
            else:
                checkpoints.append(int(entry))
                yaml_paths.append(yaml_path)

    if len(checkpoints) == 0:
        return None

    # Sort checkpoints
    sorted_pairs = sorted(zip(checkpoints, yaml_paths))
    checkpoints, yaml_paths = zip(*sorted_pairs)

    return {
        "num_demos" : num_demos,
        "checkpoints" : checkpoints,
        "yaml_paths" : yaml_paths
    }


def main(args: Args):
    args.root_dir = os.path.abspath(args.root_dir)
    if not os.path.exists(args.root_dir):
        raise ValueError(f"Root directory \"{args.root_dir}\" does not exist.")

    # Collect evaluation files
    dir_entries = sorted(os.listdir(args.root_dir), reverse=True)
    evaluations = []
    for entry in dir_entries:
        result = parse_evaluation_directory(entry, args)
        if result:
            evaluations.append(result)

    if len(evaluations) == 0:
        raise ValueError(f"No valid evaluation directories matching prefix \"{args.prefix}\" in root directory: \"{args.root_dir}\".")

    # Load evaluation data
    for eval in evaluations:
        success_percentages, avg_durations = [], []
        for yaml_path in eval["yaml_paths"]:
            with open(yaml_path) as file:
                annotations = yaml.safe_load(file)
                success_percentages.append(annotations["success_percentage"])
                avg_durations.append(annotations["average_duration_s"])
        eval["success_percentages"] = success_percentages
        eval["avg_durations"] = avg_durations

    # Generate success percentage graph
    plt.figure(figsize=(4, 2.5))
    x_ticks = set()
    for eval in evaluations:
        plt.plot(eval["checkpoints"], eval["success_percentages"], label=str(eval["num_demos"]) + " demos")
        x_ticks.update(eval["checkpoints"])
    plt.xticks(list(x_ticks))
    plt.xlabel(f"Training steps @ batch_size = {args.batch_size}")
    plt.ylabel("Success rate [%]")
    plt.grid(True)
    plt.legend(loc="lower right", frameon=True)
    plt.show()

    # Generate average duration graph
    plt.figure(figsize=(4, 2.5))
    x_ticks = set()
    for eval in evaluations:
        plt.plot(eval["checkpoints"], eval["avg_durations"], label=str(eval["num_demos"]) + " demos")
        x_ticks.update(eval["checkpoints"])
    plt.xticks(list(x_ticks))
    plt.xlabel(f"Training steps @ batch_size = {args.batch_size}")
    plt.ylabel("Average duration [sec]")
    plt.grid(True)
    plt.legend(loc="lower center", frameon=True)
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
