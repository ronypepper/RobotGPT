#!/usr/bin/env bash
set -euo pipefail  # Stricter error handling

# Parse named flags
TRAIN_CONFIG=""
HF_USER=""
DATASET=""
RESUME=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for --config"; exit 1; }
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --hf_user)
            [[ $# -ge 2 ]] || { echo "Missing value for --hf_user"; exit 1; }
            HF_USER="$2"
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || { echo "Missing value for --dataset"; exit 1; }
            DATASET="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift 1
            ;;
        --overwrite)
            OVERWRITE=1
            shift 1
            ;;
        --help|-h)
            cat << EOF
Usage:
  $0 --config <CONFIG> --dataset <NAME>

Required arguments:
  --config          Name of the openpi TrainConfig to use for training
  --hf_user         Hugging Face username of the dataset used for training in robotgpt_output/datasets/lerobot/
  --dataset         Name of the dataset to train on in robotgpt_output/datasets/lerobot/{hf_user}/

Optional:
  --resume          Resume a previous training run
  --overwrite       Overwrite a previous training run
  -h, --help        Show this help message

Example:
  $0 --config robotgpt_franka_single_arm_pi05_lora --hf_user YOUR_HF_USERNAME --dataset dataset_1
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Try: $0 --help"
            exit 1
            ;;
    esac
done

# Check if all required named flags were specified
if [[ -z "$TRAIN_CONFIG" ]]; then
    echo "--config is required"
    exit 1
fi
if [[ -z "$DATASET" ]]; then
    echo "--dataset is required"
    exit 1
fi
if [[ -z "$HF_USER" ]]; then
    echo "--hf_user is required"
    exit 1
fi

# Parse --resume and --overwrite flags
if [[ "$RESUME" -eq 1 && "$OVERWRITE" -eq 1 ]]; then
    echo "--resume and --overwrite flags are exclusive"
    exit 1
fi
EXISTING_CHECKPOINT_FLAG=""
if [[ "$RESUME" -eq 1 ]]; then
    EXISTING_CHECKPOINT_FLAG="--resume"
elif [[ "$OVERWRITE" -eq 1 ]]; then
    EXISTING_CHECKPOINT_FLAG="--overwrite"
fi

# Change cwd to shell script directory regardless of where script was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------------------------------------------------

# Change to robotgpt_output to get assets, checkpoints and wandb outputs saved there
cd ../../robotgpt_output

# Prepare environment
source ../openpi/.venv/bin/activate
export HF_LEROBOT_HOME="$(pwd)/datasets/lerobot"

# Compute normalization statistics if they don't already exist
ASSET_PATH="assets/${TRAIN_CONFIG}/${HF_USER}/${DATASET}/norm_stats.json"
REPO_ID="${HF_USER}/${DATASET}"
if [ -e "$ASSET_PATH" ]; then
    echo "Normalization statistics already exist at: robotgpt_output/${ASSET_PATH}"
    echo "Skipping computation."
else
    echo "Computing normalization statistics..."
    python -m robotgpt.compute_norm_stats "$TRAIN_CONFIG" --data.repo_id "$REPO_ID" --exp-name "$DATASET"
    echo "Normalization statistics saved to: robotgpt_output/${ASSET_PATH}"
fi

# Train an openpi model
echo "Starting training..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python ../openpi/scripts/train.py "$TRAIN_CONFIG" \
--data.repo_id "$REPO_ID" --exp-name "$DATASET" $EXISTING_CHECKPOINT_FLAG
echo "Training completed."