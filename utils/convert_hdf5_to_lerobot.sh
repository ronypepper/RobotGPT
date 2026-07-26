#!/usr/bin/env bash
set -euo pipefail  # Stricter error handling

# Parse named flags
HF_USER=""
DATASET=""
ROBOT_TYPE=""
OUTPUT_NAME=""
NUM_EPISODES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --robot_type)
            [[ $# -ge 2 ]] || { echo "Missing value for --robot_type"; exit 1; }
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --output_name)
            [[ $# -ge 2 ]] || { echo "Missing value for --output_name"; exit 1; }
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --num_episodes)
            [[ $# -ge 2 ]] || { echo "Missing value for --num_episodes"; exit 1; }
            NUM_EPISODES="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
Required arguments:
  --hf_user         Hugging Face username of dataset creator
  --dataset         Name of the HDF5 dataset stored in robotgpt_output/datasets/hdf5/
  --robot_type      Robot type of the dataset (must be one of the defined values in openpi/robotgpt/convert_hdf5_to_openpi_lerobot.py)

Optional:
  --output_name     Name of the converted output dataset. If not specified, name of the input dataset will be used.
  --num_episodes    Number of epsiodes from the input dataset to convert. If 0 (the default), all episodes will be used.
  -h, --help        Show this help message

Example:
  $0 --hf_user YOUR_HF_USERNAME --dataset dataset_1 --robot_type franka_single_arm
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
if [[ -z "$HF_USER" ]]; then
    echo "--hf_user is required"
    exit 1
fi
if [[ -z "$DATASET" ]]; then
    echo "--dataset is required"
    exit 1
fi
if [[ -z "$ROBOT_TYPE" ]]; then
    echo "--robot_type is required"
    exit 1
fi

# Change cwd to shell script directory regardless of where script was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------------------------------------------------

# Change to common root directory of RobotGPT project
cd ../..

# Convert dataset
echo "Starting conversion..."
source openpi/.venv/bin/activate
export HF_LEROBOT_HOME="$(pwd)/robotgpt_output/datasets/lerobot"
python -m robotgpt.convert_hdf5_to_openpi_lerobot --dataset "robotgpt_output/datasets/hdf5/${DATASET}.hdf5" \
--hf_user "$HF_USER" --robot_type "$ROBOT_TYPE" --output_name "$OUTPUT_NAME" --num_episodes "$NUM_EPISODES"

echo "Conversion completed."