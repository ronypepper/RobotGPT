#!/usr/bin/env bash
set -euo pipefail  # Stricter error handling

# Parse named flags
TASK=""
DATASET=""
NUM_DEMOS=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            [[ $# -ge 2 ]] || { echo "Missing value for --task"; exit 1; }
            TASK="$2"
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || { echo "Missing value for --dataset"; exit 1; }
            DATASET="$2"
            shift 2
            ;;
        --num_demos)
            [[ $# -ge 2 ]] || { echo "Missing value for --num_demos"; exit 1; }
            NUM_DEMOS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift 1
            ;;
        --help|-h)
            cat << EOF
Usage:
  $0 --task <TASK> --dataset <NAME> --num_demos <N>

Required arguments:
  --task            Isaac Lab task name (remember to use the 'IK-Abs' variant)
  --dataset         Name of the dataset to save in robotgpt_output/datasets/hdf5/
  --num_demos       Number of demonstrations to collect

Optional:
  --verbose         Print logs and error messages
  -h, --help        Show this help message

Example:
  $0 --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 --dataset dataset_1 --num_demos 25
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
if [[ -z "$TASK" ]]; then
    echo "--task is required"
    exit 1
fi
if [[ -z "$DATASET" ]]; then
    echo "--dataset is required"
    exit 1
fi
if [[ -z "$NUM_DEMOS" ]]; then
    echo "--num_demos is required"
    exit 1
fi

# Set Isaac output stream based on --verbose flag
if [[ "$VERBOSE" -eq 1 ]]; then
    OUT_STREAM="/dev/stdout"
else
    OUT_STREAM="/dev/null"
fi

# Setup spawned process cleanup
BG_PIDS=()
cleanup() {
    if ((${#BG_PIDS[@]})); then
        echo "Killing ${#BG_PIDS[@]} background process(es)..."

        for pid in "${BG_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done

        echo "Waiting for ${#BG_PIDS[@]} background process(es) to stop..."

        for pid in "${BG_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        echo "All background processes stopped."
    else
        echo "No background processes have been started."
    fi
}
trap cleanup EXIT

# Change cwd to shell script directory regardless of where script was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------------------------------------------------

# Change to common root directory of RobotGPT project
cd ../..

# Handle dataset file already existing
DATASET_PATH="$(pwd)/robotgpt_output/datasets/hdf5/${DATASET}.hdf5"
if [ -e "$DATASET_PATH" ]; then
    printf '"%s" already exists. Overwrite? [y/N] ' "$DATASET_PATH"
    read answer

    case "$answer" in
        [Yy]|[Yy][Ee][Ss])
            rm -r -v "$DATASET_PATH"
            ;;
        *)
            echo "Aborted."
            exit 1
            ;;
    esac
fi

# Start CloudXR server
(
    echo "Starting CloudXR server..."
    source env_isaaclab/bin/activate
    python -m isaacteleop.cloudxr --accept-eula --cloudxr-env-config=RobotGPT/dev/quest3_cloudxr.env &> "$OUT_STREAM" # Last argument enables optical hand tracking
) &
BG_PIDS+=("$!")

# Start task simulation with teleop and recording
echo "Starting simulation and task collection..."
source env_isaaclab/bin/activate
source ~/.cloudxr/run/cloudxr.env
python RobotGPT/scripts/record_demos.py --task "$TASK" \
--enable_cameras --device cpu --teleop_device motioncontroller --xr \
--dataset_file "$DATASET_PATH" --num_demos "$NUM_DEMOS" &> "$OUT_STREAM"
echo "Task collection completed."