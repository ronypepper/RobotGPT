#!/usr/bin/env bash
set -euo pipefail  # Stricter error handling

# Parse named flags
TASK=""
ROBOT_TYPE=""
TRAIN_CONFIG=""
HF_USER=""
DATASET=""
CHECKPOINT=""
NUM_ROLLOUTS=10
MAX_DURATION=0
AUTO_ANNOTATE=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            [[ $# -ge 2 ]] || { echo "Missing value for --task"; exit 1; }
            TASK="$2"
            shift 2
            ;;
        --robot_type)
            [[ $# -ge 2 ]] || { echo "Missing value for --robot_type"; exit 1; }
            ROBOT_TYPE="$2"
            shift 2
            ;;
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
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint"; exit 1; }
            CHECKPOINT="$2"
            shift 2
            ;;
        --num_rollouts)
            [[ $# -ge 2 ]] || { echo "Missing value for --num_rollouts"; exit 1; }
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --max_duration)
            [[ $# -ge 2 ]] || { echo "Missing value for --max_duration"; exit 1; }
            MAX_DURATION="$2"
            shift 2
            ;;
        --auto_annotate)
            AUTO_ANNOTATE=1
            shift 1
            ;;
        --verbose)
            VERBOSE=1
            shift 1
            ;;
        --help|-h)
            cat << EOF
Usage:
  $0 --task <TASK> --robot_type <ROBOT_TYPE> --dataset <NAME> --num_demos <N>

Required arguments:
  --task            Isaac Lab task name (remember to use the 'Pos' variant)
  --robot_type      Name of the robot used in the task. Must have openpi interfaces defined.
  --config          Name of the openpi TrainConfig used for training the policy
  --hf_user         Hugging Face username of the dataset used for training in robotgpt_output/datasets/lerobot/
  --dataset         Name of the dataset used for training in robotgpt_output/datasets/lerobot/{hf_user}/
  --checkpoint      Checkpoint of the policy to load from robotgpt_output/checkpoints/{config}/{dataset}/

Optional:
  --num_rollouts    Number of rollouts to evaluate (default is 10)
  --max_duration    Overrides maximum episode length in seconds if specified
  --auto_annotate   Do not save table cam videos and directly generate annotation.yaml for evaluation results. Requires an environment that defines success termination terms
  --verbose         Print logs and error messages
  -h, --help        Show this help message

Example:
  $0 --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--robot_type franka_single_arm \
--config robotgpt_franka_single_arm_pi05_lora \
--hf_user YOUR_HF_USERNAME --dataset dataset_1 --checkpoint 5000
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
if [[ -z "$ROBOT_TYPE" ]]; then
    echo "--robot_type is required"
    exit 1
fi
if [[ -z "$TRAIN_CONFIG" ]]; then
    echo "--config is required"
    exit 1
fi
if [[ -z "$HF_USER" ]]; then
    echo "--hf_user is required"
    exit 1
fi
if [[ -z "$DATASET" ]]; then
    echo "--dataset is required"
    exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
    echo "--checkpoint is required"
    exit 1
fi

# Set output stream based on --verbose flag
if [[ "$VERBOSE" -eq 1 ]]; then
    OUT_STREAM="/dev/stdout"
    KIT_ARGS=()
else
    OUT_STREAM="/dev/null"
    KIT_ARGS=(
        --kit_args
        "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"
    )
fi

# Setup spawned process cleanup
BG_PIDS=()
cleanup() {
    local count=${#BG_PIDS[@]}
    if (($count)); then
        echo "Killing $count background process(es)..."

        # Grafecul shutdown
        for pid in "${BG_PIDS[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done

        echo "Waiting for $count background process(es) to stop..."
        
        # Wait up to max 5 seconds per process for graceful shutdown
        for pid in "${BG_PIDS[@]}"; do
            for _ in {1..10}; do
                if kill -0 "$pid" 2>/dev/null; then
                    sleep 0.5
                else
                    break
                fi
            done
        done

        # Force kill
        for pid in "${BG_PIDS[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done

        # Reap zombies
        for pid in "${BG_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        echo "All background processes stopped."
    else
        echo "No background processes have been started."
    fi
}
trap cleanup EXIT INT TERM

# Change cwd to shell script directory regardless of where script was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------------------------------------------------

# Change to robotgpt_output to load assets and checkpoints from there
cd ../../robotgpt_output

# Start openpi policy server
(
    echo "Starting openpi policy server..."
    source ../openpi/.venv/bin/activate
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
    exec python -m robotgpt.serve_policy &> "$OUT_STREAM" \
    --config "$TRAIN_CONFIG" \
    --checkpoint_dir "checkpoints/${TRAIN_CONFIG}/${DATASET}/${CHECKPOINT}" \
    --repo_id "${HF_USER}/${DATASET}"
) &
BG_PIDS+=("$!")

# Resolve if scene videos or annotations should be created
OUTPUT_FLAG="--record_scene"
if [[ "$AUTO_ANNOTATE" -eq 1 ]]; then
    OUTPUT_FLAG="--annotate"
fi

# Start task simulation with openpi client
echo "Starting simulation..."
source ../env_isaaclab/bin/activate
python ../RobotGPT/scripts/openpi_agent.py --task "$TASK" \
--enable_cameras \
$OUTPUT_FLAG \
--robot_type "$ROBOT_TYPE" \
--output_dir "evaluation/${TRAIN_CONFIG}/${DATASET}/${CHECKPOINT}" \
--num_rollouts "$NUM_ROLLOUTS" --max_duration "$MAX_DURATION" "${KIT_ARGS[@]}"
echo "Simulation stopped."