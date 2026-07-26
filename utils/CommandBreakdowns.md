# Command Breakdowns
The four command blocks below are the basis of the four shell scripts in this directory.

## XR demonstration collection with tracked motion controllers
**Terminal 2: Start task simulation with teleop and recording**
```
source env_isaaclab/bin/activate
mkdir -p robotgpt_output/datasets/hdf5
python RobotGPT/scripts/record_demos.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --xr --experience robotgpt.python.xr.openxr.kit \
--dataset_file robotgpt_output/datasets/hdf5/DATASET_NAME.hdf5 --num_demos 25
```

## HDF5 to LeRobot dataset conversion
```
source openpi/.venv/bin/activate
export HF_LEROBOT_HOME="$(pwd)/robotgpt_output/datasets/lerobot"
python -m robotgpt.convert_hdf5_to_openpi_lerobot --dataset robotgpt_output/datasets/hdf5/DATASET_NAME.hdf5 \
--hf_user HF_USERNAME --robot_type ROBOT_TYPE
```

## Openpi model training
**Compute normalization statistics (if not done yet)**
```
cd robotgpt_output
source ../openpi/.venv/bin/activate
export HF_LEROBOT_HOME="$(pwd)/datasets/lerobot"
python -m robotgpt.compute_norm_stats TRAIN_CONFIG \
--data.repo_id HF_USERNAME/DATASET_NAME --exp-name DATASET_NAME
```
**Training model**
```
cd robotgpt_output
source ../openpi/.venv/bin/activate
export HF_LEROBOT_HOME="$(pwd)/datasets/lerobot"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python ../openpi/scripts/train.py TRAIN_CONFIG \
--data.repo_id HF_USERNAME/DATASET_NAME --exp-name DATASET_NAME --resume
```

## Openpi model evaluation
**Terminal 1: Start openpi policy server**
```
cd robotgpt_output
source ../openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -m robotgpt.serve_policy \
--config TRAIN_CONFIG \
--checkpoint_dir checkpoints/TRAIN_CONFIG/DATASET_NAME/4999 \
--repo_id HF_USERNAME/DATASET_NAME
```
**Terminal 2:  Start task simulation with openpi client**
```
cd robotgpt_output
source ../env_isaaclab/bin/activate
python ../RobotGPT/scripts/openpi_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--enable_cameras --robot_type franka_single_arm --record_scene --record_table --record_wrists \
--output_dir=evaluation/TRAIN_CONFIG/DATASET_NAME/4999
```