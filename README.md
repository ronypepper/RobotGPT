# RobotGPT

# Installation

The installation and later usage instructions assume you perform the install steps below within a common root folder, i.e. you should end up with a folder structure like this (root folder is called RobotLearning here):
```
RobotLearning
├── RobotGPT
├── IsaacLab
├── openpi
├── env_isaaclab
├── env_isaacteleop
```
### Installation Steps
1. Install Isaac Lab v2.3.2 locally (using Isaac Sim pip package and Isaac Lab from GitHub): [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

2. Install openpi locally from GitHub: [openpi GitHub](https://github.com/Physical-Intelligence/openpi).

3. Install RobotGPT from GitHub:
```
git clone https://github.com/ronypepper/RobotGPT.git
cd RobotGPT && python -m pip install -e source/RobotGPT
```

4. Modify openpi install:
    - `cp -r RobotGPT/openpi/robotgpt openpi/src`
    - Edit openpi/src/openpi/training/config.py:
        - Add "from robotgpt.configurations import get_robotgpt_configs" to bottom of imports
        - Add "*get_robotgpt_configs()," as last entry to the _CONFIGS list, right after "*polaris_config.get_polaris_configs(),"

5. Install openpi client in Isaac Lab's environment:
```
source env_isaaclab/bin/activate
cd openpi/packages/openpi-client
uv pip install -e .
```
6. Install Isaac Teleop in its own environment:
```
uv venv --python 3.12 --seed env_isaacteleop
source env_isaacteleop/bin/activate
uv pip install 'isaacteleop[cloudxr,retargeters]~=1.0.0' --extra-index-url https://pypi.nvidia.com
```
If you encounter connection issues with XR teleoperation later, you may need to configure your firewall. Refer to Isaac Teleop's Quick Start guide: [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html).
# Usage
All usage examples need to be executed from the common root folder, i.e. "Robot Learning" in the folder structure example from **Installation** above.
## Zero-action task simulation
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/zero_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--num_envs 1 --enable_cameras --device cpu
```
## XR Teleoperation with tracked motion controllers
**Terminal 1: Start CloudXR** *(Note: You'll need to accept the CloudXR EULA on first run)*
```
source env_isaacteleop/bin/activate
python -m isaacteleop.cloudxr --cloudxr-env-config=dev/quest3_cloudxr.env # Last argument enables optical hand tracking
```
**Terminal 2: Start task simulation with teleop**
```
source env_isaaclab/bin/activate
source ~/.cloudxr/run/cloudxr.env
python RobotGPT/scripts/teleop_se3_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --device cpu --teleop_device motioncontroller --xr
```

## XR demonstration collection with tracked motion controllers

**Terminal 1: Start CloudXR**
```
source env_isaacteleop/bin/activate
python -m isaacteleop.cloudxr --cloudxr-env-config=dev/quest3_cloudxr.env # Last argument enables optical hand tracking
```
**Terminal 2: Start task simulation with teleop and recording**
```
source env_isaaclab/bin/activate
source ~/.cloudxr/run/cloudxr.env
mkdir -p robotgpt_datasets
python RobotGPT/scripts/record_demos.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --device cpu --teleop_device motioncontroller --xr \
--dataset_file robotgpt_datasets/dataset.hdf5 --num_demos 25
```
**(Optional) Terminal 3: Replay recorded demonstration in simulation**
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/replay_demos.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --device cpu --dataset_file robotgpt_datasets/dataset.hdf5
```

## Merge multiple hdf5 datasets
```
source env_isaaclab/bin/activate
python IsaacLab/scripts/tools/merge_hdf5_datasets.py \
--input_files robotgpt_datasets/dataset_1.hdf5 robotgpt_datasets/dataset_2.hdf5 \
--output_file robotgpt_datasets/dataset.hdf5
```

## Create mp4 videos from a hdf5 dataset 
```
source env_isaaclab/bin/activate
mkdir -p robotgpt_datasets/videos/dataset
python IsaacLab/scripts/tools/hdf5_to_mp4.py --input_file robotgpt_datasets/dataset.hdf5 \
--output_dir robotgpt_datasets/videos/dataset --input_keys table_img wrist_img \
--video_height 224 --video_width 224 --framerate 20
```

## HDF5 to LeRobot dataset conversion
```
source openpi/.venv/bin/activate
python -m robotgpt.convert_hdf5_to_openpi_lerobot --dataset robotgpt_datasets/dataset.hdf5 \
--hf_user YOUR_USERNAME --robot_type franka_single_arm --output_name dataset --num_episodes 0
```

## Openpi training (to resume a training run uncomment the --resume argument in last command) 
```
source openpi/.venv/bin/activate
mkdir -p robotgpt_training && cd robotgpt_training
python ../openpi/scripts/compute_norm_stats.py --config-name pi05_robotgpt_franka
```
```
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python openpi/scripts/train.py pi05_robotgpt_franka \
--data.repo_id your_custom_repo_id --exp-name=my_task_2 #--resume
```

## OpenPi evaluation 
**Terminal 1: Start openpi policy server**
```
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python openpi/scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_robotgpt_franka --policy.dir=checkpoints/pi05_robotgpt_franka/my_task_2/4999
```
**Terminal 2:  Start task simulation with openpi client**
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/openpi_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--device cpu --video --video_obs --policy_name pi05_robotgpt_franka --policy_checkpoint 4999
```

## OpenPi Pi05 base model evaluation (not finetuned) 
```
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python openpi/scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_robotgpt_franka_base --policy.dir=gs://openpi-assets/checkpoints/pi05_base
```

<!-- 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_base
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos -->
