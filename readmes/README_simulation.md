# Isaac Lab Simulation Instructions


## Installation

The installation and later usage instructions assume you perform the install steps below within a common root folder, i.e. you should end up with a folder structure like this (root folder would be "RobotLearning" here):
```
RobotLearning
├── RobotGPT                            ... This repository
├── IsaacLab                            ... The Isaac Lab repository
├── openpi                              ... The openpi repository
│   └── .venv                           ... openpi's python environment
├── env_isaaclab                        ... Isaac Lab's python environment
├── env_robotgpt_evals                  ... Python environment used by some evaluation scripts
└── robotgpt_assets                     ... Simulation assets
    |── google_scanned_objects
    |   └── mujoco_scanned_objects
    └── Lightwheel_Kitchen
```

**Installation Steps**

1. Install Isaac Lab 3 using Isaac Sim pip package and Isaac Lab from GitHub (develop branch, commit cf615d32e3): [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/pip_installation.html).
> If you encounter connection issues with XR teleoperation later, you may need to configure your firewall. Refer to Isaac Teleop's Quick Start guide: [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html).

2. Install openpi from GitHub: [openpi GitHub](https://github.com/Physical-Intelligence/openpi).
    - If you want to log runs online with Weights&Biases, you may need to upgrade the wandb package to support longer API keys: `source openpi/.venv/bin/activate && uv pip install -U wandb`

3. Install RobotGPT from GitHub in Isaac Lab's environment:
```
git clone https://github.com/ronypepper/RobotGPT.git
source env_isaaclab/bin/activate
cd RobotGPT && python -m pip install -e source/RobotGPT
cd .. && cp -r RobotGPT/apps/robotgpt.python.xr.openxr.kit IsaacLab/apps/robotgpt.python.xr.openxr.kit  # Modified XR experience file fixing rendering issues
```

4. Modify openpi install with RobotGPT integration:
    - Run `cp -r RobotGPT/openpi/robotgpt openpi/src`
    - Edit openpi/src/openpi/training/config.py:
        - Add "from robotgpt.train_configurations import get_robotgpt_train_configs" to bottom of imports
        - Add "\*get_robotgpt_train_configs()," as last entry to the _CONFIGS list, right after "*polaris_config.get_polaris_configs(),"

5. Install openpi client in Isaac Lab's environment:
```
source env_isaaclab/bin/activate
cd openpi/packages/openpi-client
uv pip install -e .
```

6. Download MJCF models of Google's Scanned Objects Dataset (only required for pack items task environments):
```
mkdir -p robotgpt_assets/google_scanned_objects
cd robotgpt_assets/google_scanned_objects
git clone https://github.com/kevinzakka/mujoco_scanned_objects.git
```

7. Download Lightwheel Kitchen environment (only required for microwave task environments):
```
mkdir -p robotgpt_assets
cd robotgpt_assets
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LightwheelAI/Lightwheel_Kitchen.git
cd Lightwheel_Kitchen && git lfs pull
unzip Lightwheel_Kitchen.zip
rm Lightwheel_Kitchen.zip
```

8. Setup evaluation environment (only required for annotation and graph generation):
```
uv venv --python 3.12 --seed env_robotgpt_evals
source env_robotgpt_evals/bin/activate
uv pip install -r RobotGPT/evals/requirements.txt
```
> If you encounter missing libxcb-cursor0 error when running scripts in this environment, run: `sudo apt install libxcb-cursor0`


## Usage

The standard pipeline has four steps:
1. Collect task demonstrations with an XR headset and motion controllers using the "utils/record_xr_demos.sh" script.
2. Convert the obtained HDF5 dataset to a LeRobot dataset using the "utils/convert_hdf5_to_lerobot.sh" script.
3. Train a VLA model on it using the "utils/train_openpi_model.sh" script.
4. Serve the trained policy and evaluate its performance using the "utils/evaluate_openpi_model.sh" script.

For details about the shell scripts arguments, run `XXXXXXX.sh --help`.

> You may need to make the scripts executable by running:\
> `chmod +x record_xr_demos.sh convert_hdf5_to_lerobot.sh train_openpi_model.sh evaluate_openpi_model.sh`\
> inside "RobotGPT/utils/".


## Add a new robot to simulation

The following steps need to be done to add a new robot embodiment:
1. Define a new robot setup file in "RobotGPT/source/RobotGPT/RobotGPT/utils/robots/", which implements:
    - The *setup_<ROBOT_NAME>_joint_pos_env* function, which sets up the robot in an Isaac Lab environment (a *RobotGPTEnvCfg*) for joint position control.
    - The *_build_<ROBOT_NAME>_teleop_pipeline* function, which builds an Isaac Teleop teleoperation pipeline for absolute SE3 control of the end-effectors.
    - The *setup_<ROBOT_NAME>_ik_abs_env* function, which sets up the robot in an Isaac Lab environment (a *RobotGPTEnvCfg*) for absolute IK control via XR teleoperation using the *_build_<ROBOT_NAME>_teleop_pipeline* function.
    - The *process_observation_for_openpi_<ROBOT_NAME>* function. which converts Isaac Lab observations to the format expected by the corresponding openpi integration (see step 4).
    - The *process_openpi_action_<ROBOT_NAME>* function, which converts actions by the corresponding openpi integration to the format expected by Isaac Lab (see step 4).
2. Create a new directory containing environment configurations for the desired tasks at "RobotGPT/source/RobotGPT/RobotGPT/tasks/manager_based/<TASK_NAME>/config/". The new environment configurations use the first two functions from step 1 to setup the new robot. The "\_\_init\_\_.py" file in this folder needs to register the new environments with Isaac Lab.
3. Register the last two functions from step 1 in "RobotGPT/source/RobotGPT/RobotGPT/utils/robots/openpi_interface.py" in the *OPENPI_INTERFACE_FCTS* object under a name for the new robot type.

Then, follow [Add a new robot to training](README_training.md#add-a-new-robot-to-training) to add training support for the new robot embodiment.

See existing robot embodiment implementations for more detail.


## Add a new task environment

At a minimum, a new environment must have an *<TASK_NAME>EnvCfg* class inheriting from *RobotGPTEnvCfg*, defining a new prompt, as well as corresponding robot-specific environment configuration classes that define joint position / absolute IK control for the respective robot, using the setup functions created in step 1 of [Add a new robot to simulation](README_simulation.md#add-a-new-robot-to-simulation).

See existing task environment implementations for more detail.


## Utility commands

These are some useful commands during development. All command blocks need to be executed from the common root folder, i.e. "Robot Learning" in the **Directory Structure** section above.

### Zero-action task simulation
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/zero_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--num_envs 1 --enable_cameras --viz kit
```

### XR Teleoperation with tracked motion controllers
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/teleop_se3_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --xr --experience robotgpt.python.xr.openxr.kit
```

### Replay recorded demonstrations in simulation
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/replay_demos.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-IK-Abs-v0 \
--enable_cameras --viz kit --dataset_file robotgpt_output/datasets/hdf5/dataset.hdf5
```

### Merge multiple hdf5 datasets
```
source env_isaaclab/bin/activate
python IsaacLab/scripts/tools/merge_hdf5_datasets.py \
--input_files robotgpt_output/datasets/hdf5/dataset_1.hdf5 robotgpt_output/datasets/hdf5/dataset_2.hdf5 \
--output_file robotgpt_output/datasets/hdf5/dataset.hdf5
```

### Create mp4 videos from a hdf5 dataset
```
source env_isaaclab/bin/activate
mkdir -p robotgpt_output/datasets/videos
python IsaacLab/scripts/tools/hdf5_to_mp4.py --input_file robotgpt_output/datasets/hdf5/dataset.hdf5 \
--output_dir robotgpt_output/datasets/videos/dataset --input_keys table_img wrist_img \
--video_height 224 --video_width 224 --framerate 20
```

### Openpi Pi05 base model evaluation (not finetuned) 
**Terminal 1: Start openpi policy server**
```
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python openpi/scripts/serve_policy.py policy:checkpoint \
--policy.config=robotgpt_franka_single_arm_pi05_base \
--policy.dir=gs://openpi-assets/checkpoints/pi05_base
```
**Terminal 2:  Start task simulation with openpi client**
```
source env_isaaclab/bin/activate
python RobotGPT/scripts/openpi_agent.py --task RobotGPT-Place-Cube-In-Bin-Franka-Single-Arm-Pos-v0 \
--viz kit --record_scene --record_table --record_wrists \
--output_dir=robotgpt_output/evaluation/robotgpt_franka_single_arm_pi05_base
```