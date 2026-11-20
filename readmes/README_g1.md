# Unitree G1 Instructions

RobotGPT interface for the Unitree G1 robot with Inspire 6-DoF hands. Support is available for the built-in arm control routines, which stabilized lower body while providing arm joint command interfaces. However, we were not able to get the robot to respond to these commands interfaces described in the [official documentation](https://support.unitree.com/home/en/G1_developer/arm_control_routine). As we were mainly interested to use the project with a similar, proprietary solution developed in-house, we abandoned the built-in interfaces. However, the implementation for it is fully in place and also works with the [official simulator by Unitree](https://github.com/unitreerobotics/unitree_sim_isaaclab), which mirrors the robot's built-in interfaces. The issue was therefore likely only some version or configuration mismatch, and the implementation should work with a real robot and its built-in interfaces, even though we were not able to test it.

The selection of the command interfaces is done by a *mode* argument in relevant ROS nodes (see [Usage](README_g1.md#usage)), which may be:
- **g1_internal_controller** ... the official, built-in arm control routine interfaces by Unitree
- **g1_proprietary_controller** ... the proprietary, in-house solution
- **g1_isaaclab_simulation** ... the official, hardware-mirrored Isaac Lab simulation by Unitree

## Installation

You will need an Ubuntu PC with an NVIDIA GPU for connecting to the headset.

> Installation has been tested on Ubuntu 22.04

The XR headset connection for teleoperation is provided by [NVIDIA's Isaac Teleop ROS2 node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_teleop), which we will run using an Isaac ROS docker container. The tracked controller data provided by the ROS2 node will be utilized by the RobotGPT G1 integration ROS2 package, which is run in the same container.

1. Setup docker according to step 2 of the [Isaac ROS Getting Started Documentation](https://nvidia-isaac-ros.github.io/getting_started/index.html#initialize-isaac-ros-cli).
2. Create the docker container:
```
docker run -it \
  --name isaac_ros \
  --privileged \
  --network host \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority:ro" \
  -v /dev:/dev \
  nvcr.io/nvidia/isaac/ros:isaac_ros_89df02a734965ed64c227ef531c09d65-amd64
```
3. Prepare the docker container with Isaac Teleop and unitree_ros2:
```
docker exec -it isaac_ros bash  # Enter the docker container

# Install Isaac ROS Teleop
isaac-ros activate
sudo apt-get update
sudo apt-get install -y ros-jazzy-isaac-ros-teleop

# Install unitree_ros2 package
mkdir -p ~/RobotGPT
cd ~/RobotGPT
git clone https://github.com/unitreerobotics/unitree_ros2.git

apt install ros-jazzy-rmw-cyclonedds-cpp
apt install ros-jazzy-rosidl-generator-dds-idl
apt install libyaml-cpp-dev

source /opt/ros/jazzy/setup.bash
cd unitree_ros2/cyclonedds_ws
colcon build
```
4. Adjust unitree_ros/setup.sh in container to contain the following lines (replace YOUR_NETWORK_INTERFACE; run *ip link* to determine your interface):
```
#!/bin/bash
echo "Setup unitree ros2 environment"
source /opt/ros/jazzy/setup.bash
source ~/RobotGPT/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="YOUR_NETWORK_INTERFACE" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
export ROS_DOMAIN_ID=0
```
5. Adjust unitree_ros/setup_local.sh in container to contain the following lines:
```
#!/bin/bash
echo "Setup unitree ros2 simulation environment"
source /opt/ros/jazzy/setup.bash
source ~/RobotGPT/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="lo" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
export ROS_DOMAIN_ID=0
```
6. Install realsense ros wrappers in container by following the [official instructions](https://github.com/realsenseai/realsense-ros#installation-on-ubuntu).
7. Install Pinocchio with Casadi by following the [official instructions](https://stack-of-tasks.github.io/pinocchio/download.html).
8. Prepare RobotGPT G1 interface workspace in container:
```
docker exec -it isaac_ros bash  # Enter the docker container

# Create ROS workspace
source /opt/ros/jazzy/setup.bash
mkdir -p ~/RobotGPT/robotgpt_ws/src
cd ~/RobotGPT/robotgpt_ws/src

# Download RobotGPT G1 integration package
git clone https://github.com/ronypepper/RobotGPT.git
mv RobotGPT/ros_packages/g1/robotgpt_g1_inspire .
rm -r RobotGPT

# Download G1 urdf file for RobotGPT's inverse kinematics solver
cd ~/RobotGPT/robotgpt_ws/src/robotgpt_g1_inspire/robotgpt_g1_inspire
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cp -r xr_teleoperate/assets/g1 assets
rm -r xr_teleoperate

# Create robotgpt teleop environment
cd ~/RobotGPT/robotgpt_ws/src/robotgpt_g1_inspire/scripts
chmod +x create_robotgpt_g1_inspire_venv.sh
./create_robotgpt_g1_inspire_venv.sh

# Build ROS workspace
cd ~/RobotGPT/robotgpt_ws
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build
```


## Usage

Initialize the robot command interface you intend to use (depending on the *mode* argument).  
For the official, built-in interface by Unitree (mode:=g1_internal_controller) refer to the [official documentation](https://support.unitree.com/home/en/G1_developer/arm_control_routine).  
For the official, hardware-mirrored simulation by Unitree, refer to [Unitree Official G1 Simulation](README_g1.md#unitree-official-g1-simulation).

### Teleoperation

Begin by starting the docker container with `docker start isaac_ros`.  
Then, run these three command blocks in order in three separate terminals:
```
# Start the CloudXR instance
docker exec -it isaac_ros bash  # Enter the docker container
python3 -m isaacteleop.cloudxr
```
```
# Start the Isaac ROS Teleop node to receive topics containing controller data
docker exec -it isaac_ros bash  # Enter the docker container
export ROS_DOMAIN_ID=0
source ~/.cloudxr/run/cloudxr.env
ros2 launch isaac_ros_teleop isaac_ros_teleop.launch.py
```
```
# Start the RobotGPT teleoperation and demonstration recording node
export ROS_DOMAIN_ID=0
source ~/RobotGPT/unitree_ros2/setup.sh
source ~/RobotGPT/robotgpt_ws/install/setup.bash
ros2 launch robotgpt_g1_inspire teleoperation.launch.xml  # add arguments, e.g. mode, dataset path, hardware information, ...
```

> Note: for users of the proprietary simulation, replace "source ~/RobotGPT/unitree_ros2/setup.sh" above with "source ~/RobotGPT/unitree_ros2/cyclonedds_ws/install/setup.bash"

You may now teleoperate the robot and record multiple episodes of task demonstrations using the XR motion controllers:
 - Right primary (lower) button: Start teleoperation / Stop teleoperation and save dataset recording
 - Left primary (lower) button: Stop teleoperation and discard dataset recording
 - Right secondary (upper) button: Return robot and hands to home pose (only if teleoperation is not active)

After exiting the teleoperation node, a trainable HDF5 dataset will be saved to the path provided with the arguments (refer to [Training Instructions](README_training.md).)

### Policy Rollout

First, serve the trained policy (refer to [Training Instructions](README_training.md).)  
Then, start the policy rollout:
```
export ROS_DOMAIN_ID=0
source ~/RobotGPT/unitree_ros2/setup.bash
source ~/RobotGPT/robotgpt_ws/install/setup.bash
ros2 launch robotgpt_g1_inspire openpi_agent.launch.xml  # add arguments, e.g. mode, policy server ip, hardware information, ...
```

> Note: for users of the proprietary simulation, replace "source ~/RobotGPT/unitree_ros2/setup.sh" above with "source ~/RobotGPT/unitree_ros2/cyclonedds_ws/install/setup.bash"

You can now start policy rollout on the robot, stop the rollout or command the robot to its home pose using the following commands:  
`ros2 topic pub /client_command std_msgs/msg/String 'data: 'HOME'' --once`  
`ros2 topic pub /client_command std_msgs/msg/String 'data: 'START'' --once`  
`ros2 topic pub /client_command std_msgs/msg/String 'data: 'STOP'' --once`


## Unitree Official G1 Simulation

### Installation







Install simulation:

git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
docker pull nvidia/cuda:12.2.0-runtime-ubuntu22.04
cd unitree_sim_isaaclab
# Replace dockerfile
docker build -t unitree-sim:latest -f Dockerfile .

Create the docker container:

docker run -it \
  --name g1_sim \
  --privileged \
  --network host \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority:ro" \
  -v /dev:/dev \
  unitree-sim /bin/bash

In the container:
apt update
apt install git-lfs
cd unitree_sim_isaaclab
. fetch_assets.sh

On PC:
xhost +local:docker

Run simulation:
cd unitree_sim_isaaclab
python sim_main.py --task Isaac-PickPlace-RedBlock-G129-Inspire-Joint --robot_type g129 --device cpu  --enable_cameras --enable_inspire_dds

When simulation cannot open a window, make sure to:
xhost +local:docker
and check that echo $DISPLAY matches inside container and on host pc. If not set it in container using:
export DISPLAY=:X