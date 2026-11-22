# Franka Panda Instructions

Repository was tested on a Franka Research 3 robot arm. Franka Panda or Franka Research 3 may be configured by arguments passed to relevant ROS launch files.  
For cameras we used two Realsense D435I cameras. The camera serial numbers are configurable through arguments passed to relevant ROS launch files.


## Installation

Because libfranka must run on a PC with real-time priority, which is not officially supported by NVIDIA drivers, you will need two machines running Ubuntu:
- **PC 1** with real-time priority running libfranka and the RobotGPT Franka integration ROS1 package.
- **PC 2** with an NVIDIA GPU that connects to the XR headset using Isaac Teleop for teleoperation (or serves the VLA policy for rollout.)

Both PC 1 and PC 2 and the XR headset must be on the same network.

### PC 1 Installation Steps
> PC 1 installation has been tested on Ubuntu 20.04
1. Install the [libfranka](https://github.com/frankarobotics/libfranka) client library.
2. Install [ROS1 Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu).
3. Prepare and build the ROS workspace:
```
# Create ROS workspace
source /opt/ros/noetic/setup.bash
mkdir -p ~/RobotGPT/catkin_ws/src
cd ~/RobotGPT/catkin_ws/src

# Download franka_ros package
git clone https://github.com/frankarobotics/franka_ros.git

# Download robotgpt franka integration package
git clone https://github.com/ronypepper/RobotGPT.git
mv RobotGPT/ros_packages/franka/robotgpt .
rm -r RobotGPT

# Create robotgpt teleop environment
cd ~/RobotGPT/catkin_ws/src/robotgpt/scripts/isaac_teleop
chmod +x create_isaac_teleop_ros_bridge_venv.sh
./create_isaac_teleop_ros_bridge_venv.sh

# Create robotgpt policy client environment
cd ~/RobotGPT/catkin_ws/src/robotgpt/scripts/openpi_client
chmod +x create_openpi_agent_venv.sh
./create_openpi_agent_venv.sh

# Build catkin workspace - replace DFranka_DIR path
~/RobotGPT/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=/PATH/TO/libfranka/build
```

### PC 2 Installation Steps
> PC 1 installation has been tested on Ubuntu 22.04

The XR headset connection for teleoperation is provided by [NVIDIA's Isaac Teleop ROS2 node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_teleop), which we will run using an Isaac ROS docker container. The tracked controller data provided by the ROS2 node will be transmitted to the ROS1 system on PC 1 via a custom zmq bridge.

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
3. Prepare the docker container:
```
docker exec -it isaac_ros bash  # Enter the docker container

# Install Isaac ROS Teleop
isaac-ros activate
sudo apt-get update
sudo apt-get install -y ros-jazzy-isaac-ros-teleop

# Download RobotGPT teleoperation interface
mkdir -p ~/RobotGPT
cd ~/RobotGPT
git clone https://github.com/ronypepper/RobotGPT.git
mv RobotGPT/ros_packages/franka/robotgpt/scripts/isaac_teleop/ros2/robotgpt_isaac_teleop_ros_bridge_transmitter.py .
mv RobotGPT/ros_packages/franka/robotgpt/scripts/isaac_teleop/create_isaac_teleop_ros_bridge_venv.sh .
rm -r RobotGPT

# Create RobotGPT teleop environment
chmod +x create_isaac_teleop_ros_bridge_venv.sh
./create_isaac_teleop_ros_bridge_venv.sh
```


## Usage

### Teleoperation

First, we startup the teleoperation interface and broadcaster on PC 2.  
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
source ~/.cloudxr/run/cloudxr.env
ros2 launch isaac_ros_teleop isaac_ros_teleop.launch.py
```
```
# Start the RobotGPT node that transmits the controller data via a zmq bridge to PC 1
docker exec -it isaac_ros bash  # Enter the docker container
source ~/RobotGPT/isaac_teleop_ros_bridge_venv/bin/activate
python3 ~/RobotGPT/robotgpt_isaac_teleop_ros_bridge_transmitter.py
```

Now, we start the teleoperation and demonstration recording node on PC 1:
```
cd ~/RobotGPT/catkin_ws
source devel/setup.bash
roslaunch robotgpt franka_teleoperation.launch  # add arguments, e.g. dataset path, hardware information, ...
```

You may now teleoperate the robot and record multiple episodes of task demonstrations using the XR motion controllers:
 - Right primary (lower) button: Start teleoperation / Stop teleoperation and save dataset recording
 - Left primary (lower) button: Stop teleoperation and discard dataset recording
 - Right secondary (upper) button: Return Franka to home pose (only if teleoperation is not active)

After exiting the teleoperation node, a trainable HDF5 dataset will be saved to the path provided with the arguments.  
Copy this dataset to a GPU-enabled machine to train a VLA model (refer to [Training Instructions](README_training.md).)

### Policy Rollout

First, serve the trained policy on a GPU-enabled machine (refer to [Training Instructions](README_training.md).)  
Then, start the policy rollout client on PC 1:
```
cd ~/RobotGPT/catkin_ws
source devel/setup.bash
roslaunch robotgpt franka_openpi_agent.launch  # add arguments, e.g. policy server ip, hardware information, ...
```