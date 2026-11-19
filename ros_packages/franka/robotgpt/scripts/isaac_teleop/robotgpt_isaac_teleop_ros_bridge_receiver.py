#!/usr/bin/env python3

"""
This script is the ROS1 receiver end of a ROS2 -> ROS1 bridge for Isaac ROS Teleop.
The ROS2 transmitter node receives controller pose and input data messages from topics published by the Isaac ROS Teleop 
node, repacks and sends them over a ZMQ connection to a ROS1 receiver node, which republishes them on their topic names.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import rospy

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ByteMultiArray, MultiArrayDimension

import zmq
import msgpack


def dict_to_pose_array(data: dict):
    msg = PoseArray()

    msg.header.stamp.secs = data["header"]["stamp"]["sec"]
    msg.header.stamp.nsecs = data["header"]["stamp"]["nanosec"]
    msg.header.frame_id = data["header"]["frame_id"]

    for pose_data in data["poses"]:
        p = Pose()

        p.position.x = pose_data["position"]["x"]
        p.position.y = pose_data["position"]["y"]
        p.position.z = pose_data["position"]["z"]

        p.orientation.x = pose_data["orientation"]["x"]
        p.orientation.y = pose_data["orientation"]["y"]
        p.orientation.z = pose_data["orientation"]["z"]
        p.orientation.w = pose_data["orientation"]["w"]

        msg.poses.append(p)

    return msg


def dict_to_byte_multi_array(data: dict):
    msg = ByteMultiArray()

    msg.layout.data_offset = data["layout"]["data_offset"]

    for dim_data in data["layout"]["dim"]:
        d = MultiArrayDimension()
        d.label = dim_data["label"]
        d.size = dim_data["size"]
        d.stride = dim_data["stride"]
        msg.layout.dim.append(d)

    msg.data = [b[0] if b[0] < 128 else b[0] - 256 for b in data["data"]]

    return msg


def robotgpt_isaac_teleop_ros_bridge_receiver():
    rospy.init_node("robotgpt_isaac_teleop_ros_bridge_receiver")

    # Configuration parameters
    isaac_teleop_ip = rospy.get_param("~isaac_teleop_ip", "128.131.81.208")
    ee_pose_frame_id = rospy.get_param("~ee_pose_frame_id", "fr3_link0")

    # ROS publishers
    ee_pub = rospy.Publisher("xr_teleop/ee_pose", PoseArray, queue_size=10)
    controller_data_pub = rospy.Publisher("xr_teleop/controller_data", ByteMultiArray, queue_size=10)

    # ZMQ
    zmq_context = zmq.Context.instance()
    zmq_socket = zmq_context.socket(zmq.SUB)
    zmq_socket.connect(f"tcp://{isaac_teleop_ip}:5555")
    zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
    zmq_socket.setsockopt(zmq.RCVHWM, 1)  # no message pile-up

    try:
        while not rospy.is_shutdown():
            topic, payload = zmq_socket.recv_multipart()

            if topic == b"ee_pose":
                try:
                    msg = dict_to_pose_array(msgpack.unpackb(payload, raw=False))
                    if ee_pose_frame_id != "None":
                        msg.header.frame_id = ee_pose_frame_id
                    ee_pub.publish(msg)
                except Exception as e:
                    print(e)
            elif topic == b"controller_data":
                try:
                    msg = dict_to_byte_multi_array(msgpack.unpackb(payload, raw=False))
                    controller_data_pub.publish(msg)
                except Exception as e:
                    print(e)
    except rospy.ROSInterruptException:
        pass

    zmq_socket.close()
    zmq_context.term()


if __name__ == "__main__":
    robotgpt_isaac_teleop_ros_bridge_receiver()
