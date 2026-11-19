"""
This script is the ROS2 transmitter end of a ROS2 -> ROS1 bridge for Isaac ROS Teleop.
The ROS2 transmitter node receives controller pose and input data messages from topics published by the Isaac ROS Teleop 
node, repacks and sends them over a ZMQ connection to a ROS1 receiver node, which republishes them on their topic names.

Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import PoseArray

import zmq
import msgpack


def pose_array_to_dict(msg: PoseArray):
    return {
        "header": {
            "stamp": {
                "sec": msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec,
            },
            "frame_id": msg.header.frame_id,
        },
        "poses": [
            {
                "position": {
                    "x": p.position.x,
                    "y": p.position.y,
                    "z": p.position.z,
                },
                "orientation": {
                    "x": p.orientation.x,
                    "y": p.orientation.y,
                    "z": p.orientation.z,
                    "w": p.orientation.w,
                },
            }
            for p in msg.poses
        ],
    }


def byte_multi_array_to_dict(msg: ByteMultiArray):
    return {
        "layout": {
            "dim": [
                {
                    "label": d.label,
                    "size": d.size,
                    "stride": d.stride,
                }
                for d in msg.layout.dim
            ],
            "data_offset": msg.layout.data_offset,
        },
        "data": msg.data,
    }


class RobotGptIsaacTeleopRosBridgeTransmitter(Node):
    def __init__(self):
        super().__init__('robotgpt_isaac_teleop_ros_bridge_transmitter')

        # ZMQ
        self.zmq_context = zmq.Context.instance()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")
        self.zmq_socket.setsockopt(zmq.SNDHWM, 1)  # no message pile-up

        # ROS Subscribers
        self.ee_pose_sub = self.create_subscription(
            PoseArray,
            'xr_teleop/ee_poses',
            self.ee_pose_callback,
            10)
        self.controller_data_sub = self.create_subscription(
            ByteMultiArray,
            'xr_teleop/controller_data',
            self.controller_data_callback,
            10)

    def ee_pose_callback(self, msg: PoseArray):
        payload = msgpack.packb(pose_array_to_dict(msg), use_bin_type=True)
        self.zmq_socket.send_multipart([b"ee_pose", payload])

    def controller_data_callback(self, msg: ByteMultiArray):
        payload = msgpack.packb(byte_multi_array_to_dict(msg), use_bin_type=True)
        self.zmq_socket.send_multipart([b"controller_data", payload])

    def destroy_node(self):
        self.zmq_socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    teleop_bridge_node = RobotGptIsaacTeleopRosBridgeTransmitter()

    rclpy.spin(teleop_bridge_node)

    teleop_bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
