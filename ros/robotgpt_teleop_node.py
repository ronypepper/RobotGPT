import rclpy
from rclpy.node import Node

from std_msgs.msg import ByteMultiArray


class RobotGptTeleoperator(Node):

    def __init__(self):
        super().__init__('robotgpt_teleoperator')
        self.controller_data_subscription = self.create_subscription(
            ByteMultiArray,
            'xr_teleop/controller_data_topic',
            self.controller_data_callback,
            10)

    def controller_data_callback(self, msg):
        print(msg.data)


def main(args=None):
    rclpy.init(args=args)

    robotgpt_teleoperator = RobotGptTeleoperator()

    rclpy.spin(robotgpt_teleoperator)

    if robotgpt_teleoperator is not None:
        robotgpt_teleoperator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
