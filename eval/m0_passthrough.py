#!/usr/bin/env python3
"""m0_passthrough.py — M0: 纯 YOPO 基线（无语义感知）

launch 把 MPC 的 /cmd_vel remap 到了 /cmd_vel_mpc（供 M1–M4 的 VLM 节点转发）。
M0 不加载任何 VLM 节点，本节点只做无条件转发，把控制链路接回仿真器：

    /cmd_vel_mpc → /cmd_vel

用法：python3 m0_passthrough.py（只需 ROS 环境，无 VLM/torch 依赖）
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class M0Passthrough(Node):
    def __init__(self):
        super().__init__("m0_passthrough")
        self.create_subscription(Twist, "/cmd_vel_mpc", self._cmd_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("M0 passthrough: /cmd_vel_mpc -> /cmd_vel")

    def _cmd_callback(self, msg):
        self.pub_cmd_vel.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = M0Passthrough()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
