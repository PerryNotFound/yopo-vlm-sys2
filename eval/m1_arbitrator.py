#!/usr/bin/env python3
"""m1_arbitrator.py — M1: 仲裁节点
订阅 MPC 的 /cmd_vel 和 VLM 的 /vlm_safe
如果 safe=True → 转发 MPC 的速度
如果 safe=False → 发布 0 速度（停止）
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class M1Arbitrator(Node):
    def __init__(self):
        super().__init__('m1_arbitrator')
        
        # 订阅 MPC 的 /cmd_vel
        self.sub_mpc = self.create_subscription(
            Twist,
            '/cmd_vel_mpc',
            self.mpc_callback,
            10)
        
        # 订阅 VLM 的安全判断
        self.sub_safe = self.create_subscription(
            Bool,
            '/vlm_safe',
            self.safe_callback,
            10)
        
        # 发布最终的 /cmd_vel
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.last_mpc_cmd = Twist()
        self.safe = True
        
    def mpc_callback(self, msg):
        self.last_mpc_cmd = msg
        
    def safe_callback(self, msg):
        self.safe = msg.data
        self.get_logger().info(f"VLM safe={self.safe}")
        
        # 根据安全判断，发布速度
        if self.safe:
            self.pub_cmd.publish(self.last_mpc_cmd)
            self.get_logger().info(f"→ 转发 MPC 速度: v={self.last_mpc_cmd.linear.x:.2f}")
        else:
            stop_cmd = Twist()
            self.pub_cmd.publish(stop_cmd)
            self.get_logger().info("→ VLM 说危险，停止")

def main(args=None):
    rclpy.init(args=args)
    node = M1Arbitrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
