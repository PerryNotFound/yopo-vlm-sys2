#!/usr/bin/env python3
"""m2_arbitrator.py — M2: 真正的候选轨迹离散否决
订阅 MPC 的 /cmd_vel 和 VLM 的安全判断
如果中间安全 → 转发 MPC 速度
如果中间危险 → 修改速度，让小车绕开（左转或右转）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import re

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
PROMPT = ("你是越野导航安全检查员。观察这张图片判断前方地面是否安全："
          "蓝色水面、裸露黄色沙土、深色泥泞沼泽为危险地形。树木、树影、草地均为安全。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")

class M2Arbitrator(Node):
    def __init__(self):
        super().__init__('m2_arbitrator')
        
        # 加载 VLM 模型
        self.get_logger().info("Loading VLM model...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
        self.get_logger().info("VLM model loaded!")
        
        # 订阅 RGB 图像
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/color/image/compressed',
            self.image_callback,
            10)
        
        # 订阅 MPC 的 /cmd_vel
        self.sub_mpc = self.create_subscription(
            Twist,
            '/cmd_vel_mpc',
            self.mpc_callback,
            10)
        
        # 发布最终的 /cmd_vel
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.last_image = None
        self.last_mpc_cmd = Twist()
        
        # 定时器：每秒推理一次
        self.timer = self.create_timer(1.0, self.infer)
        
    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
    def mpc_callback(self, msg):
        self.last_mpc_cmd = msg
        
    def infer_image(self, image):
        """VLM 判断一张图安全吗"""
        image = image.resize((448, 448))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64)
        resp = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        m = re.search(r'"safe"\s*:\s*(true|false)', resp)
        safe = (m.group(1) == "true") if m else True
        return safe
        
    def infer(self):
        if self.last_image is None:
            return
        
        # 切成左、中、右三块
        h, w = self.last_image.shape[:2]
        left = Image.fromarray(cv2.cvtColor(self.last_image[:, :w//3], cv2.COLOR_BGR2RGB))
        center = Image.fromarray(cv2.cvtColor(self.last_image[:, w//3:2*w//3], cv2.COLOR_BGR2RGB))
        right = Image.fromarray(cv2.cvtColor(self.last_image[:, 2*w//3:], cv2.COLOR_BGR2RGB))
        
        # VLM 逐块判断
        safe_left = self.infer_image(left)
        safe_center = self.infer_image(center)
        safe_right = self.infer_image(right)
        
        self.get_logger().info(f"左: safe={safe_left}, 中: safe={safe_center}, 右: safe={safe_right}")
        
        # 根据安全判断，修改 MPC 速度
        cmd = Twist()
        if safe_center:
            # 中间安全，转发 MPC 速度
            cmd = self.last_mpc_cmd
            self.get_logger().info(f"→ 中间安全，转发 MPC 速度: v={cmd.linear.x:.2f}")
        elif safe_left:
            # 中间危险，左边安全，左转
            cmd.linear.x = 0.3
            cmd.angular.z = 0.5
            self.get_logger().info("→ 中间危险，左边安全，左转绕开")
        elif safe_right:
            # 中间和左边危险，右边安全，右转
            cmd.linear.x = 0.3
            cmd.angular.z = -0.5
            self.get_logger().info("→ 中间和左边危险，右边安全，右转绕开")
        else:
            # 都危险，停止（回退）
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("→ 都危险，停止（回退）")
        
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = M2Arbitrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
