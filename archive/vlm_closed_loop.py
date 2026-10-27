#!/usr/bin/env python3
"""vlm_closed_loop.py — VLM 实时闭环：订阅 RGB 图像，判断安全，发布 cmd_vel"""

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
PROMPT = ("你是越野导航安全检查员。观察图像判断前方地面是否安全："
          "蓝色水面、裸露黄色沙土、深色泥泞沼泽均为危险地形。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")

class VLMClosedLoop(Node):
    def __init__(self):
        super().__init__('vlm_closed_loop')
        
        # 加载 VLM 模型
        self.get_logger().info("Loading VLM model...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
        self.get_logger().info("VLM model loaded!")
        
        # 订阅 RGB 图像
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image/compressed',
            self.image_callback,
            10)
        
        # 发布 cmd_vel
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.last_image = None
        self.safe = True
        
        # 定时器：每秒推理一次（VLM 推理慢，不用 10Hz）
        self.timer = self.create_timer(1.0, self.infer_and_publish)
        
    def image_callback(self, msg):
        # 解压图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
    def infer_and_publish(self):
        if self.last_image is None:
            return
        
        # 转成 PIL Image
        image = Image.fromarray(cv2.cvtColor(self.last_image, cv2.COLOR_BGR2RGB)).resize((448, 448))
        
        # VLM 推理
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64)
        resp = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 解析输出
        m = re.search(r'"safe"\s*:\s*(true|false)', resp)
        self.safe = (m.group(1) == "true") if m else True
        
        self.get_logger().info(f"VLM: safe={self.safe}, resp={resp[:50]}")
        
        # 发布 cmd_vel
        twist = Twist()
        if self.safe:
            twist.linear.x = 0.5  # 前进
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.0  # 停止
            twist.angular.z = 0.0
        
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = VLMClosedLoop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
