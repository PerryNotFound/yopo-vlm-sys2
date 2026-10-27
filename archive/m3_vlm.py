#!/usr/bin/env python3
"""m3_vlm.py — M3: VLM 辅助 YOPO，VLM 判断左中右安全吗？选安全的方向
YOPO 正常跑（平时）
VLM 判断左中右安全吗？
中间安全 → YOPO 正常跑
中间危险 → 选安全的方向发布目标点
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import re
from scipy.spatial.transform import Rotation as R

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
PROMPT = ("你是越野导航安全检查员。观察这张图片，判断前方地面是否有威胁："
          "威胁包括：树木、石头、蓝色水面、黄色沙土、深色泥泞等结构化物体。"
          "安全包括：草地、泥土路、树影等非结构化地面。"
          "特别注意：树影是安全的，不是泥巴。"
          "只输出JSON：{\"type\":\"goal\",\"threat\":true} 或 {\"type\":\"goal\",\"threat\":false}")

# 连续多少次说中间有威胁才介入
THREAT_THRESHOLD = 2

class M3VLM(Node):
    def __init__(self):
        super().__init__('m3_vlm')
        
        self.get_logger().info("Loading VLM model...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
        self.get_logger().info("VLM model loaded!")
        
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/color/image/compressed',
            self.image_callback,
            10)
        
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.candidate_paths = [None] * 5
        for i in range(5):
            self.create_subscription(
                Path,
                f'/visualize/path_{i}',
                lambda msg, i=i: self.path_callback(msg, i),
                10)
        
        self.pub_vlm_goal = self.create_publisher(PoseStamped, '/vlm_goal', 10)
        
        self.last_image = None
        self.robot_pose = None
        
        # 连续威胁计数器
        self.threat_count = 0
        # 是否在介入状态
        self.is_intervening = False
        
        # 定时器：每 0.5 秒推理一次（更频繁）
        self.timer = self.create_timer(0.5, self.infer)
        
    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        yaw = rot.as_euler("xyz", degrees=False)[2]
        self.robot_pose = (x, y, yaw)
        
    def path_callback(self, msg, idx):
        self.candidate_paths[idx] = msg
        
    def infer_image(self, image):
        image = image.resize((448, 448))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64)
        resp = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        m = re.search(r'"threat"\s*:\s*(true|false)', resp)
        threat = (m.group(1) == "true") if m else False
        return threat, resp
        
    def infer(self):
        if self.last_image is None:
            return
        
        if self.robot_pose is None:
            return
        
        h, w = self.last_image.shape[:2]
        
        # 把图像切成左、中、右三块
        left = Image.fromarray(cv2.cvtColor(self.last_image[:, :w//3], cv2.COLOR_BGR2RGB))
        center = Image.fromarray(cv2.cvtColor(self.last_image[:, w//3:2*w//3], cv2.COLOR_BGR2RGB))
        right = Image.fromarray(cv2.cvtColor(self.last_image[:, 2*w//3:], cv2.COLOR_BGR2RGB))
        
        # VLM 逐块判断有没有威胁
        threat_left, _ = self.infer_image(left)
        threat_center, _ = self.infer_image(center)
        threat_right, _ = self.infer_image(right)
        
        safe_left = not threat_left
        safe_center = not threat_center
        safe_right = not threat_right
        
        self.get_logger().info(f"左: safe={safe_left}, 中: safe={safe_center}, 右: safe={safe_right}, intervening={self.is_intervening}")
        
        # 如果中间安全，退出介入状态
        if safe_center:
            if self.is_intervening:
                self.is_intervening = False
                self.threat_count = 0
                self.get_logger().info("→ 中间安全，退出介入状态，继续跑 YOPO")
            else:
                self.get_logger().info("→ 中间安全，YOPO 正常跑")
            return
        
        # 中间危险，计数器 +1
        self.threat_count += 1
        
        if self.threat_count >= THREAT_THRESHOLD:
            # 连续 THREAT_THRESHOLD 次说中间危险，开始介入
            if not self.is_intervening:
                self.is_intervening = True
                self.get_logger().info(f"→ 连续{self.threat_count}次中间危险，开始介入")
            
            # 选安全的方向，发布对应的目标点
            if safe_left:
                # 左边安全，发布左边目标点
                if self.candidate_paths[1] is not None:
                    end_pose = self.candidate_paths[1].poses[-1]
                    msg = PoseStamped()
                    msg.header.frame_id = "map"
                    msg.pose.position.x = end_pose.pose.position.x
                    msg.pose.position.y = end_pose.pose.position.y
                    msg.pose.orientation.w = 1.0
                    self.pub_vlm_goal.publish(msg)
                    self.get_logger().info("→ 介入中，左边安全，发布左边目标点")
            elif safe_right:
                # 右边安全，发布右边目标点
                if self.candidate_paths[3] is not None:
                    end_pose = self.candidate_paths[3].poses[-1]
                    msg = PoseStamped()
                    msg.header.frame_id = "map"
                    msg.pose.position.x = end_pose.pose.position.x
                    msg.pose.position.y = end_pose.pose.position.y
                    msg.pose.orientation.w = 1.0
                    self.pub_vlm_goal.publish(msg)
                    self.get_logger().info("→ 介入中，右边安全，发布右边目标点")
            else:
                # 左右都危险，选左边绕开
                if self.candidate_paths[1] is not None:
                    end_pose = self.candidate_paths[1].poses[-1]
                    msg = PoseStamped()
                    msg.header.frame_id = "map"
                    msg.pose.position.x = end_pose.pose.position.x
                    msg.pose.position.y = end_pose.pose.position.y
                    msg.pose.orientation.w = 1.0
                    self.pub_vlm_goal.publish(msg)
                    self.get_logger().info("→ 介入中，左右都危险，选左边绕开")
        else:
            self.get_logger().info(f"→ 中间危险，但还没到{self.threat_count}/{THREAT_THRESHOLD}，不介入")

def main(args=None):
    rclpy.init(args=args)
    node = M3VLM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
