#!/usr/bin/env python3
"""m2_real.py — 真正的 M2：逐条判断每条轨迹安全吗？选第一条安全的轨迹
对每条轨迹（5条），取末端位置的图像区域
VLM 判断这个区域安全吗？
否决不安全的轨迹
选第一条安全的轨迹，发布末端位置作为目标点
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
PROMPT = ("你是越野导航安全检查员。观察这张图片，判断前方中间的地面是否安全："
          "只有蓝色水面是危险地形。其他所有东西（树木、树影、草地、泥土路、深色地面）都是安全的。"
          "特别注意：树影、深色地面都不是泥巴，都是安全的。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")

# 相机内参
fx = 895.69
fy = 895.69
cx = 320.0
cy = 240.0
img_width = 640
img_height = 480

# 相机外参（假设）：相机在小车前面 0.15m，高度 0.34m
cam_offset_x = 0.15
cam_offset_z = 0.34

class M2Real(Node):
    def __init__(self):
        super().__init__('m2_real')
        
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
        
        # 定时器：每 2 秒推理一次（因为要推理 5 次）
        self.timer = self.create_timer(2.0, self.infer)
        
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
        
    def project_path_to_image(self, path):
        """把轨迹从地图坐标系投影到图像坐标系"""
        if self.robot_pose is None:
            return None
        
        rx, ry, ryaw = self.robot_pose
        points = []
        
        for pose in path.poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            
            # 地图坐标系 → 小车坐标系
            dx = px - rx
            dy = py - ry
            
            # 旋转 -yaw
            cos_yaw = np.cos(-ryaw)
            sin_yaw = np.sin(-ryaw)
            cam_x = dx * cos_yaw - dy * sin_yaw
            cam_y = dx * sin_yaw + dy * cos_yaw
            
            # 小车坐标系 → 相机坐标系
            cam_x -= cam_offset_x
            cam_z = -cam_offset_z
            
            # 只投影相机前面的点
            if cam_x <= 0:
                continue
            
            # 相机坐标系 → 图像坐标系
            u = fx * cam_x / cam_x + cx
            v = fy * (cam_z) / cam_x + cy
            
            if 0 <= u < img_width and 0 <= v < img_height:
                points.append((int(u), int(v)))
        
        return points
        
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
        return safe, resp
        
    def infer(self):
        if self.last_image is None:
            return
        
        if self.robot_pose is None:
            return
        
        safe_paths = []
        
        # 对每条轨迹，单独判断
        for i, path in enumerate(self.candidate_paths):
            if path is None:
                continue
            
            # 取轨迹末端位置的图像区域
            points = self.project_path_to_image(path)
            if points is None or len(points) < 2:
                continue
            
            # 截取末端位置周围的图像区域（100x100）
            end_u, end_v = points[-1]
            x1 = max(0, end_u - 50)
            x2 = min(img_width, end_u + 50)
            y1 = max(0, end_v - 50)
            y2 = min(img_height, end_v + 50)
            
            img_region = self.last_image[y1:y2, x1:x2]
            
            # VLM 判断这个区域安全吗？
            pil_image = Image.fromarray(cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB))
            safe, resp = self.infer_image(pil_image)
            
            self.get_logger().info(f"轨迹{i}: safe={safe}, resp={resp[:50]}")
            
            if safe:
                safe_paths.append(i)
        
        # 选第一条安全的轨迹
        if len(safe_paths) > 0:
            best_path = safe_paths[0]
            end_pose = self.candidate_paths[best_path].poses[-1]
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.pose.position.x = end_pose.pose.position.x
            msg.pose.position.y = end_pose.pose.position.y
            msg.pose.orientation.w = 1.0
            self.pub_vlm_goal.publish(msg)
            self.get_logger().info(f"→ 选轨迹{best_path}，发布末端位置作为目标点")
        else:
            self.get_logger().info("→ 所有轨迹都危险，不发布目标点")

def main(args=None):
    rclpy.init(args=args)
    node = M2Real()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
