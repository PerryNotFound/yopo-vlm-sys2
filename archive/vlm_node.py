#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_node.py — YOPO-Rally × Qwen3-VL System-2 参考节点（方案 A 完整版）

职责：
  低频（0.5–2Hz）对当前图像做语义理解，输出三种"参考"信号（软约束）：
    方案 A: /vlm_goal      (geometry_msgs/PoseStamped, map 系)  语义安全目标点
    方案 B: /vlm_semantic  (std_msgs/String, JSON)              危险区域语义（供 yopo_node 加权）
    方案 C: /vlm_recovery  (std_msgs/Bool)                      卡住/低置信度恢复指令

  设计铁律：
    1) 异步解耦 —— 独立节点，不阻塞 yopo_node 10Hz / mpc_node 20Hz；
    2) 软约束   —— 输出只是"参考"，yopo_node 侧 2s 未收到新结果即回退纯 YOPO；
    3) 单点失败不瘫痪 —— 推理失败/反投影失败 → 静默跳过本轮。

  深度约定（与 yopo_node 对齐）：
    - 输入 /camera/depth/image 可能是任意分辨率，resize 到 (160,32)（YOPO 网络输入）
    - 16UC1 单位毫米：depth_m = raw_mm / 1000；32FC1 假设单位米
    - 深度是相机光轴距离 → 水平距离 dist = d / cos(angle)
    - 相机系 = 车体系（YOPO 约定），只做水平面反投影，忽略高度

  用法：
    source /opt/ros/humble/setup.bash
    source ~/YOPO-Rally-Whole/tcp_ws/install/setup.bash
    source ~/YOPO-Rally-Whole/diablo_ws/install/setup.bash
    source ~/YOPO-Rally-Whole/YOPO-Rally/install/setup.bash
    source ~/YOPO-Rally-Whole/YOPO-Rally/.venv/bin/activate
    python ~/YOPO-Rally-Whole/yopo-vlm-sys2/deploy/vlm_node.py
"""

import json
import time

import cv2
import cv_bridge
import numpy as np
import rclpy
import rclpy.qos
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

# ---------------------------------------------------------------------------
# 可配置参数
# ---------------------------------------------------------------------------
MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
IMAGE_TOPIC = "/camera/depth/image"
VLM_FREQ_HZ = 2.0                        # 0.5–2Hz 低频
MAX_NEW_TOKENS = 128
STALE_TIMEOUT = 2.0                      # yopo_node 侧超时阈值（本节点只打时间戳）

# 深度图约定（与 yopo_node.py 对齐）
IMG_W, IMG_H = 160, 32                   # YOPO 网络输入分辨率
MAX_DEPTH_M = 20.0                       # max_depth = 20 m
CAM_FOV_RAD = np.pi / 2                  # 相机水平 FOV 90°

PROMPT_A = (
    "你是越野机器人的视觉语义助手。给定当前深度/视觉图像，判断："
    "前方是否有 YOPO-Rally 几何感知无法识别的语义危险（水面/沙地/泥地/深坑/倒木等）。"
    "若安全，输出 JSON：{\"type\":\"goal\",\"safe\":true}"
    "若存在危险但可绕行，输出安全的行进方向：{\"type\":\"goal\",\"safe\":true,"
    "\"pixel_x\":<图像横向像素(0=最左,160=最右)>,\"pixel_y\":<纵向像素(0=顶部,32=底部)>}"
    "若前方被危险封锁无法通行，输出：{\"type\":\"goal\",\"safe\":false}"
    "只输出 JSON，不要其他文字。"
)


# ---------------------------------------------------------------------------
# 推理后端抽象
# ---------------------------------------------------------------------------
class VLMBackend:
    def infer(self, image: np.ndarray, prompt: str) -> str:
        raise NotImplementedError


class TransformersBackend(VLMBackend):
    """Qwen3-VL via transformers（>=4.57.0，官方入口 AutoModelForImageTextToText）"""

    def __init__(self, model_id: str, device: str = "cuda", max_new_tokens: int = 128):
        self._max_new_tokens = max_new_tokens
        self._device = device
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype="auto", device_map="auto"
        ).eval()

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        import torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image_rgb], return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        out = self._processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return out.strip()


class LlamaCppBackend(VLMBackend):
    """GGUF + llama.cpp 后端（占位）。接入 export_int4.sh 产物后启用。"""

    def __init__(self, server_url: str = "http://127.0.0.1:8080/v1"):
        self._url = server_url

    def infer(self, image_bgr: np.ndarray, prompt: str) -> str:
        raise NotImplementedError("LlamaCppBackend 待接入（先跑通 TransformersBackend）")


# ---------------------------------------------------------------------------
# ROS2 节点
# ---------------------------------------------------------------------------
class VLMNode(Node):
    def __init__(self):
        super().__init__("vlm_node")
        self._bridge = cv_bridge.CvBridge()

        self._backend = TransformersBackend(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)

        # 输入缓存（墙钟时间戳，与 _tick 同源）
        self._latest_image_bgr = None
        self._latest_depth_raw = None       # (IMG_H, IMG_W) float32, 单位米
        self._latest_image_stamp = 0.0
        # odom 缓存（map 系车体位姿）
        self._odom_pos = np.zeros(2)
        self._odom_yaw = 0.0
        self._odom_stamp = 0.0

        self._goal_pub = self.create_publisher(
            PoseStamped, "/vlm_goal", rclpy.qos.qos_profile_system_default
        )
        self._semantic_pub = self.create_publisher(
            String, "/vlm_semantic", rclpy.qos.qos_profile_system_default
        )
        self._recovery_pub = self.create_publisher(
            Bool, "/vlm_recovery", rclpy.qos.qos_profile_system_default
        )

        self._depth_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self._depth_callback, rclpy.qos.qos_profile_sensor_data
        )
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, rclpy.qos.qos_profile_sensor_data
        )
        self._timer = self.create_timer(1.0 / VLM_FREQ_HZ, self._tick)
        self.get_logger().info("vlm_node started (VLM System-2, soft constraint)")

    # -- 输入 ------------------------------------------------------------
    def _depth_callback(self, msg: Image):
        depth = self._bridge.imgmsg_to_cv2(msg).astype(np.float32)
        # 1) 原始深度 → 米
        if msg.encoding.upper() == "16UC1":
            depth_m = depth.astype(np.float32) / 1000.0      # mm → m
            depth_m[depth_m <= 0.0] = np.nan                 # 无效像素
            depth_m[depth_m > MAX_DEPTH_M] = MAX_DEPTH_M     # 截断
        elif msg.encoding.upper() == "32FC1":
            depth_m = np.nan_to_num(depth, nan=0.0, posinf=MAX_DEPTH_M, neginf=0.0)
            depth_m[depth_m <= 0.0] = np.nan
            depth_m[depth_m > MAX_DEPTH_M] = MAX_DEPTH_M
        else:
            return
        # 2) resize 到 YOPO 网络输入分辨率（保证 VLM 像素坐标与反投影一致）
        depth_m = cv2.resize(depth_m, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST_EXACT)
        self._latest_depth_raw = depth_m
        # 3) 灰度可视化图（VLM 输入，3 通道 BGR）
        depth_vis = np.clip(depth_m / MAX_DEPTH_M * 255.0, 0, 255).astype(np.uint8)
        self._latest_image_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        self._latest_image_stamp = time.time()

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self._odom_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        self._odom_yaw = rot.as_euler("xyz", degrees=False)[2]
        self._odom_stamp = time.time()

    # -- 主循环 -----------------------------------------------------------
    def _tick(self):
        if self._latest_image_bgr is None:
            return
        if time.time() - self._latest_image_stamp > 1.0:
            return  # 数据源断开，不推理
        try:
            raw = self._backend.infer(self._latest_image_bgr, PROMPT_A)
        except Exception as e:
            self.get_logger().warn(f"VLM infer failed: {e}", throttle_duration_sec=5)
            return
        parsed = self._parse_json(raw)
        if parsed is None:
            self.get_logger().warn(f"VLM JSON parse failed: {raw[:80]}", throttle_duration_sec=5)
            return
        self._publish(parsed)
        self.get_logger().info(f"VLM result: {parsed}")

    @staticmethod
    def _parse_json(raw: str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    # -- 输出 ------------------------------------------------------------
    def _publish(self, parsed: dict):
        if parsed.get("type") == "recovery":
            self._recovery_pub.publish(Bool(data=True))
            return
        if parsed.get("type") == "goal":
            if not parsed.get("safe", False):
                self._recovery_pub.publish(Bool(data=True))
                return
            if "pixel_x" in parsed and "pixel_y" in parsed:
                goal = self._pixel_to_map_goal(parsed["pixel_x"], parsed["pixel_y"])
                if goal is not None:
                    self._goal_pub.publish(goal)
            return
        if parsed.get("type") == "mask":
            msg = String()
            msg.data = json.dumps(parsed, ensure_ascii=False)
            self._semantic_pub.publish(msg)

    # -- 核心：像素 → map 系目标点（方案 A 的几何） --------------------------
    def _pixel_to_map_goal(self, pixel_x: float, pixel_y: float):
        """深度反投影：像素 → 车体系水平面点 → map 系点。

        约定（与 yopo_node 一致）：
          深度 d 是相机光轴距离；水平角 θ = arctan((px - W/2)/(W/2) * tan(FOV/2))；
          水平距离 dist = d / cos(θ)；
          车体系 x=前方, y=左侧（像素右 → 车右 → -y）；
          最终 map 点 = odom_pos + R(yaw) · [dist, -dist·sinθ]ᵀ
        """
        if self._latest_depth_raw is None or time.time() - self._odom_stamp > 1.0:
            return None
        px = int(round(float(pixel_x)))
        py = int(round(float(pixel_y)))
        px = min(max(px, 0), IMG_W - 1)
        py = min(max(py, 0), IMG_H - 1)

        d = float(self._latest_depth_raw[py, px])
        if not np.isfinite(d) or d <= 0.05 or d >= MAX_DEPTH_M:
            return None

        half_w = IMG_W / 2.0
        ang = np.arctan((px - half_w) / half_w * np.tan(CAM_FOV_RAD / 2.0))
        dist = d / np.cos(ang)
        forward = dist
        lateral = -dist * np.sin(ang)          # 像素右 → 车右 → 车体系 -y

        cos_y, sin_y = np.cos(self._odom_yaw), np.sin(self._odom_yaw)
        x_map = self._odom_pos[0] + cos_y * forward - sin_y * lateral
        y_map = self._odom_pos[1] + sin_y * forward + cos_y * lateral

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x_map)
        msg.pose.position.y = float(y_map)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg


def main():
    rclpy.init()
    node = VLMNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
