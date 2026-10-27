#!/usr/bin/env python3
"""m3_fusion.py — M3: 语义安全分数正向融合（Positive Fusion）

与 M2（m2_veto.py 硬否决）的对照实验：不否决任何候选轨迹，
而是给每条候选加上连续的语义代价加成，由 yopo_node 的 argmin 自然偏向安全轨迹：

    cost_fused = cost_yopo + penalty_semantic

与 M2 的关键差异（论文对比点）：
- M2：危险区域的轨迹代价置 +inf（宁可错杀；可能全否决→停车）
- M3：危险只表现为代价梯度（平滑、永不停车；但危险轨迹权重可能降得不够，
  代价接近时仍可能选进水的轨迹）

惩罚计算：沿每条候选轨迹采样多点，取
    penalty_i = SEMANTIC_COST_MAX * max_p [ zone_score(zone(p)) * (1 - dist(p)/DANGER_RANGE_M) ]
其中 zone_score 为该区域的危险像素占比（0~1，颜色分割给出），
距离因子让近处危险权重更大。vlm_zone 模式下 zone_score 退化为 0/1。

话题：
  订阅 /camera/color/image/compressed, /odom, /visualize/path_{0..4},
       /cmd_vel_mpc（MPC 输出，本节点转发给 /cmd_vel）
  发布 /vlm_penalty (std_msgs/Float32MultiArray) 每条候选的语义代价加成
        /vlm_veto_debug (std_msgs/String)      VLM 原始输出，调试用
        /cmd_vel (geometry_msgs/Twist)         转发给仿真器
"""

import json
import math
import re
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from PIL import Image
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"

# ==================== 模式与区域 ====================

MODE = "hybrid"  # "vlm_zone" | "hybrid" | "color"，与 M2 消融对齐

ZONES = ("far_left", "left", "center", "right", "far_right")
ZONE_COUNT = len(ZONES)

PROMPT_ZONE = (
    "你是越野机器人的语义安全系统。以下危险地形深度传感器看不见，只有视觉能识别："
    "蓝色水面（水坑、湖泊）、黄色沙地（细沙、沙坑）、湿滑的深色泥泞。"
    "以下都是安全的：草地、泥土路、干燥地面、树影、树木、石头（树木石头由其他系统避障）。"
    "将图像横向等分为五个区域：最左、左、中、右、最右，"
    "判断每个区域前方的地面是否包含危险地形，注意只占区域边缘一小角的危险不要算整个区域危险。"
    '只输出JSON：{"danger":["far_left","left","center","right","far_right"]}，'
    '都没有则输出{"danger":[]}'
)

PROMPT_BINARY = (
    "你是越野机器人的语义安全系统。观察这张第一人称视角图片前方的地面。"
    "危险地形：蓝色水面（水坑、湖泊）、黄色沙地、湿滑泥泞。"
    "安全：草地、泥土路、干燥地面、树影、树木、石头。"
    "注意区分：蓝色水面在画面下方的地面上，蓝色天空在画面上方地平线上方，天空是安全的。"
    "图中前方地面上是否存在危险地形？"
    '只输出JSON：{"hazard":true} 或 {"hazard":false}'
)

# ==================== 相机与几何 ====================

# RGB 相机水平视场角（度），焦距由图像宽度自动推算。
CAMERA_HFOV_DEG = 69.4

CANDIDATE_NUM = 5
# 每条轨迹均匀采样多少个点
PATH_SAMPLE_POINTS = 6
# 只评价该距离（米）以内的轨迹点
DANGER_RANGE_M = 10.0

# ==================== 融合权重 ====================

# 语义代价上限（与 YOPO 代价同量纲；yopo_transforms 中 cost_max=50）
SEMANTIC_COST_MAX = 50.0

# ==================== 颜色分割 ====================

GROUND_ROW_FRAC = 0.45   # 只看地平线以下的地面区域
MIN_HAZARD_PIXELS = 300  # 区域内危险像素绝对数量下限（抗噪点）

DETECT_WATER = True
DETECT_SAND = True
DETECT_MUD = False  # 深棕色与树影高度混淆，默认关

# OpenCV HSV：H∈[0,179], S,V∈[0,255]
# WaterDepthBased 真实水面材质标定（2026-10，sample_hsv.py --nogui 聚类）：
# 主体灰蓝（H95-120,S19-48）+ 天空反光（H85-96,S8-40,V198+）+ 蓝色反射（H105-110,S55-95）
# S 下限 8 用于排除纯白/纯灰物体（机器人自身、石头）
HSV_WATER = ((80, 8, 85), (125, 255, 255))
HSV_SAND = ((15, 60, 120), (40, 255, 255))
HSV_MUD = ((5, 60, 20), (20, 255, 120))

# VLM verdict 粘性：False 需连续多次才采信（True 立即生效）
VLM_CLEAR_ROUNDS = 2

LOOP_SLEEP = 0.05
PENALTY_PUBLISH_PERIOD = 0.1  # 10Hz


class VLMFusionNode(Node):
    def __init__(self):
        super().__init__("vlm_fusion")

        # ---------- VLM（color 模式不加载，秒开） ----------
        self.model = None
        if MODE != "color":
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            self.get_logger().info("Loading VLM model...")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map="auto"
            )
            self.get_logger().info("VLM model loaded!")

        # ---------- ROS ----------
        self.create_subscription(
            CompressedImage, "/camera/color/image/compressed",
            self._image_callback, 10,
        )
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        for i in range(CANDIDATE_NUM):
            self.create_subscription(
                Path, f"/visualize/path_{i}",
                lambda msg, i=i: self._path_callback(msg, i), 10,
            )

        self.pub_penalty = self.create_publisher(
            Float32MultiArray, "/vlm_penalty", 10
        )
        self.pub_debug = self.create_publisher(String, "/vlm_veto_debug", 10)

        # launch 把 MPC 的 /cmd_vel remap 到了 /cmd_vel_mpc，
        # 本节点兼任转发器，把速度指令还给仿真器订阅的 /cmd_vel
        self.create_subscription(Twist, "/cmd_vel_mpc", self._cmd_vel_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # ---------- 状态 ----------
        self._lock = threading.Lock()
        self._latest_image = None      # BGR np.ndarray
        self._image_seq = 0
        self._robot_pose = None        # (x, y, yaw)
        self._candidate_paths = [None] * CANDIDATE_NUM

        self._zone_scores = {z: 0.0 for z in ZONES}  # 颜色分割的区域危险分数
        self._vlm_zones = set()                      # vlm_zone 模式的 VLM 分区
        self._vlm_hazard = True                      # hybrid 模式 VLM 确认；默认保守
        self._vlm_clear_count = 0

        self._last_infer_seq = 0

        self.create_timer(PENALTY_PUBLISH_PERIOD, self._update_penalty)

        self._stop_event = threading.Event()
        self._infer_thread = None
        if self.model is not None:
            self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
            self._infer_thread.start()

        self.get_logger().info(f"vlm_fusion started, MODE={MODE}")

    # ---------- ROS 回调 ----------

    def _image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        scores = self._color_zone_scores(img) if MODE != "vlm_zone" else None
        with self._lock:
            self._latest_image = img
            self._image_seq += 1
            if scores is not None:
                self._zone_scores = scores

    def _odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        with self._lock:
            self._robot_pose = (x, y, yaw)

    def _path_callback(self, msg, idx):
        with self._lock:
            self._candidate_paths[idx] = msg

    def _cmd_vel_callback(self, msg):
        self.pub_cmd_vel.publish(msg)

    # ---------- 颜色分割（连续分数） ----------

    def _color_zone_scores(self, img_bgr):
        """HSV 阈值分割，返回 {区域: 危险像素占比 0~1}"""
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros((h, w), np.uint8)
        if DETECT_WATER:
            mask |= cv2.inRange(hsv, np.array(HSV_WATER[0]), np.array(HSV_WATER[1]))
        if DETECT_SAND:
            mask |= cv2.inRange(hsv, np.array(HSV_SAND[0]), np.array(HSV_SAND[1]))
        if DETECT_MUD:
            mask |= cv2.inRange(hsv, np.array(HSV_MUD[0]), np.array(HSV_MUD[1]))
        mask[: int(h * GROUND_ROW_FRAC), :] = 0  # 只看地面

        scores = {}
        zw = w / ZONE_COUNT
        for i, z in enumerate(ZONES):
            zm = mask[:, int(i * zw): int((i + 1) * zw)]
            cnt = int(np.count_nonzero(zm))
            scores[z] = cnt / zm.size if cnt >= MIN_HAZARD_PIXELS else 0.0
        return scores

    # ---------- VLM 推理（语义确认） ----------

    def _infer_image(self, image_bgr, prompt):
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        latency = time.time() - t0
        resp = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return resp, latency

    def _infer_loop(self):
        while not self._stop_event.is_set() and rclpy.ok():
            with self._lock:
                img = self._latest_image
                seq = self._image_seq
            if img is None or seq == self._last_infer_seq:
                time.sleep(LOOP_SLEEP)
                continue
            self._last_infer_seq = seq

            try:
                if MODE == "vlm_zone":
                    raw, latency = self._infer_image(img, PROMPT_ZONE)
                    danger = set()
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    if m:
                        try:
                            danger = {
                                z for z in json.loads(m.group(0)).get("danger", [])
                                if z in ZONES
                            }
                        except json.JSONDecodeError:
                            danger = {z for z in ZONES if z in m.group(0)}
                    with self._lock:
                        self._vlm_zones = danger
                    self.get_logger().info(f"VLM zones={sorted(danger)} latency={latency:.2f}s")
                else:  # hybrid：二值语义确认 + 粘性
                    raw, latency = self._infer_image(img, PROMPT_BINARY)
                    m = re.search(r'"hazard"\s*:\s*(true|false)', raw)
                    hazard = (m.group(1) == "true") if m else True  # 解析失败保守
                    with self._lock:
                        if hazard:
                            self._vlm_hazard = True
                            self._vlm_clear_count = 0
                        else:
                            self._vlm_clear_count += 1
                            if self._vlm_clear_count >= VLM_CLEAR_ROUNDS:
                                self._vlm_hazard = False
                    self.get_logger().info(
                        f"VLM hazard_raw={hazard} hazard={self._vlm_hazard} "
                        f"latency={latency:.2f}s"
                    )
            except Exception as e:
                self.get_logger().warn(f"VLM inference failed, skip this round: {e}")
                time.sleep(LOOP_SLEEP)
                continue

            try:
                self.pub_debug.publish(String(data=raw))
            except Exception:
                break  # 节点正在关闭

    # ---------- 轨迹 → 语义代价 ----------

    def _img_fx(self, img_w):
        return (img_w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)

    def _path_penalty(self, path: Path, robot_pose, img_w, zone_scores):
        """单条候选的语义代价：采样点区域分数 × 距离衰减，取最大"""
        n = len(path.poses)
        if n == 0:
            return 0.0
        rx, ry, yaw = robot_pose
        fx = self._img_fx(img_w)
        cx = img_w / 2.0
        worst = 0.0
        for i in np.linspace(0, n - 1, min(PATH_SAMPLE_POINTS, n)).astype(int):
            p = path.poses[i].pose.position
            dx, dy = p.x - rx, p.y - ry
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if local_x <= 0.1:  # 身后或正侧方
                continue
            dist = math.hypot(local_x, local_y)
            if dist > DANGER_RANGE_M:
                continue
            bearing = math.atan2(local_y, local_x)  # >0 为左
            u = cx - fx * math.tan(bearing)
            if u < 0 or u >= img_w:  # 视野外：无法确认，不罚
                continue
            zone = ZONES[min(int(u / (img_w / ZONE_COUNT)), ZONE_COUNT - 1)]
            score = zone_scores.get(zone, 0.0)
            worst = max(worst, score * (1.0 - dist / DANGER_RANGE_M))
        return SEMANTIC_COST_MAX * worst

    # ---------- 主更新（10Hz） ----------

    def _update_penalty(self):
        with self._lock:
            pose = self._robot_pose
            paths = list(self._candidate_paths)
            zone_scores = dict(self._zone_scores)
            vlm_zones = set(self._vlm_zones)
            vlm_hazard = self._vlm_hazard
        if pose is None:
            return

        # 按模式合成区域分数（0~1）
        if MODE == "vlm_zone":
            scores = {z: 1.0 if z in vlm_zones else 0.0 for z in ZONES}
        elif MODE == "color":
            scores = zone_scores
        else:  # hybrid：颜色分数 × VLM 语义确认
            scores = zone_scores if vlm_hazard else {z: 0.0 for z in ZONES}

        img_w = self._latest_image.shape[1] if self._latest_image is not None else 640
        penalties = [
            self._path_penalty(path, pose, img_w, scores) if path is not None else 0.0
            for path in paths
        ]

        try:
            msg = Float32MultiArray()
            msg.data = [float(p) for p in penalties]
            self.pub_penalty.publish(msg)
        except Exception:
            return

        if any(p > 0 for p in penalties):
            nonzero = {z: round(s, 2) for z, s in scores.items() if s > 0}
            self.get_logger().info(
                f"mode={MODE} penalties={[round(p, 1) for p in penalties]} "
                f"zone_scores={nonzero}"
            )

    def destroy_node(self):
        self._stop_event.set()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VLMFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
