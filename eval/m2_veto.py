#!/usr/bin/env python3
"""m2_veto.py — M2: 候选轨迹语义离散否决 + 全否决恢复（Semantic Veto + Recovery）

yopo_node 的两个 VLM 接口：
- /vlm_veto (Int32MultiArray) 硬否决：被否决候选代价置 +inf，全部否决则停车
- /vlm_goal (PoseStamped)     软引导：恢复态下把 YOPO 候选扇区引向安全方向

三种模式（顶部 MODE 切换，用于论文消融）：
- "vlm_zone"：VLM 整图单次推理直接输出危险区域。4B 模型空间定位不可靠
  （左右混淆、区域抖动），仅作对照。
- "color"：纯 HSV 颜色分割定位危险（水面=蓝、沙地=黄），无 VLM。论文基线。
- "hybrid"（默认）：颜色分割做空间定位（快、准），VLM 做二值语义确认
  "前方是否有危险地形"（慢、排除颜色误报）。VLM verdict 粘性化：
  hazard=True 立即生效，hazard=False 需连续 VLM_CLEAR_ROUNDS 次才采信。

恢复状态机（M4 雏形）：
- NORMAL：正常否决。YOPO 扇区内存在未被否决的候选即可继续行驶。
- RECOVERY：全部候选被否决持续 RECOVERY_ENTER_SEC 秒进入。
  每拍选取一个安全区域（优先保持上次选择，否则选最靠中间的），
  向该方向发布 /vlm_goal 引导 YOPO 扇区转向，直到视野内危险消失。
- STUCK：恢复态下连安全区域都没有（四面环水），保持停车等待，
  论文中计为受控失败（优于驶入危险）。

其他设计要点：
1. 沿每条轨迹采样多点投影到图像区域判定，轨迹任一点落入危险区域即否决。
2. 距离门控：只否决 DANGER_RANGE_M 以内进入危险区域的轨迹。
3. 视野外不否决：采样点超出 RGB 视场时该点不参与否决。
4. VLM 只管语义危险（水面/沙地/泥泞）；树木石头由 YOPO 深度规划负责。
5. 区域防抖：危险持续 DANGER_CONFIRM_SEC 才否决，安全持续
   SAFE_RELEASE_SEC 才解除（对称防抖，抑制否决集抖动导致的方向抖动）。

话题：
  订阅 /camera/color/image/compressed, /odom, /visualize/path_{0..4},
       /cmd_vel_mpc（MPC 输出，本节点转发给 /cmd_vel）
  发布 /vlm_veto (std_msgs/Int32MultiArray) 被否决的候选序号
        /vlm_goal (geometry_msgs/PoseStamped) 恢复态引导目标（map 系）
        /vlm_state (std_msgs/String)         NORMAL/RECOVERY/STUCK，计指标用
        /vlm_veto_debug (std_msgs/String)    VLM 原始输出，调试用
        /cmd_vel (geometry_msgs/Twist)       转发给仿真器
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
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from PIL import Image
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, String

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"

# ==================== 模式与区域 ====================

MODE = "hybrid"  # "vlm_zone" | "hybrid" | "color"

ZONES = ("far_left", "left", "center", "right", "far_right")
ZONE_COUNT = len(ZONES)

PROMPT_ZONE = (
    "你是越野机器人的语义安全系统。以下危险地形深度传感器看不见，只有视觉能识别："
    "浅蓝色水面（水坑、湖泊）、黄色沙地（细沙、沙坑）、湿滑的深色泥泞。"
    "以下都是安全的：草地、泥土路、干燥地面、树影、树木、石头（树木石头由其他系统避障）。"
    "将图像横向等分为五个区域：最左、左、中、右、最右，"
    "判断每个区域前方的地面是否包含危险地形，注意只占区域边缘一小角的危险不要算整个区域危险。"
    '只输出JSON：{"danger":["far_left","left","center","right","far_right"]}，'
    '都没有则输出{"danger":[]}'
)

PROMPT_BINARY = (
    "你是越野机器人的语义安全系统。观察这张第一人称视角图片前方的地面。"
    "危险地形：浅蓝色水面（水坑、湖泊）、黄色沙地、湿滑泥泞。"
    "安全：草地、泥土路、干燥地面、树影、树木、石头。"
    "注意区分：浅蓝色水面在画面下方的地面上，蓝色天空在画面上方地平线上方，天空是安全的。"
    "图中前方地面上是否存在危险地形？"
    '只输出JSON：{"hazard":true} 或 {"hazard":false}'
)

# ==================== 相机与几何 ====================

# RGB 相机水平视场角（度），焦距由图像宽度自动推算。
# 真机参考：D435 RGB ≈ 69.4°，D435 深度 ≈ 87°，YOPO 深度输入 90°。
CAMERA_HFOV_DEG = 69.4

CANDIDATE_NUM = 5
# 每条轨迹均匀采样多少个点做区域判定
PATH_SAMPLE_POINTS = 6
# 距离门控：只否决该距离（米）以内进入危险区域的轨迹
DANGER_RANGE_M = 10.0

# ==================== 颜色分割（color/hybrid 模式） ====================

# 只检测图像下半部分（地平线以下是地面）；相机俯仰变了要调这个
GROUND_ROW_FRAC = 0.45
# 一个区域内危险像素占比超过该阈值才算危险
ZONE_HAZARD_FRAC = 0.04
# 一个区域内危险像素绝对数量下限（抗噪点）
MIN_HAZARD_PIXELS = 300

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

# ==================== 防抖与粘性 ====================

DANGER_CONFIRM_SEC = 0.5   # 区域危险持续这么久才否决
SAFE_RELEASE_SEC = 2.0    # 区域安全持续这么久才解除
VLM_CLEAR_ROUNDS = 2       # hybrid 模式：VLM 连续多少次说无危险才采信

# ==================== 恢复行为 ====================

RECOVERY_ENTER_SEC = 2.0   # 全否决持续这么久进入恢复态
RECOVERY_GOAL_DIST = 7.0   # /vlm_goal 距离（与 yopo_node 的 min(7, ...) 对齐）

LOOP_SLEEP = 0.05
VETO_PUBLISH_PERIOD = 0.1  # 否决掩码发布周期（10Hz，快于 VLM 推理）


class VLMVetoNode(Node):
    def __init__(self):
        super().__init__("vlm_veto")

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

        self.pub_veto = self.create_publisher(Int32MultiArray, "/vlm_veto", 10)
        self.pub_vlm_goal = self.create_publisher(PoseStamped, "/vlm_goal", 10)
        self.pub_state = self.create_publisher(String, "/vlm_state", 10)
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

        self._color_zones = set()      # 颜色分割结果（图像回调里即时更新）
        self._vlm_zones = set()        # vlm_zone 模式的 VLM 分区结果
        self._vlm_hazard = True        # hybrid 模式的 VLM 二值确认；默认保守
        self._vlm_clear_count = 0      # VLM 连续说无危险的次数（粘性用）

        self._last_infer_seq = 0
        now = time.monotonic()
        self._zone_danger_since = {z: now for z in ZONES}
        self._zone_safe_since = {z: now for z in ZONES}
        self._zone_vetoed = {z: False for z in ZONES}

        # 恢复状态机
        self._state = "NORMAL"         # NORMAL | RECOVERY | STUCK
        self._all_veto_since = None
        self._recovery_zone = None

        self.create_timer(VETO_PUBLISH_PERIOD, self._update_veto)

        self._stop_event = threading.Event()
        self._infer_thread = None
        if self.model is not None:
            self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
            self._infer_thread.start()

        self.get_logger().info(f"vlm_veto started, MODE={MODE}")

    # ---------- ROS 回调 ----------

    def _image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        color_zones = self._color_danger_zones(img) if MODE != "vlm_zone" else set()
        with self._lock:
            self._latest_image = img
            self._image_seq += 1
            self._color_zones = color_zones

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

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self.pub_state.publish(String(data=state))
            self.get_logger().warn(f"state -> {state}")

    # ---------- 颜色分割（空间定位） ----------

    def _color_danger_zones(self, img_bgr):
        """HSV 阈值分割危险地形，返回有危险的区域集合"""
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

        zones = set()
        zw = w / ZONE_COUNT
        for i, z in enumerate(ZONES):
            zm = mask[:, int(i * zw): int((i + 1) * zw)]
            cnt = int(np.count_nonzero(zm))
            if cnt >= MIN_HAZARD_PIXELS and cnt / zm.size >= ZONE_HAZARD_FRAC:
                zones.add(z)
        return zones

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
                else:  # hybrid：二值语义确认 + 粘性（False 需连续确认）
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

    # ---------- 轨迹 → 区域 → 否决 ----------

    def _img_fx(self, img_w):
        return (img_w / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)

    def _zone_center_bearing(self, zone, img_w):
        """区域中心对应的方位角（弧度，>0 为左）"""
        u_c = (ZONES.index(zone) + 0.5) * img_w / ZONE_COUNT
        return math.atan((img_w / 2.0 - u_c) / self._img_fx(img_w))

    def _candidate_zones(self, path: Path, robot_pose, img_w):
        """沿轨迹均匀采样多点，返回 DANGER_RANGE_M 以内轨迹点经过的图像区域集合"""
        n = len(path.poses)
        if n == 0:
            return set()
        rx, ry, yaw = robot_pose
        fx = self._img_fx(img_w)
        cx = img_w / 2.0
        zones = set()
        for i in np.linspace(0, n - 1, min(PATH_SAMPLE_POINTS, n)).astype(int):
            p = path.poses[i].pose.position
            dx, dy = p.x - rx, p.y - ry
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if local_x <= 0.1:  # 身后或正侧方
                continue
            if math.hypot(local_x, local_y) > DANGER_RANGE_M:  # 距离门控
                continue
            bearing = math.atan2(local_y, local_x)  # >0 为左
            u = cx - fx * math.tan(bearing)         # 左侧目标 u 小
            if u < 0 or u >= img_w:                 # 视野外：无法确认，不否决
                continue
            zones.add(ZONES[min(int(u / (img_w / ZONE_COUNT)), ZONE_COUNT - 1)])
        return zones

    def _update_veto_state(self, danger_zones):
        """时间制对称防抖：危险持续 DANGER_CONFIRM_SEC 才否决，安全持续 SAFE_RELEASE_SEC 才解除"""
        now = time.monotonic()
        for z in ZONES:
            if z in danger_zones:
                self._zone_safe_since[z] = now
                if now - self._zone_danger_since[z] >= DANGER_CONFIRM_SEC:
                    self._zone_vetoed[z] = True
            else:
                self._zone_danger_since[z] = now
                if now - self._zone_safe_since[z] >= SAFE_RELEASE_SEC:
                    self._zone_vetoed[z] = False
        return {z for z in ZONES if self._zone_vetoed[z]}

    # ---------- 主更新（10Hz） ----------

    def _update_veto(self):
        with self._lock:
            pose = self._robot_pose
            paths = list(self._candidate_paths)
            color_zones = set(self._color_zones)
            vlm_zones = set(self._vlm_zones)
            vlm_hazard = self._vlm_hazard
        if pose is None:
            return

        if MODE == "vlm_zone":
            danger = vlm_zones
        elif MODE == "color":
            danger = color_zones
        else:  # hybrid：颜色定位 + VLM 语义确认
            danger = color_zones if vlm_hazard else set()

        vetoed_zones = self._update_veto_state(danger)

        img_w = self._latest_image.shape[1] if self._latest_image is not None else 640
        veto_idx = []
        for i, path in enumerate(paths):
            if path is None:
                continue
            if self._candidate_zones(path, pose, img_w) & vetoed_zones:
                veto_idx.append(i)

        # ---------- 恢复状态机 ----------
        now = time.monotonic()
        all_vetoed = len(veto_idx) == CANDIDATE_NUM and all(
            p is not None for p in paths
        )
        if all_vetoed:
            if self._all_veto_since is None:
                self._all_veto_since = now
            elif now - self._all_veto_since >= RECOVERY_ENTER_SEC:
                self._set_state("RECOVERY")
        else:
            self._all_veto_since = None

        if self._state == "RECOVERY":
            safe_zones = [z for z in ZONES if z not in vetoed_zones]
            if not danger:
                # 视野内危险消失，退出恢复，回归正常导航
                self._recovery_zone = None
                self._set_state("NORMAL")
            elif not safe_zones:
                # 真·被困：保持停车等待（/vlm_goal 不发，veto 保持全否决）
                self._set_state("STUCK")
            else:
                # 选安全区域：优先保持上次选择（防抖），否则选最靠中间的
                if self._recovery_zone not in safe_zones:
                    self._recovery_zone = min(
                        safe_zones, key=lambda z: abs(ZONES.index(z) - ZONE_COUNT // 2)
                    )
                # 发布 /vlm_goal 把 YOPO 候选扇区引向安全方向
                bearing = self._zone_center_bearing(self._recovery_zone, img_w)
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x = pose[0] + RECOVERY_GOAL_DIST * math.cos(pose[2] + bearing)
                goal.pose.position.y = pose[1] + RECOVERY_GOAL_DIST * math.sin(pose[2] + bearing)
                goal.pose.orientation.w = 1.0
                self.pub_vlm_goal.publish(goal)

        elif self._state == "STUCK":
            # 危险从某些方向解除则回到恢复态继续引导
            if [z for z in ZONES if z not in vetoed_zones]:
                self._set_state("RECOVERY")

        try:
            msg = Int32MultiArray()
            msg.data = veto_idx
            self.pub_veto.publish(msg)
        except Exception:
            return

        if veto_idx or self._state != "NORMAL":
            self.get_logger().info(
                f"state={self._state} mode={MODE} danger={sorted(danger)} "
                f"vetoed_zones={sorted(vetoed_zones)} veto_paths={veto_idx}"
            )

    def destroy_node(self):
        self._stop_event.set()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VLMVetoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
