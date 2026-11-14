#!/usr/bin/env python3
"""vlm_semantic_avoid.py — 纯 VLM 语义避障节点（无颜色分割）

设计定位：
- 底层运动规划完全交给 YOPO（深度几何避障 + 候选轨迹）；
- 本节点只做一件事：让 VLM 直接看图，识别水坑/泥泞/沙地/树干等
  语义障碍在图像五分区（far_left/left/center/right/far_right）中的位置，
  把穿过危险区域的 YOPO 候选轨迹否决掉；
- 全部候选被否决时进入恢复状态机，用 /vlm_goal 把 YOPO 扇区引向安全方向。

与 m2_veto.py 的关系：接口完全兼容（同样发布 /vlm_veto、/vlm_goal、
/vlm_state、/vlm_veto_debug，同样转发 /cmd_vel_mpc -> /cmd_vel），
可以原位替换运行；m2/m3/m4 保留作论文消融对照。区别：
- 删掉全部 HSV 颜色分割代码和 hybrid/color 模式——颜色法在该工况
  （半米高相机、掠射角水面反射）已实证失效，感知 100% 由 VLM 承担；
- VLM 输出从"危险区域列表"升级为"类别+区域"的结构化 hazards，
  每次否决能记录是哪一类语义障碍触发的（计指标、写论文可用）。

VLM 输出协议（JSON）：
  {"hazards":[{"type":"water","zone":"center"},{"type":"tree","zone":"left"}]}
  type ∈ {water, mud, sand, tree}，zone ∈ 五分区；空则 {"hazards":[]}。
  兼容旧格式 {"danger":["center"]}（type 记为 unknown）。

话题：
  订阅 /camera/color/image/compressed, /odom, /visualize/path_{0..4},
       /cmd_vel_mpc（MPC 输出，本节点转发给 /cmd_vel）
  发布 /vlm_veto (std_msgs/Int32MultiArray) 被否决的候选序号
       /vlm_goal (geometry_msgs/PoseStamped) 恢复态引导目标（map 系）
       /vlm_state (std_msgs/String)         NORMAL/RECOVERY/STUCK
       /vlm_hazards (std_msgs/String)       当前生效 hazards 的 JSON（防抖后）
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

# ==================== 语义类别与区域 =================

ZONES = ("far_left", "left", "center", "right", "far_right")
ZONE_COUNT = len(ZONES)

# VLM 可识别的语义类别；VETO_CLASSES 决定哪些类别参与否决
CLASSES = ("water", "mud", "sand", "tree")
VETO_CLASSES = frozenset(CLASSES)  # 想忽略某类（如树交给 YOPO）就从这里删掉

# 安全余量：地形类危险（水面/泥/沙）的否决向左右各扩 VETO_MARGIN_ZONES 个
# 分区——水坑跨分区交界时 VLM 只报一个区，不扩的话贴边轨迹会漏否决擦进水里；
# tree 不扩（竖直障碍由 YOPO 深度层负责，扩了会在密林里过度保守）
MARGIN_CLASSES = frozenset({"water", "mud", "sand", "unknown"})
VETO_MARGIN_ZONES = 1

PROMPT = (
    "你是越野机器人的视觉语义感知系统，负责发现前方会影响通行安全的障碍。"
    "观察这张车头第一人称相机图像，识别以下类别："
    "water：水面、水坑、湖泊（浅水反光的也算）；"
    "mud：湿滑泥泞、深泥浆；"
    "sand：松软的沙地、沙坑；"
    "tree：树干、倒木等竖直木质障碍。"
    "将图像横向等分为五个区域：far_left（最左）、left（左）、center（中）、"
    "right（右）、far_right（最右）。"
    "只报告区域前方的地面上会影响通行的危险目标；"
    "即使目标只占区域一部分、或距离还较远，只要确实在地面上的也要报告（提前预警比漏报更重要）。"
    "草地、泥土路、落叶、树影、石头、天空都是安全的，不要报告。"
    '只输出JSON：{"hazards":[{"type":"water","zone":"center"}]}，'
    "type 只能是 water/mud/sand/tree，"
    "zone 只能是 far_left/left/center/right/far_right；"
    '没有危险则输出 {"hazards":[]}'
)

MAX_NEW_TOKENS = 96  # hazards 列表比二值回答长，给足输出空间

# ==================== 相机与几何 =================

# RGB 相机水平视场角（度），焦距由图像宽度自动推算。
CAMERA_HFOV_DEG = 69.4

CANDIDATE_NUM = 5
# 每条轨迹均匀采样多少个点做区域判定
PATH_SAMPLE_POINTS = 6
# 距离门控：只否决该距离（米）以内进入危险区域的轨迹
DANGER_RANGE_M = 10.0

# ==================== 防抖与粘性 =================

DANGER_CONFIRM_SEC = 0.25  # 区域危险持续这么久才否决（VLM 单帧约 0.3s，即首次报出即生效）
SAFE_RELEASE_SEC = 2.0     # 区域安全持续这么久才解除

# ==================== 恢复行为 =================

RECOVERY_ENTER_SEC = 1.0   # 全否决持续这么久进入恢复态
RECOVERY_GOAL_DIST = 7.0   # /vlm_goal 距离（与 yopo_node 的 min(7, ...) 对齐）

LOOP_SLEEP = 0.05
VETO_PUBLISH_PERIOD = 0.1  # 否决掩码发布周期（10Hz，快于 VLM 推理）


def parse_hazards(raw: str):
    """从 VLM 输出解析 [(cls, zone)]；JSON 失败时退化为文本扫描"""
    hazards = []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            items = data.get("hazards")
            if items is None:  # 兼容旧协议 {"danger":["center"]}
                items = [{"type": "unknown", "zone": z}
                         for z in data.get("danger", [])]
            for it in items:
                if not isinstance(it, dict):
                    continue
                cls, zone = it.get("type"), it.get("zone")
                if zone in ZONES and (cls in CLASSES or cls == "unknown"):
                    hazards.append((cls, zone))
        except (json.JSONDecodeError, AttributeError):
            pass
    if not hazards:
        # 兜底：文本扫描——类别词与区域词按字符距离贪心配对
        # （每处提及最多配一次；长词在前，避免 left 吃掉 far_left）
        text = raw.lower()
        zone_pat = r"far_left|far_right|left|right|center"
        cls_pos = [(m.start(), cls) for cls in CLASSES
                   for m in re.finditer(re.escape(cls), text)]
        zone_pos = [(m.start(), m.group(0))
                    for m in re.finditer(zone_pat, text)]
        pairs = sorted(
            (abs(cp - zp), ci, zi)
            for ci, (cp, _) in enumerate(cls_pos)
            for zi, (zp, _) in enumerate(zone_pos)
        )
        used_cls, used_zone = set(), set()
        for _, ci, zi in pairs:
            if ci in used_cls or zi in used_zone:
                continue
            used_cls.add(ci)
            used_zone.add(zi)
            hazards.append((cls_pos[ci][1], zone_pos[zi][1]))
    return hazards


class VLMSemanticAvoidNode(Node):
    def __init__(self):
        super().__init__("vlm_semantic_avoid")

        # ---------- VLM ----------
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
        self.pub_hazards = self.create_publisher(String, "/vlm_hazards", 10)
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

        self._vlm_hazards = []         # 最近一次 VLM 解析出的 [(cls, zone)]
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
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            f"vlm_semantic_avoid started, classes={sorted(VETO_CLASSES)}"
        )

    # ---------- ROS 回调 ----------

    def _image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        with self._lock:
            self._latest_image = img
            self._image_seq += 1

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

    # ---------- VLM 推理 ----------

    def _infer_image(self, image_bgr):
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False)
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
                raw, latency = self._infer_image(img)
                hazards = parse_hazards(raw)
                with self._lock:
                    self._vlm_hazards = hazards
                self.get_logger().info(
                    f"VLM hazards={hazards} latency={latency:.2f}s"
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
            hazards = list(self._vlm_hazards)
        if pose is None:
            return

        # VETO_CLASSES 过滤后的危险区域；地形类危险向左右扩安全余量
        danger = set()
        for cls, zone in hazards:
            if cls not in VETO_CLASSES and cls != "unknown":
                continue
            danger.add(zone)
            if cls in MARGIN_CLASSES:
                i = ZONES.index(zone)
                for k in range(1, VETO_MARGIN_ZONES + 1):
                    if i - k >= 0:
                        danger.add(ZONES[i - k])
                    if i + k < ZONE_COUNT:
                        danger.add(ZONES[i + k])
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
                # 视野内危险消失，退出恢复，回归正常导航。
                # 同时重置全否决计时——否决集还有 SAFE_RELEASE_SEC 迟滞没解除，
                # 不重置的话会因 _all_veto_since 仍是旧值而立即重新进入
                # RECOVERY，状态机在两种状态间高频抖动。
                self._recovery_zone = None
                self._all_veto_since = None
                self._set_state("NORMAL")
            elif not safe_zones:
                # 真·被困：保持停车等待（/vlm_goal 不发，veto 保持全否决）
                self._set_state("STUCK")
            else:
                # 选逃生方向：离危险最远的分区（果断掉头，而不是贴着水挪），
                # 并列时取最靠中间的；优先保持上次选择（防抖）
                if self._recovery_zone not in safe_zones:
                    self._recovery_zone = max(
                        safe_zones,
                        key=lambda z: (
                            min(abs(ZONES.index(z) - ZONES.index(v))
                                for v in vetoed_zones),
                            -abs(ZONES.index(z) - ZONE_COUNT // 2),
                        ),
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
            # 防抖后的生效 hazards（计指标用）：类别 × 被否决区域
            active = [
                {"type": cls, "zone": z}
                for cls, z in hazards
                if z in vetoed_zones and (cls in VETO_CLASSES or cls == "unknown")
            ]
            self.pub_hazards.publish(String(data=json.dumps(
                {"state": self._state, "hazards": active}, ensure_ascii=False
            )))
        except Exception:
            return

        if veto_idx or self._state != "NORMAL":
            self.get_logger().info(
                f"state={self._state} danger={sorted(danger)} "
                f"vetoed_zones={sorted(vetoed_zones)} veto_paths={veto_idx}"
            )

    def destroy_node(self):
        self._stop_event.set()
        self._infer_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VLMSemanticAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
