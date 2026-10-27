#!/usr/bin/env python3
"""sample_hsv.py — HSV 标定工具：标定 m2_veto.py / m3_fusion.py 的 HSV_WATER 等阈值

两种用法：

1. 交互模式（默认）：抓一帧相机图像，弹窗框选水面区域，输出 HSV 统计
    python deploy/sample_hsv.py

2. 无 GUI 模式（窗口闪退时用）：对图像地面部分做 K-means 颜色聚类，
   打印每类的 HSV 范围与占比，并保存着色图 hsv_clusters.png 和原图 frame.png，
   对照图片找出哪一类是水面即可
    python deploy/sample_hsv.py --nogui
"""

import os
import sys

# Wayland 下 conda/opencv 的 Qt 插件常缺 wayland 后端，强制 xcb 防窗口闪退
if os.environ.get("WAYLAND_DISPLAY") and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

GROUND_ROW_FRAC = 0.45  # 与 m2_veto.py / m3_fusion.py 保持一致
N_CLUSTERS = 5


class FrameGrabber(Node):
    def __init__(self):
        super().__init__("hsv_sampler")
        self.frame = None
        self.create_subscription(
            CompressedImage, "/camera/color/image/compressed", self._cb, 10
        )

    def _cb(self, msg):
        self.frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def grab_frame():
    rclpy.init()
    node = FrameGrabber()
    print("等待相机图像...")
    while node.frame is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    img = node.frame.copy()
    node.destroy_node()
    rclpy.shutdown()
    return img


def print_hsv_stats(hsv_pixels, prefix=""):
    """hsv_pixels: (N,3) float64，打印 P5/P95 与建议阈值"""
    p5 = np.percentile(hsv_pixels, 5, axis=0)
    p95 = np.percentile(hsv_pixels, 95, axis=0)
    mean = hsv_pixels.mean(axis=0)
    print(f"{prefix}H: {p5[0]:6.1f}~{p95[0]:6.1f}  S: {p5[1]:6.1f}~{p95[1]:6.1f}  "
          f"V: {p5[2]:6.1f}~{p95[2]:6.1f}  (mean H={mean[0]:.0f} S={mean[1]:.0f} V={mean[2]:.0f})")
    print(f"{prefix}建议阈值: (({max(0, int(p5[0] - 5))}, {max(0, int(p5[1] - 20))}, "
          f"{max(0, int(p5[2] - 20))}), ({min(179, int(p95[0] + 5))}, 255, 255))")


def gui_mode(img):
    print("框选目标区域后按回车确认（可多次框选），按 ESC 退出")
    while True:
        try:
            roi = cv2.selectROI("框选水面/危险区域", img, fromCenter=False)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"\nGUI 创建窗口失败（{e.err}）。请改用无 GUI 模式：")
            print("    python deploy/sample_hsv.py --nogui")
            return
        x, y, w, h = (int(v) for v in roi)
        if w == 0 or h == 0:
            break
        crop = img[y: y + h, x: x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        print(f"ROI {w}x{h} @({x},{y}), {len(hsv)} px")
        print_hsv_stats(hsv, prefix="  ")


def nogui_mode(img):
    h, w = img.shape[:2]
    ground = img[int(h * GROUND_ROW_FRAC):, :]
    cv2.imwrite("frame.png", img)
    print(f"原图已保存 frame.png（地面部分为下 {1 - GROUND_ROW_FRAC:.0%}）")

    # K-means 聚类地面区域的 HSV
    hsv = cv2.cvtColor(ground, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, N_CLUSTERS, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    # 着色图：每个像素染成所属类别的中心颜色
    centers_bgr = cv2.cvtColor(
        centers.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_HSV2BGR
    ).reshape(-1, 3)
    painted = centers_bgr[labels].reshape(ground.shape).astype(np.uint8)
    cv2.imwrite("hsv_clusters.png", painted)
    print("聚类着色图已保存 hsv_clusters.png（同色块=同一类）\n")

    order = np.argsort([-(labels == i).sum() for i in range(N_CLUSTERS)])
    for rank, i in enumerate(order):
        cnt = int((labels == i).sum())
        print(f"类别 {i}（占比 {cnt / len(labels):.1%}）:")
        print_hsv_stats(hsv.reshape(-1, 3)[labels == i].astype(np.float64), prefix="  ")
    print("\n对照 hsv_clusters.png 找出水面所属的类别编号，把它的建议阈值填入")
    print("m2_veto.py 和 m3_fusion.py 的 HSV_WATER。反光水面可能横跨多个类别，")
    print("取各类范围的并集，或把 S/V 下限放宽。")


def main():
    img = grab_frame()
    if "--nogui" in sys.argv:
        nogui_mode(img)
    else:
        gui_mode(img)


if __name__ == "__main__":
    main()
