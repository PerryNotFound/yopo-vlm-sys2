#!/usr/bin/env python3
"""sample_hsv.py — HSV 标定工具：抓一帧相机图像，框选区域，输出 HSV 统计

用途：更换危险地形纹理（如真实水面材质）后，重新标定 m2_veto.py / m3_fusion.py
里的 HSV_WATER / HSV_SAND 阈值。

用法：
    python deploy/sample_hsv.py
    # 弹窗后框选水面区域，回车确认，可框多次，按 ESC 退出
    # 输出的 P5~P95 范围可直接填入 HSV 阈值（略加余量）
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class FrameGrabber(Node):
    def __init__(self):
        super().__init__("hsv_sampler")
        self.frame = None
        self.create_subscription(
            CompressedImage, "/camera/color/image/compressed", self._cb, 10
        )

    def _cb(self, msg):
        self.frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def main():
    rclpy.init()
    node = FrameGrabber()
    print("等待相机图像...")
    while node.frame is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    img = node.frame.copy()
    node.destroy_node()
    rclpy.shutdown()

    print("框选目标区域后按回车确认（可多次框选），按 ESC 退出")
    while True:
        roi = cv2.selectROI("框选水面/危险区域", img, fromCenter=False)
        cv2.destroyAllWindows()
        x, y, w, h = (int(v) for v in roi)
        if w == 0 or h == 0:
            break
        crop = img[y: y + h, x: x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        print(f"ROI {w}x{h} @({x},{y}), {len(hsv)} px")
        for i, name in enumerate(("H", "S", "V")):
            lo, hi = np.percentile(hsv[:, i], 5), np.percentile(hsv[:, i], 95)
            print(f"  {name}: P5={lo:6.1f}  P95={hi:6.1f}  mean={hsv[:, i].mean():6.1f}")
        print("  建议阈值（P5/P95 加余量）:")
        h_lo, h_hi = np.percentile(hsv[:, 0], 5), np.percentile(hsv[:, 0], 95)
        s_lo = np.percentile(hsv[:, 1], 5)
        v_lo = np.percentile(hsv[:, 2], 5)
        print(f"  (({max(0, int(h_lo - 5))}, {max(0, int(s_lo - 20))}, "
              f"{max(0, int(v_lo - 20))}), ({min(179, int(h_hi + 5))}, 255, 255))")


if __name__ == "__main__":
    main()
