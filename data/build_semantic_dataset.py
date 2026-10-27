#!/usr/bin/env python3
"""build_semantic_dataset.py — 把 YOPO-Sim 生成的 rgb/seg 打包成语义微调数据集
用法: python build_semantic_dataset.py <TrainingData路径> <输出目录>
输出: cls_*.png 类别图(0=地面 1=水面 2=沙地 3=泥地) + dataset.json
"""
import argparse, glob, json, os, re
import numpy as np
from PIL import Image

SEM_COLORS = {
    1: np.array([0, 0, 255], dtype=int),     # 水面 蓝
    2: np.array([255, 255, 0], dtype=int),   # 沙地 黄
    3: np.array([140, 69, 18], dtype=int),   # 泥地 棕
}
TOL = 45  # 容差

def seg_to_class(seg_rgb):
    cls = np.zeros(seg_rgb.shape[:2], dtype=np.uint8)
    for c, ref in SEM_COLORS.items():
        dist = np.abs(seg_rgb.astype(int) - ref).sum(axis=2)
        cls[dist <= TOL] = c
    return cls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for sc in sorted(glob.glob(os.path.join(args.data_dir, "Scene_*"))):
        tex = os.path.join(sc, "Textures")
        if not os.path.isdir(tex):
            continue
        for seg_p in sorted(set(glob.glob(os.path.join(tex, "seg_*.png")) + glob.glob(os.path.join(tex, "seg*.png")))):
            m = re.search(r"seg_?(\d+)\.png$", seg_p)
            if not m:
                continue
            n = m.group(1)
            rgb_p = os.path.join(tex, f"rgb{n}.png")
            if not os.path.exists(rgb_p):
                rgb_p = os.path.join(tex, f"rgb_{n}.png")
            if not os.path.exists(rgb_p):
                continue
            seg = np.array(Image.open(seg_p).convert("RGB"))
            cls = seg_to_class(seg)
            out_cls = os.path.join(args.out_dir, f"cls_{n}.png")
            Image.fromarray(cls).save(out_cls)
            counts = {int(c): int((cls == c).sum()) for c in [0, 1, 2, 3]}
            danger = counts[1] + counts[2] + counts[3]
            records.append({
                "frame": n, "scene": os.path.basename(sc),
                "rgb": rgb_p, "seg": seg_p, "cls": out_cls,
                "class_pixels": counts,
                "danger_ratio": round(danger / cls.size, 4),
                "has_danger": danger > 0,
            })
    records.sort(key=lambda r: int(r["frame"]))
    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    nd = sum(1 for r in records if r["has_danger"])
    print(f"共 {len(records)} 帧, 含危险地形 {nd} 帧")
    print("类别分布: " + ", ".join(
        f"类{c}={sum(r['class_pixels'][c] for r in records)}" for c in [0, 1, 2, 3]))

if __name__ == "__main__":
    main()
