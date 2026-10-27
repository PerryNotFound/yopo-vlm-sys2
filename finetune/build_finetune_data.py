#!/usr/bin/env python3
"""build_finetune_data.py v2 — 明显标签(≥1%危险 / <0.1%安全) + 剔除模糊帧 + 危险帧平衡复制"""
import json, random, os
random.seed(42)
DS = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/data/semantic_ds/dataset.json"
OUT = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune"
os.makedirs(OUT, exist_ok=True)
DANGER_MIN = 0.01
SAFE_MAX = 0.001
PROMPT = ("你是越野导航安全检查员。观察图像判断前方地面是否安全："
          "蓝色水面、裸露黄色沙土、深色泥泞沼泽均为危险地形。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")
def make_rec(r):
    safe = r["danger_ratio"] < SAFE_MAX
    answer = '{"type":"goal","safe":true}' if safe else '{"type":"goal","safe":false}'
    return {"image": r["rgb"], "messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}
recs = json.load(open(DS))
recs = [r for r in recs if r["danger_ratio"] >= DANGER_MIN or r["danger_ratio"] < SAFE_MAX]
print(f"筛选后 {len(recs)} 帧 (危险 {sum(1 for r in recs if r['danger_ratio']>=DANGER_MIN)} / 安全 {sum(1 for r in recs if r['danger_ratio']<SAFE_MAX)})")
random.shuffle(recs)
n_val = max(1, len(recs) // 5)
val, train = recs[:n_val], recs[n_val:]
safe_train = [r for r in train if r["danger_ratio"] < SAFE_MAX]
danger_train = [r for r in train if r["danger_ratio"] >= DANGER_MIN]
if safe_train and danger_train:
    if len(danger_train) > len(safe_train):
        safe_train = safe_train * (len(danger_train) // len(safe_train))
    else:
        danger_train = danger_train * (len(safe_train) // len(danger_train))
train = danger_train + safe_train
random.shuffle(train)
with open(f"{OUT}/train.jsonl", "w") as f:
    for r in train: f.write(json.dumps(make_rec(r), ensure_ascii=False) + "\n")
with open(f"{OUT}/val.jsonl", "w") as f:
    for r in val: f.write(json.dumps(make_rec(r), ensure_ascii=False) + "\n")
print(f"train {len(train)} 条 (危险 {len(danger_train)} / 安全 {len(safe_train)})")
print(f"val   {len(val)} 条")
