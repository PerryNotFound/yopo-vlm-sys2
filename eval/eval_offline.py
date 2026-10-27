#!/usr/bin/env python3
"""eval_offline.py — 基线(预训练) vs 微调 在 val 集上的语义安全判断准确率
用法: source venv && python eval/eval_offline.py
"""
import json, re
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
ADAPTER = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/checkpoints"
VAL = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/val.jsonl"
PROMPT = ("你是越野导航安全检查员。观察图像判断前方地面是否安全："
          "蓝色水面、裸露黄色沙土、深色泥泞沼泽均为危险地形。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")

def predict(model, processor, img_path):
    image = Image.open(img_path).convert("RGB").resize((448, 448))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def parse(resp):
    m = re.search(r'"safe"\s*:\s*(true|false)', resp)
    return (m.group(1) == "true") if m else None

def run(model, processor, recs, name):
    tp = tn = fp = fn = 0; bad = []
    for r in recs:
        resp = predict(model, processor, r["image"])
        pred = parse(resp); gt = not r["has_danger"]
        if pred is None: bad.append((r["frame"], resp)); continue
        if pred == gt:
            tp += 1 if pred else 0; tn += 1 if not pred else 0
        else:
            fp += 1 if pred else 0; fn += 1 if not pred else 0
    acc = (tp + tn) / max(1, len(recs))
    print(f"[{name}] 准确率 {acc:.3f}  (TP {tp} / TN {tn} / FP {fp} / FN {fn} / 共 {len(recs)})")
    if bad: print("  解析失败示例:", bad[:3])
recs = [json.loads(l) for l in open(VAL)]
ds = json.load(open("/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/data/semantic_ds/dataset.json"))
label_map = {r["rgb"]: (r["danger_ratio"] >= 0.01) for r in ds}
for r in recs:
    r["has_danger"] = label_map[r["image"]]
print(f"val 共 {len(recs)} 帧, 含危险 {sum(1 for r in recs if r['has_danger'])} 帧")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
run(base, processor, recs, "基线(预训练)")

model = PeftModel.from_pretrained(base, ADAPTER)
run(model, processor, recs, "微调(YOPO+VLM)")
print("评测完成")
