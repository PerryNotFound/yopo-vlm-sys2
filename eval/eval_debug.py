#!/usr/bin/env python3
"""eval_debug.py — 基线 vs 微调 对 val 抽样帧的原始输出对比"""
import json, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
ADAPTER = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/checkpoints"
VAL = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/val.jsonl"
DS = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/data/semantic_ds/dataset.json"
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

recs = [json.loads(l) for l in open(VAL)]
ds = json.load(open(DS))
label_map = {r["rgb"]: (r["danger_ratio"] >= 0.01) for r in ds}
for r in recs:
    r["has_danger"] = label_map[r["image"]]
danger = [r for r in recs if r["has_danger"]][:3]
safe = [r for r in recs if not r["has_danger"]][:2]
samples = [(r["image"], "危险" if r["has_danger"] else "正常") for r in danger + safe]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)

for p, tag in samples:
    b = predict(base, processor, p)
    m = predict(model, processor, p)
    print(f"[{tag}] {p.split('/')[-1]}")
    print(f"  基线: {b}")
    print(f"  微调: {m}")
print("完成")
