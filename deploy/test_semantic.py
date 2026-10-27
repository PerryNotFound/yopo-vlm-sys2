import sys, json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
img_path = sys.argv[1] if len(sys.argv) > 1 else "rgb33.png"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

image = Image.open(img_path)
prompt = ("你是越野导航安全检查员。观察图像判断前方地面是否安全："
          "蓝色水面、裸露黄色沙土、深色泥泞沼泽均为危险地形。"
          "只输出JSON：{\"type\":\"goal\",\"safe\":false} 或 {\"type\":\"goal\",\"safe\":true}")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64)
resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("VLM 输出:", resp)
