#!/usr/bin/env python3
"""train_qwen3vl.py v3 — QLoRA 微调 Qwen3-VL-4B-Instruct（YOPO 语义安全检查）"""
import json, os
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

MODEL_ID = "/home/admin0/YOPO-Rally-Whole/models/Qwen3-VL-4B-Instruct"
TRAIN = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/train.jsonl"
VAL = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/val.jsonl"
OUT = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/finetune/checkpoints"
IMG_SIZE = 448

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"

def load_recs(path):
    recs = [json.loads(l) for l in open(path)]
    data = []
    for r in recs:
        user_text = r["messages"][0]["content"][1]["text"]
        ans = r["messages"][1]["content"][0]["text"] + "<|im_end|>"
        image = Image.open(r["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}],
            tokenize=False, add_generation_prompt=True)
        ep = processor(text=prompt, images=image, return_tensors="pt", add_special_tokens=False)
        ea = processor(text=ans, return_tensors="pt", add_special_tokens=False)
        p_ids = ep["input_ids"][0]; a_ids = ea["input_ids"][0]
        data.append({
            "input_ids": torch.cat([p_ids, a_ids]),
            "attention_mask": torch.ones(p_ids.shape[0] + a_ids.shape[0], dtype=torch.long),
            "pixel_values": ep["pixel_values"][0],
            "image_grid_thw": ep["image_grid_thw"][0],
            "mm_token_type_ids": torch.cat([ep["mm_token_type_ids"][0], torch.zeros(a_ids.shape[0], dtype=torch.long)]),
            "labels": torch.cat([torch.full_like(p_ids, -100), a_ids]),
        })
    return data

class Collator:
    def __call__(self, batch):
        pad = processor.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad).long()
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch])
        if image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.squeeze(1)
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            [b["mm_token_type_ids"] for b in batch], batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": mm_token_type_ids, "labels": labels}

train_ds = load_recs(TRAIN); val_ds = load_recs(VAL)
print(f"train {len(train_ds)} 条 / val {len(val_ds)} 条")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  bias="none", task_type="CAUSAL_LM")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=1,
    per_device_eval_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine",
    logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
    bf16=True, remove_unused_columns=False, report_to="none",
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=Collator())
trainer.train()
trainer.save_model(OUT)
print("微调完成 →", OUT)
