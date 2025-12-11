#!/usr/bin/env python3
"""
Train BLIP (image->caption) with DeepSpeed ZeRO-3 on ROCO.
Save this as train_deepspeed.py and run with `deepspeed --num_gpus=...`.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer
)
from datasets import load_dataset, Dataset as HFDataset

# ---- Utility: simple CSV-based dataset fallback ----
class RocoCSVImageCaptionDataset(Dataset):
    """
    Expects a CSV (or TSV) with columns: image_path, caption
    image_path can be a local path or a URL (PIL can handle many).
    """
    def __init__(self, csv_path, processor, image_column="image_path", caption_column="caption", transforms=None, max_length=64):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.image_col = image_column
        self.caption_col = caption_column
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row[self.image_col]).convert("RGB")
        caption = str(row[self.caption_col])
        # Processor will resize/normalize and tokenize text if needed
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        # Flatten batch dimension
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        return item

# ---- Collator ----
def collate_fn(batch):
    # Batch is list of dicts with pixel_values, input_ids, attention_mask
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

# ---- DeepSpeed config generator ----
def get_deepspeed_config(train_batch_size, gradient_accumulation_steps, fp16=True, offload=False):
    ds = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 100,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 5e-5, "warmup_num_steps": 500}
        },
        "fp16": {"enabled": fp16},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "wall_clock_breakdown": False
    }
    if offload:
        ds["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
        ds["zero_optimization"]["offload_param"]["device"] = "cpu"
    return ds

# ---- Main training function ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to ROCO CSV or use HF dataset identifier")
    parser.add_argument("--hf_dataset", type=str, default="eltorio/ROCOv2-radiology", help="Optional HF dataset id if you want to load ROCO directly")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip-image-captioning-base", help="Pretrained BLIP model id")
    parser.add_argument("--output_dir", type=str, default="./outputs/blip_roco_z3")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 via DeepSpeed")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload for ZeRO")
    parser.add_argument("--save_interval_steps", type=int, default=2000)
    args = parser.parse_args()

    # ---- Prepare model + processor ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(args.model_id)   # tokenization + image transforms
    model = BlipForConditionalGeneration.from_pretrained(args.model_id)

    # Important: defer moving to device - DeepSpeed will handle device placement via deepspeed.init
    model.config.max_length = args.max_length

    # ---- Dataset loading ----
    if args.data_path:
        # CSV fallback
        dataset = RocoCSVImageCaptionDataset(args.data_path, processor, max_length=args.max_length)
        dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        # Try to use Hugging Face dataset (ROCOv2 radiology) if available
        print(f"Loading HF dataset: {args.hf_dataset}")
        ds = load_dataset(args.hf_dataset, split="train")
        # Convert to HF Dataset with images loaded as PIL (the dataset may already have image column)
        def preprocess_hf(example):
            # The HF processor expects PIL images; ensure the dataset image column is a PIL image
            # The 'img' or 'image' column name may vary; common names: 'image', 'img'
            image_field = "image" if "image" in example else ("img" if "img" in example else None)
            if image_field is None:
                raise ValueError("HF dataset does not contain an 'image' or 'img' column. Provide --data_path instead.")
            img = example[image_field]
            caption_field = "caption" if "caption" in example else ("text" if "text" in example else None)
            if caption_field is None:
                raise ValueError("HF dataset missing caption column.")
            return {"pixel_values": processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0),
                    "input_ids": processor(text=example[caption_field], return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length)["input_ids"].squeeze(0),
                    "attention_mask": processor(text=example[caption_field], return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length)["attention_mask"].squeeze(0)}
        # Map may be slow; better to use HF in streaming or preprocessed; here we keep things simple:
        ds = ds.map(lambda ex: {"caption": ex["caption"] if "caption" in ex else ex.get("text", "")}, remove_columns=[c for c in ds.column_names if c not in ("image","caption")])
        # Wrap HF dataset into DataLoader via torch
        def hf_collate(features):
            pixel_values = torch.stack([processor(images=f["image"], return_tensors="pt")["pixel_values"].squeeze(0) for f in features])
            texts = [f["caption"] for f in features]
            texts_tokens = processor(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length)
            return {"pixel_values": pixel_values, "input_ids": texts_tokens["input_ids"], "attention_mask": texts_tokens["attention_mask"]}
        dataloader = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=hf_collate)

    # ---- DeepSpeed initialization ----
    try:
        import deepspeed
    except ImportError:
        raise ImportError("deepspeed is required. Install with `pip install deepspeed`")

    ds_config = get_deepspeed_config(args.train_batch_size, args.gradient_accumulation_steps, fp16=args.fp16, offload=args.offload)
    os.makedirs(args.output_dir, exist_ok=True)
    # Save config for reference
    with open(os.path.join(args.output_dir, "ds_config.json"), "w") as f:
        json.dump(ds_config, f, indent=2)

    # Replace optimizer and let DeepSpeed create engine
    # Create a "dummy" optimizer here; deepspeed.initialize will replace it if optimizer config present in ds_config
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # ---- Training loop ----
    global_step = 0
    model_engine.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # Move inputs to device of model_engine
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(pixel_values=batch["pixel_values"], labels=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            if global_step % 50 == 0 and model_engine.local_rank == 0:
                print(f"[Epoch {epoch}][Step {step}] loss: {loss.item():.4f}")

            if global_step % args.save_interval_steps == 0 and model_engine.local_rank == 0 and global_step > 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model_engine.save_checkpoint(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

            global_step += 1

    # final save
    if model_engine.local_rank == 0:
        model_engine.save_checkpoint(args.output_dir)
        print("Training complete. Final checkpoint saved.")

if __name__ == "__main__":
    main()
