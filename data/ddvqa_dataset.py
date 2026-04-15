"""DD-VQA JSONL dataset for Stage 2+ fine-tuning (from scripts/prepare_ddvqa.py).

Uses the tokenizer and image_processor from PaliGemma separately to
avoid injecting <image> placeholder tokens (visual features are
handled externally via DINOv2 + I-MoF in FaceGroundVLM).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DDVQADataset(Dataset):
    """JSONL with ``image``, ``question``, ``answer``, ``label``, ``method``."""

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        processor: Any,
        dino_transform: Any,
        max_length: int = 256,
    ):
        self.image_root = image_root
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.dino_transform = dino_transform
        self.max_length = max_length

        self.samples: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_real = sum(1 for s in self.samples if s.get("label", "").lower() == "real")
        n_fake = len(self.samples) - n_real
        logger.info(
            "Loaded %d samples from %s (real=%d, fake=%d)",
            len(self.samples), metadata_path, n_real, n_fake,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def is_real(self, idx: int) -> bool:
        return self.samples[idx].get("label", "").lower() == "real"

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        row = self.samples[idx]
        img_rel = row.get("image", "")
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(self.image_root, img_rel)
        if not os.path.isfile(img_path):
            return None

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        question = row.get("question", "")
        answer = row.get("answer", "")
        full_text = f"{question}\n{answer}"

        pixel_values_siglip = self.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prefix_enc = self.tokenizer(
            f"{question}\n",
            add_special_tokens=True,
        )
        prefix_len = len(prefix_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        pixel_values_dino = self.dino_transform(img)

        label_str = row.get("label", "fake").lower()
        method = row.get("method", "unknown")

        return {
            "pixel_values_siglip": pixel_values_siglip,
            "pixel_values_dino": pixel_values_dino,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_str": label_str,
            "method": method,
        }


def collate_ddvqa(batch: list) -> dict[str, Any] | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    tensor_keys = [
        "pixel_values_siglip", "pixel_values_dino",
        "input_ids", "attention_mask", "labels",
    ]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["label_str"] = [b["label_str"] for b in batch]
    out["method"] = [b["method"] for b in batch]
    return out
