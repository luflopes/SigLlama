"""DD-VQA JSONL dataset for Stage 2+ fine-tuning (from scripts/prepare_ddvqa.py).

Backbone-agnostic: receives tokenizer and image_processor separately plus
an explicit ``backbone`` flag so the prompt formatting matches the LLM.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .prompt_formats import format_vqa_prompt

logger = logging.getLogger(__name__)


def _sample_is_real(row: dict) -> bool:
    """Robustly read the real/fake flag from a DD-VQA row.

    The prepare script writes ``is_real`` (boolean). Older/manual metadata
    files may use a string ``label`` field with values like "real"/"fake".
    """
    if "is_real" in row:
        return bool(row["is_real"])
    return str(row.get("label", "")).lower() == "real"


def _sample_label_str(row: dict) -> str:
    return "real" if _sample_is_real(row) else "fake"


class DDVQADataset(Dataset):
    """JSONL with ``image``, ``question``, ``answer``, ``is_real``/``label``, ``method``."""

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer: Any,
        image_processor: Any,
        dino_transform: Any,
        backbone: str = "paligemma",
        max_length: int = 256,
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dino_transform = dino_transform
        self.backbone = backbone
        self.max_length = max_length

        self.samples: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_real = sum(1 for s in self.samples if _sample_is_real(s))
        n_fake = len(self.samples) - n_real
        logger.info(
            "Loaded %d samples from %s (real=%d, fake=%d, backbone=%s)",
            len(self.samples), metadata_path, n_real, n_fake, backbone,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def is_real(self, idx: int) -> bool:
        return _sample_is_real(self.samples[idx])

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
        image_id = img_rel if img_rel else os.path.basename(img_path)
        full_text, prefix_text = format_vqa_prompt(question, answer, self.backbone)

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

        prefix_enc = self.tokenizer(prefix_text, add_special_tokens=True)
        prefix_len = len(prefix_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        pixel_values_dino = self.dino_transform(img)

        label_str = _sample_label_str(row)
        method = row.get("method", "unknown")

        return {
            "pixel_values_siglip": pixel_values_siglip,
            "pixel_values_dino": pixel_values_dino,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_str": label_str,
            "method": method,
            "image": image_id,
            "question": question,
            "answer": answer,
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
    for meta_key in ("label_str", "method", "image", "question", "answer"):
        out[meta_key] = [b[meta_key] for b in batch]
    return out
