"""LCS-558K image-captioning dataset for Stage 1 adapter pre-training.

Backbone-agnostic: accepts tokenizer and image_processor separately
(from ``training.factory.build_processor_and_transforms``) plus an
explicit ``backbone`` flag so the prompt formatting is consistent with
the downstream LLM.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .prompt_formats import format_caption_prompt

logger = logging.getLogger(__name__)


def _extract_caption(item: dict) -> str:
    if "blip_caption" in item:
        return item["blip_caption"]
    if "conversations" in item:
        for turn in item["conversations"]:
            if turn.get("from") == "gpt":
                return turn.get("value", "")
    return item.get("caption", "")


class LCS558KDataset(Dataset):
    """JSON array of ``{image, blip_caption}`` or ``{image, conversations}``."""

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer: Any,
        image_processor: Any,
        dino_transform: Any,
        backbone: str = "paligemma",
        max_length: int = 128,
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dino_transform = dino_transform
        self.backbone = backbone
        self.max_length = max_length

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: list[dict] = []
        skipped = 0
        for item in raw:
            img_rel = item.get("image", "")
            img_path = os.path.join(image_root, img_rel)
            if not os.path.isfile(img_path):
                skipped += 1
                continue
            caption = _extract_caption(item)
            if not caption:
                skipped += 1
                continue
            self.samples.append({"image_path": img_path, "caption": caption})

        logger.info(
            "Loaded %d samples from %s (skipped %d, backbone=%s)",
            len(self.samples), metadata_path, skipped, backbone,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.samples[idx]
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        text = format_caption_prompt(sample["caption"], self.backbone)

        pixel_values_siglip = self.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        token_ids: list[int] = self.tokenizer(
            text, add_special_tokens=True,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not token_ids or token_ids[-1] != eos_id):
            token_ids.append(int(eos_id))

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        real_len = len(token_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id or 0
        token_ids = token_ids + [pad_id] * (self.max_length - real_len)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:real_len] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        pixel_values_dino = self.dino_transform(img)

        return {
            "pixel_values_siglip": pixel_values_siglip,
            "pixel_values_dino": pixel_values_dino,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_skip_none(batch: list) -> dict[str, torch.Tensor] | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
