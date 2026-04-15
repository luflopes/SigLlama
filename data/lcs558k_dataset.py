"""LCS-558K image-captioning dataset for Stage 1 adapter pre-training."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

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
        processor: Any,
        dino_transform: Any,
        max_length: int = 128,
    ):
        self.image_root = image_root
        self.processor = processor
        self.dino_transform = dino_transform
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
            "Loaded %d samples from %s (skipped %d)", len(self.samples), metadata_path, skipped
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.samples[idx]
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        caption = sample["caption"]

        proc_out = self.processor(
            images=img,
            text="",
            suffix=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        pixel_values_siglip = proc_out["pixel_values"].squeeze(0)
        input_ids = proc_out["input_ids"].squeeze(0)
        attention_mask = proc_out["attention_mask"].squeeze(0)

        if "labels" in proc_out and proc_out["labels"] is not None:
            labels = proc_out["labels"].squeeze(0)
        else:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            labels[0] = -100

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
