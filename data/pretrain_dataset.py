"""Image-caption dataset for Stage 1 adapter pre-training.

Loads images and BLIP captions from LCS-558K (or any dataset with the same
metadata format).  No soft tokens or landmarks at this stage.
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


def _extract_caption(item: dict) -> str:
    """Extract caption text from either metadata format.

    Priority: ``blip_caption`` field > ``conversations`` GPT turn > ``caption``.
    """
    if "blip_caption" in item:
        return item["blip_caption"]

    if "conversations" in item:
        for turn in item["conversations"]:
            if turn.get("from") == "gpt":
                return turn.get("value", "")

    return item.get("caption", "")


class PretrainDataset(Dataset):
    """Simple image + caption dataset for visual-language alignment.

    Supports two LCS-558K metadata formats:

    **Meta format** (``blip_laion_cc_sbu_558k_meta.json``)::

        [{"id": "...", "image": "path.jpg", "blip_caption": "..."}, ...]

    **Conversations format** (``blip_laion_cc_sbu_558k.json``)::

        [{"id": "...", "image": "path.jpg",
          "conversations": [{"from": "human", ...}, {"from": "gpt", "value": "caption"}]}, ...]

    Parameters
    ----------
    metadata_path : str
        Path to the JSON metadata file (either format).
    image_root : str
        Root directory that ``image`` paths are relative to.
    processor : SiglipImageProcessor
        HuggingFace image processor for SigLIP.
    tokenizer : PreTrainedTokenizer
        LLM tokenizer (e.g. LlamaTokenizer).
    max_length : int
        Maximum number of caption tokens (including BOS/EOS).
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        processor: Any,
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: list[dict[str, str]] = []
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
            "Loaded %d samples from %s (skipped %d)",
            len(self.samples), metadata_path, skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None:
        sample = self.samples[idx]

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        encoding = self.tokenizer(
            sample["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[0] = -100  # BOS is given, not predicted

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_skip_none(
    batch: list[dict[str, torch.Tensor] | None],
) -> dict[str, torch.Tensor] | None:
    """Stack a batch of samples, silently dropping ``None`` entries."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
