"""Deepfake VQA dataset for Stage 2 fine-tuning.

Loads face-cropped images and question-answer pairs from DD-VQA
(prepared by ``scripts/prepare_ddvqa.py``).

Input sequence layout::

    [visual_patches]  <s> {question}\n{answer} </s> [pad...]
                      |--- label=-100 ---|--- predicted ---|

Only the answer tokens contribute to the loss.
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


class DeepfakeVQADataset(Dataset):
    """Image + question + answer dataset for deepfake explanation.

    Parameters
    ----------
    metadata_path : str
        Path to a JSONL file where each line is::

            {"image": "...", "question": "...", "answer": "...",
             "method": "...", "is_real": true/false, ...}
    image_root : str
        Root directory for face-cropped images (the ``frames/`` subdir).
    processor : SiglipImageProcessor
        HuggingFace image processor for SigLIP.
    tokenizer : PreTrainedTokenizer
        LLM tokenizer (e.g. LlamaTokenizer).
    max_length : int
        Maximum total tokens for the text sequence.
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        processor: Any,
        tokenizer: Any,
        max_length: int = 256,
    ):
        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: list[dict[str, Any]] = []
        skipped = 0

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                img_path = os.path.join(image_root, item["image"])
                if not os.path.isfile(img_path):
                    skipped += 1
                    continue

                self.samples.append({
                    "image_path": img_path,
                    "question": item["question"],
                    "answer": item["answer"],
                    "method": item.get("method", ""),
                    "is_real": item.get("is_real", False),
                })

        logger.info(
            "Loaded %d VQA samples from %s (skipped %d missing images)",
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

        question = sample["question"]
        answer = sample["answer"]

        full_text = f"{question}\n{answer}"

        question_ids = self.tokenizer(
            question, add_special_tokens=False
        )["input_ids"]
        prompt_len = len(question_ids) + 1  # +1 for BOS

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_vqa(
    batch: list[dict[str, torch.Tensor] | None],
) -> dict[str, torch.Tensor] | None:
    """Stack a batch of VQA samples, silently dropping ``None`` entries."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
