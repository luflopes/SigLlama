"""Data collator for SigLlama mixed-modality batches.

Handles padding and stacking of visual, soft-token, and text tensors
into a single batch dict.
"""
from __future__ import annotations

from typing import Any

import torch


class SigLlamaCollator:
    """Collates a list of dataset items into a batched dict.

    Expects each item to have at minimum:
        - det_features:      [max_det, 6]
        - landmark_features: [num_landmarks * 2]

    Optional keys (present when processor/tokenizer are set):
        - pixel_values:  [C, H, W]
        - input_ids:     [max_text_length]
        - attention_mask:[max_text_length]
    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated: dict[str, Any] = {}

        collated["det_features"] = torch.stack([b["det_features"] for b in batch])
        collated["landmark_features"] = torch.stack([b["landmark_features"] for b in batch])

        if "pixel_values" in batch[0]:
            collated["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

        if "input_ids" in batch[0]:
            collated["input_ids"] = torch.stack([b["input_ids"] for b in batch])
            collated["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

        collated["image_ids"] = [b["image_id"] for b in batch]
        collated["captions"] = [b["caption"] for b in batch]

        return collated
