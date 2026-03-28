"""PyTorch Dataset for SigLlama training.

Loads pre-extracted soft tokens (NDJSON) together with raw images and
captions, returning tensors ready for the model's forward pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SigLlamaDataset(Dataset):
    """Dataset that combines visual features, soft tokens, and captions.

    Parameters
    ----------
    ndjson_path : str
        Path to the NDJSON file produced by ``extract_soft_tokens.py``.
    image_root : str
        Root directory for resolving image paths.
    processor : callable or None
        SigLIP image processor (applies transforms and returns pixel_values).
    tokenizer : callable or None
        LLM tokenizer for encoding captions.
    max_detections : int
        Pad/truncate detections to this length.
    num_landmarks : int
        Expected number of face-mesh landmarks (468).
    max_text_length : int
        Maximum tokenised caption length.
    """

    def __init__(
        self,
        ndjson_path: str,
        image_root: str,
        processor: Any = None,
        tokenizer: Any = None,
        max_detections: int = 20,
        num_landmarks: int = 468,
        max_text_length: int = 256,
    ):
        self.image_root = Path(image_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_detections = max_detections
        self.num_landmarks = num_landmarks
        self.max_text_length = max_text_length

        self.samples: list[dict] = []
        with open(ndjson_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.samples[idx]

        item: dict[str, Any] = {
            "image_id": record["image_id"],
            "caption": record.get("caption", ""),
        }

        # --- Image (pixel_values) ---
        img_path = self.image_root / record["image_path"]
        if self.processor is not None and img_path.is_file():
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"]
            item["pixel_values"] = pixel_values.squeeze(0)

        # --- Detection tokens ---
        item["det_features"] = self._encode_detections(record.get("detection_tokens", []))

        # --- Landmark tokens ---
        item["landmark_features"] = self._encode_landmarks(record.get("landmark_tokens", []))

        # --- Text ---
        if self.tokenizer is not None:
            enc = self.tokenizer(
                record.get("caption", ""),
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)

        return item

    def _encode_detections(self, dets: list[dict]) -> torch.Tensor:
        """Encode detections as [cx, cy, w, h, class_id, confidence] tensor."""
        feat = torch.zeros(self.max_detections, 6)
        for i, det in enumerate(dets[: self.max_detections]):
            bbox = det["bbox"]
            feat[i] = torch.tensor([
                bbox[0], bbox[1], bbox[2], bbox[3],
                det["class_id"],
                det["confidence"],
            ])
        return feat

    def _encode_landmarks(self, lm_list: list[dict]) -> torch.Tensor:
        """Flatten normalised landmarks into a single vector."""
        vec = torch.zeros(self.num_landmarks * 2)
        if lm_list:
            pts = lm_list[0].get("landmarks_normalized", [])
            flat = np.array(pts).flatten()[: self.num_landmarks * 2]
            vec[: len(flat)] = torch.from_numpy(flat).float()
        return vec
