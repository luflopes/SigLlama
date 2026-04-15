"""Utilities for saving visual samples during training.

Saves denormalized images alongside generated and reference texts
so training quality can be visually inspected.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# DINOv2 / ImageNet normalization constants
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm_dino(t: torch.Tensor) -> torch.Tensor:
    """Inverse ImageNet normalization on a [3, H, W] tensor."""
    return (t.cpu().float() * _DINO_STD + _DINO_MEAN).clamp(0, 1)


def decode_ids(token_ids: torch.Tensor, tokenizer) -> list[str]:
    """Decode token ids, replacing -100 (label mask) with pad before decoding."""
    ids = token_ids.clone()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids[ids == -100] = pad
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def save_samples(
    pixel_values_dino: torch.Tensor,
    generated_texts: list[str],
    reference_texts: list[str],
    output_dir: Path,
    step: int,
    max_samples: int = 4,
) -> None:
    """Save sample images + texts to ``output_dir/samples/step_<N>/``."""
    sample_dir = output_dir / "samples" / f"step_{step}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(generated_texts), max_samples, pixel_values_dino.shape[0])

    lines: list[str] = []
    for i in range(n):
        img = TF.to_pil_image(_denorm_dino(pixel_values_dino[i]))
        fname = f"img_{i}.jpg"
        img.save(sample_dir / fname, quality=90)

        lines.append(f"=== {fname} ===")
        lines.append(f"Generated : {generated_texts[i]}")
        lines.append(f"Reference : {reference_texts[i]}")
        lines.append("")

    (sample_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %d sample images to %s", n, sample_dir)
