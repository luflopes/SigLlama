"""Backbone-aware factory for text+image preprocessing components.

TinyLLaVAGroundVLM needs a tokenizer, an image processor (for SigLIP-side
features), and a DINOv2 transform. The downstream datasets consume a
uniform protocol::

    tokenizer(text, ...)            -> input_ids, attention_mask
    image_processor(images=..., return_tensors="pt")["pixel_values"]
    dino_transform(PIL.Image) -> [3, H, W]

This module centralises the construction.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from torchvision import transforms as T

logger = logging.getLogger(__name__)


# ImageNet stats used by DINOv2 pre-processing.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_dino_transform(image_size: int) -> Any:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_augment_transform(cfg: dict) -> Callable | None:
    """Build a PIL-level augmentation applied before model preprocessing.

    Controlled by the ``augmentation`` config key (default ``false``).
    Only colour/photometric transforms are used — no geometric transforms
    — so bounding-box annotations in Stage 3 remain valid.

    Returns ``None`` when augmentation is disabled.
    """
    if not cfg.get("augmentation", False):
        return None

    aug = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
        T.RandomGrayscale(p=0.05),
    ])
    logger.info("Augmentation enabled (color-only, bbox-safe)")
    return aug


def build_processor_and_transforms(cfg: dict) -> dict[str, Any]:
    """Return a dict with keys: ``tokenizer``, ``image_processor``,
    ``dino_transform``, ``processor`` (always ``None``), and ``backbone``.

    The ``backbone`` parameter is kept for backward compatibility with
    older configs that pass ``backbone: "tinyllava"`` explicitly.
    """
    backbone = cfg.get("backbone", "tinyllava").lower()

    if backbone != "tinyllava":
        raise ValueError(
            f"Unknown backbone '{backbone}'. Only 'tinyllava' is supported."
        )

    from transformers import AutoImageProcessor, AutoTokenizer
    tokenizer_id = cfg.get(
        "tokenizer_id", cfg.get(
            "tinyllama_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    image_processor = AutoImageProcessor.from_pretrained(
        cfg.get("siglip_model", "google/siglip-so400m-patch14-384"),
    )
    image_size = int(cfg.get("image_size", 384))
    dino_transform = _build_dino_transform(image_size)
    augment_transform = build_augment_transform(cfg)
    logger.info("Built TinyLLaVA processor (img=%d, aug=%s)", image_size, augment_transform is not None)
    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "dino_transform": dino_transform,
        "augment_transform": augment_transform,
        "processor": None,
        "image_size": image_size,
    }
