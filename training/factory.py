"""Backbone-aware factory for text+image preprocessing components.

Both FaceGroundVLM (PaliGemma2) and TinyLLaVAGroundVLM need a tokenizer,
an image processor (for SigLIP-side features), and a DINOv2 transform.
These differ in resolution and tokenizer family, but the downstream
datasets consume a uniform protocol::

    tokenizer(text, ...)            -> input_ids, attention_mask
    image_processor(images=..., return_tensors="pt")["pixel_values"]
    dino_transform(PIL.Image) -> [3, H, W]

This module centralises the construction.
"""
from __future__ import annotations

import logging
from typing import Any

from torchvision.transforms import Compose, Normalize, Resize, ToTensor

logger = logging.getLogger(__name__)


# ImageNet stats used by DINOv2 pre-processing.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_dino_transform(image_size: int) -> Any:
    return Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_processor_and_transforms(cfg: dict) -> dict[str, Any]:
    """Return a dict with keys: ``tokenizer``, ``image_processor``,
    ``dino_transform``, ``processor`` (full processor when available,
    else ``None``), and ``backbone``.
    """
    backbone = cfg.get("backbone", "paligemma").lower()

    if backbone == "paligemma":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["paligemma_model"])
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        image_size = 448
        dino_transform = _build_dino_transform(image_size)
        logger.info("Built PaliGemma processor (img=%d)", image_size)
        return {
            "backbone": backbone,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "dino_transform": dino_transform,
            "processor": processor,
            "image_size": image_size,
        }

    if backbone == "tinyllava":
        from transformers import AutoImageProcessor, AutoTokenizer
        tokenizer_id = cfg.get(
            "tokenizer_id", cfg.get(
                "tinyllama_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if tokenizer.pad_token is None:
            # TinyLlama doesn't ship a pad token; reuse EOS to enable padding.
            tokenizer.pad_token = tokenizer.eos_token
        # Use the right-pad convention expected by causal LMs.
        tokenizer.padding_side = "right"

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.get("siglip_model", "google/siglip-so400m-patch14-384"),
        )
        image_size = int(cfg.get("image_size", 384))
        dino_transform = _build_dino_transform(image_size)
        logger.info("Built TinyLLaVA processor (img=%d)", image_size)
        return {
            "backbone": backbone,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "dino_transform": dino_transform,
            "processor": None,
            "image_size": image_size,
        }

    raise ValueError(f"Unknown backbone '{backbone}'")
