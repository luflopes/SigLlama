"""FaceGroundVLM model components and backbone factory."""
from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def build_model(cfg: dict, *, use_lora: bool | None = None) -> nn.Module:
    """Instantiate the VLM backbone selected by ``cfg['backbone']``.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config. Must contain ``backbone`` ∈ {"paligemma",
        "tinyllava"} plus the per-backbone hyper-parameters.
    use_lora : bool | None
        Override the ``use_lora`` flag (stage 1 disables it; stage 2+
        enables it). When ``None`` we infer from ``cfg.get('use_lora', False)``.
    """
    backbone = cfg.get("backbone", "paligemma").lower()
    do_lora = cfg.get("use_lora", False) if use_lora is None else use_lora

    first_token_w = float(cfg.get("first_token_loss_weight", 1.0))

    if backbone == "paligemma":
        from .face_ground_vlm import FaceGroundVLM
        model = FaceGroundVLM(
            paligemma_model=cfg["paligemma_model"],
            dinov2_model=cfg["dinov2_model"],
            mof_strategy=cfg.get("mof_strategy", "interleave"),
            use_dino=cfg.get("use_dino", True),
            use_lora=do_lora,
            lora_rank=int(cfg.get("lora_rank", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_target_modules=cfg.get("lora_target_modules"),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            first_token_loss_weight=first_token_w,
        )
        return model

    if backbone == "tinyllava":
        from .tinyllava_ground_vlm import TinyLLaVAGroundVLM
        tinyllava_weights = cfg.get("tinyllava_weights")
        if cfg.get("load_tinyllava_weights", True) is False:
            tinyllava_weights = None
        model = TinyLLaVAGroundVLM(
            siglip_model=cfg.get("siglip_model", "google/siglip-so400m-patch14-384"),
            tinyllama_model=cfg.get(
                "tinyllama_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ),
            tinyllava_weights=tinyllava_weights,
            use_dino=cfg.get("use_dino", True),
            dinov2_model=cfg.get("dinov2_model", "facebook/dinov2-large"),
            mof_strategy=cfg.get("mof_strategy", "interleave"),
            vision_select_layer=int(cfg.get("vision_select_layer", -2)),
            connector_hidden_dim=int(cfg.get("connector_hidden_dim", 2048)),
            use_lora=do_lora,
            lora_rank=int(cfg.get("lora_rank", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_target_modules=cfg.get("lora_target_modules"),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            first_token_loss_weight=first_token_w,
            train_connector=bool(cfg.get("train_connector", False)),
        )
        return model

    raise ValueError(
        f"Unknown backbone '{backbone}'. Expected 'paligemma' or 'tinyllava'."
    )


def model_hidden_size(model: Any) -> int:
    """Return the LLM hidden size for both backbones (used by LoRA-MoE)."""
    if hasattr(model, "hidden_size"):
        return int(model.hidden_size)
    # FaceGroundVLM exposes this via paligemma.config.text_config.
    if hasattr(model, "paligemma"):
        return int(model.paligemma.config.text_config.hidden_size)
    raise AttributeError("Cannot determine hidden size for the given model.")


def model_trainable_parameters(model: Any) -> list:
    """Return the list of parameters that should be optimized in stage 1.

    Stage 1 trains only the bridging adapters: DINOv2 adapter always, and
    the SigLIP->LLM MLP connector only when TinyLLaVA is used AND it was
    NOT loaded from the pre-trained TinyLLaVA checkpoint.
    """
    params: list = []
    if getattr(model, "dino_adapter", None) is not None:
        params.extend(p for p in model.dino_adapter.parameters() if p.requires_grad)
    return params
