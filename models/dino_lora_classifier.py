"""DINOv2 with LoRA fine-tuning and dual classification heads.

Supports two modes:
  - **LoRA single**: one LoRA adapter applied to DINOv2 q/v projections.
  - **LoRA-MoE**: N expert LoRA adapters blended by a learned router.

Both modes use dual supervision:
  - Binary head (CLS token -> Real / Fake)
  - Forgery-type head (CLS token -> Original / Deepfakes / Face2Face / FaceShifter / FaceSwap / NeuralTextures)

The CLS token is used for classification while the patch tokens are
available downstream for the VLM adapter (I-MoF pipeline).

Usage::

    # Single LoRA
    model = DINOv2LoRAClassifier(use_moe=False)

    # LoRA-MoE with 6 experts
    model = DINOv2LoRAClassifier(use_moe=True, num_experts=6)
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import Dinov2Model

METHOD_TO_IDX: dict[str, int] = {
    "Original": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceShifter": 3,
    "FaceSwap": 4,
    "NeuralTextures": 5,
}
NUM_METHODS = len(METHOD_TO_IDX)


def _build_head(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class LoRAMoERouter(nn.Module):
    """Router that maps the CLS token to soft expert weights."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Return soft expert weights from CLS token [B, D] -> [B, E]."""
        return F.softmax(self.net(cls_token), dim=-1)


class DINOv2LoRAClassifier(nn.Module):
    """DINOv2 with LoRA (single or MoE) and dual classification heads.

    The backbone is kept frozen; only LoRA parameters and heads are trained.
    For MoE mode, a router selects/blends expert LoRA weights per sample.
    """

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-large",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        use_moe: bool = False,
        num_experts: int = 6,
        router_hidden_dim: int = 256,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.num_experts = num_experts

        if lora_target_modules is None:
            lora_target_modules = ["query", "value"]

        self.dinov2 = Dinov2Model.from_pretrained(dino_model)
        for p in self.dinov2.parameters():
            p.requires_grad = False

        dino_dim = self.dinov2.config.hidden_size

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        self.dinov2 = get_peft_model(self.dinov2, lora_config)
        self.dinov2.print_trainable_parameters()

        if use_moe:
            base_lora_sd = get_peft_model_state_dict(self.dinov2)
            self.expert_lora_params = nn.ParameterList()
            for _ in range(num_experts):
                expert = nn.ParameterDict()
                for k, v in base_lora_sd.items():
                    safe_key = k.replace(".", "__")
                    expert[safe_key] = nn.Parameter(v.clone())
                self.expert_lora_params.append(expert)

            self._param_keys = [k.replace(".", "__") for k in base_lora_sd.keys()]
            self._orig_keys = list(base_lora_sd.keys())

            self.router = LoRAMoERouter(dino_dim, num_experts, router_hidden_dim)

        self.binary_head = _build_head(dino_dim, head_hidden_dim, 2, head_dropout)
        self.forgery_head = _build_head(dino_dim, head_hidden_dim, NUM_METHODS, head_dropout)

    def _get_frozen_cls(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get CLS token from a forward pass with LoRA disabled (for routing)."""
        self.dinov2.disable_adapter_layers()
        with torch.no_grad():
            out = self.dinov2(pixel_values=pixel_values)
            cls_frozen = out.last_hidden_state[:, 0, :].detach()
        self.dinov2.enable_adapter_layers()
        return cls_frozen

    def _blend_lora(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Blend expert LoRA parameters: [B, E] averaged over batch."""
        avg_w = weights.mean(dim=0)
        blended = {}
        for safe_k, orig_k in zip(self._param_keys, self._orig_keys):
            blended[orig_k] = sum(
                avg_w[e] * self.expert_lora_params[e][safe_k]
                for e in range(self.num_experts)
            )
        return blended

    def forward(
        self,
        pixel_values: torch.Tensor,
        binary_labels: torch.Tensor | None = None,
        method_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Returns dict with keys: binary_logits, forgery_logits, cls_token,
        patch_tokens, and optionally loss, binary_loss, forgery_loss,
        router_loss, router_weights.
        """
        router_weights = None
        router_loss = torch.tensor(0.0, device=pixel_values.device)

        if self.use_moe:
            cls_frozen = self._get_frozen_cls(pixel_values)
            router_weights = self.router(cls_frozen)

            blended_sd = self._blend_lora(router_weights)
            set_peft_model_state_dict(self.dinov2, blended_sd)

            if method_labels is not None:
                valid_mask = method_labels >= 0
                if valid_mask.any():
                    router_loss = F.cross_entropy(
                        router_weights[valid_mask], method_labels[valid_mask]
                    )

        out = self.dinov2(pixel_values=pixel_values)
        hidden = out.last_hidden_state
        cls_token = hidden[:, 0, :]
        patch_tokens = hidden[:, 1:, :]

        cls_float = cls_token.to(self.binary_head[0].weight.dtype)
        binary_logits = self.binary_head(cls_float)
        forgery_logits = self.forgery_head(cls_float)

        result = {
            "binary_logits": binary_logits,
            "forgery_logits": forgery_logits,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }

        if router_weights is not None:
            result["router_weights"] = router_weights
            result["router_loss"] = router_loss

        if binary_labels is not None:
            binary_loss = F.cross_entropy(binary_logits, binary_labels)
            result["binary_loss"] = binary_loss

        if method_labels is not None:
            valid_mask = method_labels >= 0
            if valid_mask.any():
                forgery_loss = F.cross_entropy(
                    forgery_logits[valid_mask], method_labels[valid_mask]
                )
            else:
                forgery_loss = torch.tensor(0.0, device=pixel_values.device)
            result["forgery_loss"] = forgery_loss

        if binary_labels is not None and method_labels is not None:
            result["loss"] = result["binary_loss"] + result.get("forgery_loss", 0.0)
            if self.use_moe:
                result["loss"] = result["loss"] + result["router_loss"]

        return result

    def predict_verdict(self, pixel_values: torch.Tensor) -> list[str]:
        """Return verdict strings for evaluation."""
        out = self.forward(pixel_values)
        preds = out["binary_logits"].argmax(dim=-1)
        return ["Real" if p == 0 else "Fake" for p in preds.tolist()]

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the current LoRA state dict (for saving)."""
        return get_peft_model_state_dict(self.dinov2)
