"""LoRA Mixture-of-Experts (LoRA-MoE) for domain-specific deepfake detection.

Each expert is a separate LoRA adapter specialised for a manipulation
technique (Deepfakes, Face2Face, FaceSwap, NeuralTextures).  A learned
router selects expert weights based on the pooled visual features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


class LoRAMoERouter(nn.Module):
    """Lightweight router that maps visual features to expert weights."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Return soft expert weights from mean-pooled visual features.

        Parameters
        ----------
        visual_features : Tensor [B, N, D]
            Visual embeddings (I-MoF output).

        Returns
        -------
        Tensor [B, num_experts]  (softmax weights)
        """
        pooled = visual_features.mean(dim=1)
        return F.softmax(self.net(pooled), dim=-1)
