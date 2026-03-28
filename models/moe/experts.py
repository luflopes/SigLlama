"""Expert modules for the MoE layer.

Each expert is specialised for a deepfake generation domain (GAN, VAE,
Diffusion, Transformers). Experts can be either small MLPs or LoRA-style
low-rank matrices applied to the base LLM weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Expert(nn.Module):
    """Standard MLP expert.

    Parameters
    ----------
    hidden_dim : int
        Input and output dimension.
    intermediate_dim : int
        Hidden layer size (typically 4x hidden_dim).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int | None = None):
        super().__init__()
        intermediate_dim = intermediate_dim or hidden_dim * 4
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoRAExpert(nn.Module):
    """Low-rank expert (LoRA-style).

    Instead of a full MLP, applies a low-rank delta: x + B @ A @ x.

    Parameters
    ----------
    hidden_dim : int
        Input/output dimension.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    """

    def __init__(self, hidden_dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.lora_a = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.lora_b(self.lora_a(x))
