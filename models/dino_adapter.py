"""Single-layer MLP adapter that projects DINOv2 features into PaliGemma2's
Gemma2 embedding space.

Architecture follows TruthLens (Kundu et al., 2025): a single linear layer
maps DINOv2-Large hidden dim (1024) to Gemma2 hidden dim (2304).
"""
from __future__ import annotations

import torch.nn as nn


class DINOv2Adapter(nn.Module):
    """Linear projection from DINOv2 feature space to Gemma2 input space."""

    def __init__(self, dino_dim: int = 1024, gemma_dim: int = 2304):
        super().__init__()
        self.proj = nn.Linear(dino_dim, gemma_dim)

    def forward(self, dino_features):
        """Map DINOv2 patch tokens to Gemma2 space.

        Parameters
        ----------
        dino_features : Tensor [B, N, dino_dim]
            Patch tokens from DINOv2 (CLS excluded).

        Returns
        -------
        Tensor [B, N, gemma_dim]
        """
        return self.proj(dino_features)
