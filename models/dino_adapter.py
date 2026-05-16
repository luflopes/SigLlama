"""Single-layer MLP adapter that projects DINOv2 features into the LLM
embedding space.

Architecture follows TruthLens (Kundu et al., 2025): a single linear layer
maps DINOv2-Large hidden dim (1024) to the target LLM hidden dim (e.g. 2048
for TinyLlama).
"""
from __future__ import annotations

import torch.nn as nn


class DINOv2Adapter(nn.Module):
    """Linear projection from DINOv2 feature space to LLM input space."""

    def __init__(self, dino_dim: int = 1024, target_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(dino_dim, target_dim)

    def forward(self, dino_features):
        """Map DINOv2 patch tokens to LLM embedding space.

        Parameters
        ----------
        dino_features : Tensor [B, N, dino_dim]
            Patch tokens from DINOv2 (CLS excluded).

        Returns
        -------
        Tensor [B, N, target_dim]
        """
        return self.proj(dino_features)
