"""MLP Adapter that projects SigLIP visual features into TinyLlama's
embedding space.

Architecture: Linear -> GELU -> Linear (with optional layer norm).
This is the component trained during Stage 1 (pre-training on LCS-558K).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    """Two-layer MLP that maps visual features to the LLM hidden dimension.

    Parameters
    ----------
    visual_dim : int
        Input dimension (SigLIP hidden size, e.g. 768).
    llm_dim : int
        Output dimension (TinyLlama hidden size, e.g. 2048).
    use_layer_norm : bool
        Whether to apply LayerNorm before the projection.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        llm_dim: int = 2048,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(visual_dim) if use_layer_norm else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        visual_features : Tensor [B, num_patches, visual_dim]

        Returns
        -------
        Tensor [B, num_patches, llm_dim]
            Visual embeddings aligned with the LLM embedding space.
        """
        return self.proj(self.norm(visual_features))
