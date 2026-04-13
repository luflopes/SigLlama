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



class CrossAttentionAdapter(nn.Module):
    """
    Compact cross-attention adapter (BLIP-2 style bottleneck).

    Pipeline:
    visual (768) → proj (512) → cross-attn (queries) → FFN → proj (2048)

    Output:
    [B, num_queries, llm_dim]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        hidden_dim: int = 512,
        llm_dim: int = 2048,
        num_queries: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 🔹 Queries aprendíveis
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )

        # 🔹 Projeção visual (768 → 512)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # 🔹 Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 🔹 Normalizações
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        # 🔹 Feed-forward (estilo transformer)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # 🔹 Projeção final para o LLM (512 → 2048)
        self.to_llm = nn.Linear(hidden_dim, llm_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        visual_features : Tensor [B, num_patches, visual_dim]

        Returns
        -------
        Tensor [B, num_queries, llm_dim]
        """

        B = visual_features.size(0)

        # 🔹 Projeta features visuais
        visual = self.visual_proj(visual_features)  # [B, N, 512]

        # 🔹 Expande queries
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, 512]

        # 🔹 Cross-attention
        q = self.norm_q(queries)
        attn_out, _ = self.cross_attn(q, visual, visual)

        # 🔹 Residual
        x = queries + attn_out

        # 🔹 FFN + residual
        x = x + self.ffn(self.norm_out(x))

        # 🔹 Projeta para espaço do LLM
        out = self.to_llm(x)  # [B, Q, 2048]

        return out

