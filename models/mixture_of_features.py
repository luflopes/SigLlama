"""Mixture of Features (MoF) strategies for combining SigLIP and DINOv2
visual tokens.

Two strategies from TruthLens (Kundu et al., 2025):
- **I-MoF** (Interleave): alternate tokens from each encoder.
- **C-MoF** (Concatenate): stack all tokens sequentially.

I-MoF consistently outperforms C-MoF in the TruthLens ablation study.
"""
from __future__ import annotations

import torch


def interleave_mof(
    siglip_tokens: torch.Tensor,
    dino_tokens: torch.Tensor,
) -> torch.Tensor:
    """Interleave SigLIP and DINOv2 tokens along the sequence dimension.

    [s0, d0, s1, d1, ..., sN, dN]  ->  2N tokens total.
    """
    B, N, D = siglip_tokens.shape
    interleaved = torch.stack([siglip_tokens, dino_tokens], dim=2)
    return interleaved.reshape(B, 2 * N, D)


def concatenate_mof(
    siglip_tokens: torch.Tensor,
    dino_tokens: torch.Tensor,
) -> torch.Tensor:
    """Concatenate SigLIP and DINOv2 tokens along the sequence dimension.

    [s0, s1, ..., sN, d0, d1, ..., dN]  ->  2N tokens total.
    """
    return torch.cat([siglip_tokens, dino_tokens], dim=1)


MOF_STRATEGIES = {
    "interleave": interleave_mof,
    "concatenate": concatenate_mof,
}
