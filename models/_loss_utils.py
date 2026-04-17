"""Shared loss helpers for backbones.

Isolated so both ``TinyLLaVAGroundVLM`` and ``FaceGroundVLM`` can share the
same weighted-CE implementation without copy/paste.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_first_token_ce(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    first_token_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy with extra weight on the *first valid token* per row.

    Rationale: in DD-VQA the answer opens with a verdict token
    (``Real.``/``Fake.``) that dilutes into a ~1% slice of the vanilla CE,
    so the classifier head is barely supervised. Re-weighting the first
    non-ignored position pushes gradient mass onto the classification
    decision without changing anything downstream.

    Parameters
    ----------
    shift_logits : ``[B, T, V]``
        Already-shifted logits (positions predicting ``shift_labels``).
    shift_labels : ``[B, T]``
        Labels with ``-100`` for ignored positions.
    first_token_loss_weight : float
        Multiplier applied to the first valid position of every row. When
        equal to ``1.0`` the function collapses to standard mean-CE.

    Returns
    -------
    Scalar loss tensor on the same device as ``shift_logits``.
    """
    if first_token_loss_weight == 1.0:
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    B, T = shift_labels.shape
    ce_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    ce = ce_flat.view(B, T)

    valid = shift_labels.ne(-100)
    weights = valid.to(ce.dtype)

    # argmax on a bool/int tensor returns the first True per row when any
    # True exists; rows with no valid tokens are ignored below.
    first_idx = valid.int().argmax(dim=-1)
    has_any = valid.any(dim=-1)
    if has_any.any():
        rows = torch.arange(B, device=weights.device)[has_any]
        cols = first_idx[has_any]
        weights[rows, cols] = float(first_token_loss_weight)

    denom = weights.sum().clamp(min=1.0)
    return (ce * weights).sum() / denom
