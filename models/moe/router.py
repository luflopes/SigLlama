"""Gating router for the Mixture-of-Experts layer.

The router takes hidden states and produces gating weights that determine
which experts to activate. Supports top-k expert selection and load-
balancing auxiliary loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Top-k gating router.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    num_experts : int
        Total number of expert modules.
    top_k : int
        Number of experts activated per token.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : Tensor [B, seq_len, hidden_dim]

        Returns
        -------
        gate_weights : Tensor [B, seq_len, top_k]
            Normalised weights for the selected experts.
        expert_indices : Tensor [B, seq_len, top_k]
            Indices of the selected experts.
        aux_loss : Tensor scalar
            Load-balancing auxiliary loss.
        """
        logits = self.gate(hidden_states)  # [B, seq_len, num_experts]
        probs = F.softmax(logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        aux_loss = self._load_balance_loss(probs, top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _load_balance_loss(
        self, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load-balancing loss (Switch Transformer style)."""
        num_tokens = probs.shape[0] * probs.shape[1]
        one_hot = F.one_hot(indices, self.num_experts).float().sum(dim=2)
        tokens_per_expert = one_hot.sum(dim=(0, 1)) / num_tokens
        avg_prob_per_expert = probs.mean(dim=(0, 1))
        return (tokens_per_expert * avg_prob_per_expert).sum() * self.num_experts
