"""Mixture-of-Experts layer.

Combines the Router with multiple Expert modules. For each token the router
selects top-k experts and the final output is a weighted sum of their
outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .router import Router
from .experts import Expert, LoRAExpert


class MoELayer(nn.Module):
    """Domain-specialised MoE layer for deepfake detection.

    Parameters
    ----------
    hidden_dim : int
        Hidden size of the LLM.
    num_experts : int
        Number of expert modules (default 4: GAN, VAE, Diffusion, Transformers).
    top_k : int
        Experts activated per token.
    expert_type : str
        ``"mlp"`` for standard MLP experts, ``"lora"`` for low-rank experts.
    lora_rank : int
        Rank when using LoRA experts.
    """

    DOMAIN_LABELS = ["GAN", "VAE", "Diffusion", "Transformers"]

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = "mlp",
        lora_rank: int = 16,
    ):
        super().__init__()
        self.router = Router(hidden_dim, num_experts, top_k)

        if expert_type == "lora":
            self.experts = nn.ModuleList(
                [LoRAExpert(hidden_dim, rank=lora_rank) for _ in range(num_experts)]
            )
        else:
            self.experts = nn.ModuleList(
                [Expert(hidden_dim) for _ in range(num_experts)]
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : Tensor [B, seq_len, hidden_dim]

        Returns
        -------
        output : Tensor [B, seq_len, hidden_dim]
            Weighted combination of selected expert outputs.
        aux_loss : Tensor scalar
            Router load-balancing loss.
        """
        gate_weights, expert_indices, aux_loss = self.router(hidden_states)
        # gate_weights:   [B, seq_len, top_k]
        # expert_indices: [B, seq_len, top_k]

        B, S, _ = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        for k in range(self.router.top_k):
            idx = expert_indices[:, :, k]       # [B, S]
            weight = gate_weights[:, :, k]      # [B, S]

            for e_idx, expert in enumerate(self.experts):
                mask = (idx == e_idx)            # [B, S]
                if not mask.any():
                    continue
                expert_input = hidden_states[mask]
                expert_out = expert(expert_input)
                output[mask] += weight[mask].unsqueeze(-1) * expert_out

        return output, aux_loss
