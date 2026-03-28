"""SigLlama: multimodal deepfake detection model.

Assembles SigLIP encoder + Visual Adapter + Soft Token Embedder + TinyLlama
decoder + MoE layer into a single end-to-end model.

Architecture overview:
    Image  -->  SigLIP  -->  Adapter  -->  visual embeddings  ─┐
    Image  -->  YOLO + FaceMesh  -->  SoftTokenEmbedder  ──────┤
    Text   -->  LlamaTokenizer  -->  text embeddings  ─────────┤
                                                               v
                                                    [concat along seq dim]
                                                               |
                                                          TinyLlama
                                                               |
                                                          MoE Layer
                                                               |
                                                          LM Head  -->  output text
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .siglip_encoder import SigLIPEncoder
from .adapter import VisualAdapter
from .soft_token_embedder import SoftTokenEmbedder
from .moe import MoELayer


@dataclass
class SigLlamaConfig:
    """Configuration for the SigLlama model."""

    siglip_model: str = "google/siglip-base-patch16-224"
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    freeze_siglip: bool = True
    freeze_llm: bool = False

    # Adapter
    visual_dim: int = 768       # SigLIP hidden size
    llm_dim: int = 2048         # TinyLlama hidden size

    # Soft tokens
    max_detections: int = 20
    num_landmarks: int = 468

    # MoE
    num_experts: int = 4
    moe_top_k: int = 2
    expert_type: str = "lora"   # "mlp" or "lora"
    lora_rank: int = 16
    moe_aux_weight: float = 0.01


class SigLlama(nn.Module):
    """Full SigLlama multimodal model.

    Stages:
        1. Pre-training: freeze SigLIP + LLM, train Adapter only (LCS-558K)
        2. Fine-tuning:  unfreeze LLM (or apply LoRA), train with MoE
    """

    def __init__(self, config: SigLlamaConfig):
        super().__init__()
        self.config = config

        # Visual encoder
        self.siglip = SigLIPEncoder(
            model_name=config.siglip_model,
            freeze=config.freeze_siglip,
        )

        # Visual -> LLM projection
        self.adapter = VisualAdapter(
            visual_dim=config.visual_dim,
            llm_dim=config.llm_dim,
        )

        # Soft token projection
        self.soft_token_embedder = SoftTokenEmbedder(
            llm_dim=config.llm_dim,
            max_detections=config.max_detections,
            num_landmarks=config.num_landmarks,
        )

        # Language model (decoder)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)

        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # MoE layer (applied to LLM hidden states before the final LM head)
        self.moe = MoELayer(
            hidden_dim=config.llm_dim,
            num_experts=config.num_experts,
            top_k=config.moe_top_k,
            expert_type=config.expert_type,
            lora_rank=config.lora_rank,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        det_features: Optional[torch.Tensor] = None,
        landmark_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass stub.

        This will be fully implemented during the training phase.
        The general flow is:

        1. Encode image through SigLIP -> adapter -> visual_embeds
        2. Encode soft tokens (dets + landmarks) -> soft_embeds
        3. Get text embeddings from LLM embedding layer
        4. Concatenate [soft_embeds, visual_embeds, text_embeds] along seq dim
        5. Pass through LLM decoder layers
        6. Apply MoE on hidden states
        7. Project to vocabulary -> compute loss
        """
        raise NotImplementedError(
            "Full forward pass will be implemented in Stage 1/2 training scripts."
        )
