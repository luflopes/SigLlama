"""FaceGroundVLM: PaliGemma2 + DINOv2 with Mixture-of-Features for
explainable deepfake detection with facial region grounding.

Architecture (following TruthLens, Kundu et al. 2025)::

    Image ──> SigLIP-So400m/14 ──> PaliGemma Projector ──> siglip_proj [B,1024,2304]
    Image ──> DINOv2-Large     ──> DINOv2 Adapter       ──> dino_proj  [B,1024,2304]
                                                                 │
                                                        I-MoF interleave
                                                                 │
                                                     visual_embeds [B,2048,2304]
                                                                 │
    Question ──> tokenizer ──> embed_tokens ──> text_embeds [B,T,2304]
                                                                 │
                                             ┌───────────────────┘
                                             v
                                  [visual_embeds ∥ text_embeds]
                                             │
                                        Gemma2-2B (LoRA)
                                             │
                                          LM Head ──> loss / generation

Novel contributions over TruthLens:
- <loc> token grounding for facial regions (Phase 3-4)
- LoRA-MoE per manipulation technique (Phase 5)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    Dinov2Model,
    PaliGemmaForConditionalGeneration,
)

from .dino_adapter import DINOv2Adapter
from .mixture_of_features import MOF_STRATEGIES


class FaceGroundVLM(nn.Module):
    """PaliGemma2 + DINOv2 + DINOv2Adapter + MoF.

    Parameters
    ----------
    paligemma_model : str
        HuggingFace model id for PaliGemma2 (e.g. google/paligemma2-3b-pt-448).
    dinov2_model : str
        HuggingFace model id for DINOv2 (e.g. facebook/dinov2-large).
    mof_strategy : str
        Feature mixing strategy: ``"interleave"`` (I-MoF) or ``"concatenate"`` (C-MoF).
    use_lora : bool
        Apply LoRA adapters to the Gemma2 LLM.
    lora_rank, lora_alpha : int
        LoRA hyper-parameters.
    lora_target_modules : list[str] | None
        Gemma2 modules to apply LoRA to.
    lora_dropout : float
        Dropout for LoRA layers.
    """

    def __init__(
        self,
        paligemma_model: str = "google/paligemma2-3b-pt-448",
        dinov2_model: str = "facebook/dinov2-large",
        mof_strategy: str = "interleave",
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.mof_fn = MOF_STRATEGIES[mof_strategy]

        # --- PaliGemma2 (SigLIP + Projector + Gemma2) ---
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            paligemma_model,
            torch_dtype=torch.bfloat16,
        )
        self.paligemma.requires_grad_(False)

        # --- DINOv2 (frozen) ---
        self.dinov2 = Dinov2Model.from_pretrained(
            dinov2_model,
            torch_dtype=torch.bfloat16,
        )
        self.dinov2.requires_grad_(False)

        # --- DINOv2 Adapter (trainable) ---
        dino_dim = self.dinov2.config.hidden_size
        gemma_dim = self.paligemma.config.text_config.hidden_size
        self.dino_adapter = DINOv2Adapter(dino_dim=dino_dim, gemma_dim=gemma_dim)

        # --- LoRA on Gemma2 ---
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_target_modules, lora_dropout)

    def _apply_lora(self, rank, alpha, target_modules, dropout):
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.paligemma.language_model = get_peft_model(
            self.paligemma.language_model, config
        )

    @property
    def language_model(self):
        return self.paligemma.language_model

    def _get_embed_tokens(self) -> nn.Embedding:
        lm = self.paligemma.language_model
        if self.use_lora:
            return lm.get_base_model().model.embed_tokens
        return lm.model.embed_tokens

    def _encode_vision(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image through both vision towers and mix features."""
        with torch.no_grad():
            siglip_out = self.paligemma.vision_tower(
                pixel_values=pixel_values_siglip
            ).last_hidden_state
            siglip_proj = self.paligemma.multi_modal_projector(siglip_out)

            dino_out = self.dinov2(
                pixel_values=pixel_values_dino
            ).last_hidden_state
            dino_patches = dino_out[:, 1:, :]  # exclude CLS

        dino_adapted = self.dino_adapter(dino_patches.to(self.dino_adapter.proj.weight.dtype))

        return self.mof_fn(siglip_proj, dino_adapted)

    def forward(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = pixel_values_siglip.shape[0]
        device = pixel_values_siglip.device

        visual_embeds = self._encode_vision(pixel_values_siglip, pixel_values_dino)
        P = visual_embeds.shape[1]

        text_embeds = self._get_embed_tokens()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attn = torch.ones(B, P, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)

        visual_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([visual_labels, labels], dim=1)

        outputs = self.paligemma.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Auto-regressive generation with visual context."""
        was_training = self.training
        self.eval()

        visual_embeds = self._encode_vision(pixel_values_siglip, pixel_values_dino)
        P = visual_embeds.shape[1]
        B = visual_embeds.shape[0]
        device = visual_embeds.device

        text_embeds = self._get_embed_tokens()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attn = torch.ones(B, P, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)

        output_ids = self.paligemma.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        if was_training:
            self.train()
        return output_ids

    def trainable_summary(self) -> str:
        adapter_p = sum(p.numel() for p in self.dino_adapter.parameters() if p.requires_grad)
        lora_p = 0
        if self.use_lora:
            lora_p = sum(
                p.numel()
                for p in self.paligemma.language_model.parameters()
                if p.requires_grad
            )
        total = sum(p.numel() for p in self.parameters())
        trainable = adapter_p + lora_p
        return (
            f"Trainable: {trainable:,} ({100*trainable/total:.2f}%) | "
            f"Adapter: {adapter_p:,} | LoRA: {lora_p:,} | Total: {total:,}"
        )
