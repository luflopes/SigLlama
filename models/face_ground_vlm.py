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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Dinov2Model,
    PaliGemmaForConditionalGeneration,
)

from .dino_adapter import DINOv2Adapter
from .mixture_of_features import MOF_STRATEGIES

logger = logging.getLogger(__name__)


def _resolve_paligemma_components(paligemma):
    """Return (vision_tower, mm_projector, lm_backbone, embed_tokens)
    handling both transformers v4.x and v5.x model layouts."""
    if hasattr(paligemma, "vision_tower"):
        vt = paligemma.vision_tower
        proj = paligemma.multi_modal_projector
        lm = paligemma.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
            et = lm.model.embed_tokens
        else:
            et = lm.embed_tokens
    elif hasattr(paligemma, "model"):
        vt = paligemma.model.vision_tower
        proj = paligemma.model.multi_modal_projector
        lm = paligemma.model.language_model
        et = lm.embed_tokens
    else:
        raise AttributeError(
            "Cannot resolve PaliGemma components. "
            f"Top-level attrs: {list(paligemma._modules.keys())}"
        )
    return vt, proj, lm, et


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
            attn_implementation="sdpa",
        )
        self.paligemma.requires_grad_(False)

        # --- Resolve internal component references (v4/v5 compat) ---
        vt, proj, lm, _ = _resolve_paligemma_components(self.paligemma)
        self._vision_tower = vt
        self._mm_projector = proj
        self._lm_ref_attr = self._detect_lm_attr()

        logger.info(
            "PaliGemma components resolved: vision_tower=%s, lm=%s",
            type(vt).__name__, type(lm).__name__,
        )

        # --- DINOv2 (frozen) ---
        self.dinov2 = Dinov2Model.from_pretrained(
            dinov2_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.dinov2.requires_grad_(False)

        # --- DINOv2 Adapter (trainable) ---
        dino_dim = self.dinov2.config.hidden_size
        gemma_dim = self.paligemma.config.text_config.hidden_size
        self.dino_adapter = DINOv2Adapter(dino_dim=dino_dim, gemma_dim=gemma_dim)

        # --- LoRA on Gemma2 ---
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_target_modules, lora_dropout)

        # Text-only logits: apply lm_head only on text positions to
        # avoid materializing [B, full_seq, vocab] (saves ~17× memory).
        self._has_separate_lm_head = (
            hasattr(self.paligemma, "model")
            and hasattr(self.paligemma, "lm_head")
        )
        if self._has_separate_lm_head:
            logger.info(
                "Text-only logits optimization enabled "
                "(lm_head computed only on text positions)"
            )

    def _detect_lm_attr(self) -> str:
        """Return the attribute path to the language model for LoRA."""
        if hasattr(self.paligemma, "language_model"):
            return "language_model"
        return "model.language_model"

    def _get_lm(self):
        """Return the language model sub-module (potentially LoRA-wrapped)."""
        if self._lm_ref_attr == "language_model":
            return self.paligemma.language_model
        return self.paligemma.model.language_model

    def _set_lm(self, new_lm):
        """Replace the language model sub-module (for LoRA wrapping)."""
        if self._lm_ref_attr == "language_model":
            self.paligemma.language_model = new_lm
        else:
            self.paligemma.model.language_model = new_lm

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
        lm = self._get_lm()
        self._set_lm(get_peft_model(lm, config))

    @property
    def language_model(self):
        return self._get_lm()

    def _get_embed_tokens(self) -> nn.Embedding:
        lm = self._get_lm()
        if self.use_lora:
            base = lm.get_base_model()
        else:
            base = lm
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens
        raise AttributeError(f"Cannot find embed_tokens on {type(base).__name__}")

    def enable_gradient_checkpointing(self):
        self._get_lm().gradient_checkpointing_enable()

    def enable_input_require_grads(self):
        self._get_lm().enable_input_require_grads()

    def _encode_vision(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image through both vision towers and mix features."""
        with torch.no_grad():
            siglip_out = self._vision_tower(
                pixel_values=pixel_values_siglip
            ).last_hidden_state
            siglip_proj = self._mm_projector(siglip_out)

            dino_out = self.dinov2(
                pixel_values=pixel_values_dino
            ).last_hidden_state
            dino_patches = dino_out[:, 1:, :]  # exclude CLS

        dino_adapted = self.dino_adapter(dino_patches.to(self.dino_adapter.proj.weight.dtype))

        return self.mof_fn(siglip_proj, dino_adapted)

    def _forward_text_logits(
        self,
        inputs_embeds: torch.Tensor,
        full_attn: torch.Tensor,
        full_ttids: torch.Tensor,
        num_visual: int,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Memory-optimized forward: lm_head only on text positions.

        Instead of computing logits for the entire sequence
        ``[B, 2048+T, vocab]`` and then casting to float32 for the loss
        (which allocates ~17× more memory than needed), this method:

        1. Runs the base PaliGemmaModel to get hidden states.
        2. Selects only ``[last_visual_pos : end]`` hidden states.
        3. Applies ``lm_head`` only on those positions.

        The last visual position is included because its logit predicts
        the first text token in the standard shifted causal-LM loss.
        """
        base_out = self.paligemma.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            token_type_ids=full_ttids,
        )
        hidden = (
            base_out.last_hidden_state
            if hasattr(base_out, "last_hidden_state")
            else base_out[0]
        )

        text_hidden = hidden[:, num_visual - 1 :, :]  # [B, T+1, D]
        del hidden, base_out

        text_logits = self.paligemma.lm_head(text_hidden)  # [B, T+1, vocab]
        del text_hidden

        shift_logits = text_logits[:, :-1, :].contiguous()  # [B, T, vocab]
        del text_logits
        shift_labels = labels.contiguous()  # [B, T]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return {"loss": loss}

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

        # token_type_ids: 0=visual (bidirectional attn), 1=text (causal attn)
        visual_ttids = torch.zeros(B, P, dtype=torch.long, device=device)
        text_ttids = torch.ones(B, input_ids.shape[1], dtype=torch.long, device=device)
        full_ttids = torch.cat([visual_ttids, text_ttids], dim=1)

        if self._has_separate_lm_head:
            return self._forward_text_logits(
                inputs_embeds, full_attn, full_ttids, P, labels,
            )

        # Fallback: full forward (transformers v4.x or non-standard layout)
        visual_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([visual_labels, labels], dim=1)

        outputs = self.paligemma(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
            token_type_ids=full_ttids,
        )

        return {"loss": outputs.loss}

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

        output_ids = self.paligemma.generate(
            input_ids=None,
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
                for p in self._get_lm().parameters()
                if p.requires_grad
            )
        total = sum(p.numel() for p in self.parameters())
        trainable = adapter_p + lora_p
        return (
            f"Trainable: {trainable:,} ({100*trainable/total:.2f}%) | "
            f"Adapter: {adapter_p:,} | LoRA: {lora_p:,} | Total: {total:,}"
        )
