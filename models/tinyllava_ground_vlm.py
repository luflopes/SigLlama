"""TinyLLaVAGroundVLM: alternative backbone to FaceGroundVLM.

Based on TinyLLaVA-share-Sig-TL (Zhou et al., 2024): SigLIP-So400m/14 +
TinyLlama-1.1B joined by a 2-layer MLP (``mlp2x_gelu``) connector. All three
components are loaded from the fine-tuned ``bczhou/TinyLLaVA-1.5B``
checkpoint to reuse the multimodal alignment obtained via the share recipe.

Architectural choices::

    Image ──> SigLIP-So400m/14 (fine-tuned) ──> MLP connector ──> [B, 729, 2048]
                                                                        │
    Image ──> DINOv2-Large  (optional, frozen) ──> DINOv2 Adapter ──> [B, 729, 2048]
                                                                        │
                                                                I-MoF interleave
                                                                        │
                                                            visual_embeds
                                                                        │
    Question ──> tokenizer ──> embed_tokens ──> text_embeds
                                                                        │
                                                    TinyLlama-1.1B (LoRA)
                                                                        │
                                                                    LM Head

Notes:
- SigLIP is read at select_layer=-2 (penultimate), select_feature='patch'
  to match the original TinyLLaVA configuration.
- DINOv2 is optional via ``use_dino``. When disabled, only SigLIP tokens are
  used (729 tokens).
- The full model exposes the same public API as ``FaceGroundVLM`` so that
  training/eval scripts can treat both interchangeably.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Dinov2Model,
    SiglipVisionModel,
)

from .dino_adapter import DINOv2Adapter
from .mixture_of_features import MOF_STRATEGIES

logger = logging.getLogger(__name__)


def _load_tinyllava_weights(path: str) -> dict[str, Any]:
    """Load the 3-component state_dict produced by ``extract_tinyllava_weights.py``."""
    ckpt = torch.load(path, map_location="cpu")
    for key in ("siglip", "connector", "llm"):
        if key not in ckpt:
            raise KeyError(
                f"Missing '{key}' in {path}. "
                "Regenerate via scripts/extract_tinyllava_weights.py."
            )
    return ckpt


class TinyLLaVAGroundVLM(nn.Module):
    """SigLIP + MLP connector + TinyLlama (+ optional DINOv2 I-MoF).

    Parameters
    ----------
    siglip_model : str
        Base SigLIP architecture (e.g. ``google/siglip-so400m-patch14-384``).
    tinyllama_model : str
        Base TinyLlama architecture (e.g. ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``).
    tinyllava_weights : str | None
        Path to ``outputs/tinyllava_weights.pt`` (from extraction script).
        When provided, overrides the base weights with the fine-tuned ones.
    use_dino : bool
        Enable DINOv2 + I-MoF branch. When False, only SigLIP features are
        used and the MLP connector alone bridges vision -> LLM.
    dinov2_model : str
        DINOv2 repo id (used only when ``use_dino=True``).
    mof_strategy : str
        ``"interleave"`` or ``"concatenate"`` (used only when ``use_dino=True``).
    vision_select_layer : int
        Which SigLIP hidden state to read (-2 = penultimate, matches TinyLLaVA).
    connector_hidden_dim : int
        Hidden size of the 2-layer MLP (default 2048 = TinyLlama hidden size).
    use_lora, lora_rank, lora_alpha, lora_target_modules, lora_dropout :
        Same semantics as ``FaceGroundVLM``.
    """

    def __init__(
        self,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        tinyllama_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tinyllava_weights: str | None = None,
        use_dino: bool = True,
        dinov2_model: str = "facebook/dinov2-large",
        mof_strategy: str = "interleave",
        vision_select_layer: int = -2,
        connector_hidden_dim: int = 2048,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.use_dino = use_dino
        self.mof_fn = MOF_STRATEGIES[mof_strategy] if use_dino else None
        self.vision_select_layer = vision_select_layer

        # --- SigLIP vision tower (will be overridden by fine-tuned weights) ---
        self.siglip = SiglipVisionModel.from_pretrained(
            siglip_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.siglip.requires_grad_(False)
        siglip_hidden = self.siglip.config.hidden_size  # 1152 for So400m

        # --- TinyLlama LLM (will be overridden by fine-tuned weights) ---
        llm_config = AutoConfig.from_pretrained(tinyllama_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            tinyllama_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.llm.requires_grad_(False)
        llm_hidden = self.llm.config.hidden_size  # 2048 for TinyLlama-1.1B

        # --- MLP connector (mlp2x_gelu): Linear-GELU-Linear ---
        # Indices 0 and 2 in the Sequential match the checkpoint keys
        # 'mm_projector.0.*' and 'mm_projector.2.*'.
        self.connector = nn.Sequential(
            nn.Linear(siglip_hidden, connector_hidden_dim),
            nn.GELU(),
            nn.Linear(connector_hidden_dim, llm_hidden),
        ).to(torch.bfloat16)
        self.connector.requires_grad_(False)

        # --- Optional DINOv2 branch ---
        if self.use_dino:
            self.dinov2 = Dinov2Model.from_pretrained(
                dinov2_model,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.dinov2.requires_grad_(False)
            dino_dim = self.dinov2.config.hidden_size
            self.dino_adapter = DINOv2Adapter(
                dino_dim=dino_dim, gemma_dim=llm_hidden,
            )
        else:
            self.dinov2 = None
            self.dino_adapter = None

        if tinyllava_weights is not None:
            self._load_tinyllava_checkpoint(tinyllava_weights)

        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_target_modules, lora_dropout)

        self._embed_tokens_ref = self._resolve_embed_tokens()

        logger.info(
            "TinyLLaVAGroundVLM built (use_dino=%s, use_lora=%s, siglip=%d, llm=%d)",
            use_dino, use_lora, siglip_hidden, llm_hidden,
        )

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def _load_tinyllava_checkpoint(self, path: str) -> None:
        ckpt = _load_tinyllava_weights(path)

        # SigLIP: strict=False because the TinyLLaVA share recipe only
        # updates layers >=12 -- older layers in the base repo are also
        # fine but the checkpoint may miss some keys (head, etc.).
        missing_s, unexpected_s = self.siglip.load_state_dict(
            ckpt["siglip"], strict=False,
        )
        logger.info(
            "SigLIP loaded from TinyLLaVA (missing=%d, unexpected=%d)",
            len(missing_s), len(unexpected_s),
        )

        missing_c, unexpected_c = self.connector.load_state_dict(
            ckpt["connector"], strict=True,
        )
        logger.info(
            "Connector loaded from TinyLLaVA (missing=%d, unexpected=%d)",
            len(missing_c), len(unexpected_c),
        )

        missing_l, unexpected_l = self.llm.load_state_dict(
            ckpt["llm"], strict=False,
        )
        logger.info(
            "LLM loaded from TinyLLaVA (missing=%d, unexpected=%d)",
            len(missing_l), len(unexpected_l),
        )

    def _apply_lora(self, rank, alpha, target_modules, dropout) -> None:
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
        self.llm = get_peft_model(self.llm, config)

    # ------------------------------------------------------------------
    # Public-facing accessors (mirror FaceGroundVLM API)
    # ------------------------------------------------------------------

    @property
    def language_model(self):
        return self.llm

    def _resolve_embed_tokens(self) -> nn.Embedding:
        base = self.llm.base_model.model if self.use_lora else self.llm
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens
        raise AttributeError(
            f"Cannot find embed_tokens on {type(base).__name__}"
        )

    def _get_embed_tokens(self) -> nn.Embedding:
        return self._embed_tokens_ref

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def enable_input_require_grads(self) -> None:
        self.llm.enable_input_require_grads()

    # ------------------------------------------------------------------
    # Vision encoding
    # ------------------------------------------------------------------

    def _encode_siglip(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run SigLIP and project via the MLP connector.

        Returns ``[B, N, llm_hidden]`` with CLS-like tokens excluded
        (SigLIP-So400m/14 at 384 yields 27x27 = 729 patch tokens).
        """
        with torch.no_grad():
            siglip_out = self.siglip(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            hidden = siglip_out.hidden_states[self.vision_select_layer]
            # SigLIP uses no explicit CLS; 'patch' select_feature means
            # we keep every token. Following TinyLLaVA configuration.
            siglip_feats = hidden
        return self.connector(siglip_feats.to(self.connector[0].weight.dtype))

    def _encode_dino(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run DINOv2, drop CLS, project via DINOv2 adapter."""
        assert self.dinov2 is not None and self.dino_adapter is not None
        with torch.no_grad():
            dino_out = self.dinov2(pixel_values=pixel_values).last_hidden_state
            dino_patches = dino_out[:, 1:, :]  # exclude CLS
        return self.dino_adapter(
            dino_patches.to(self.dino_adapter.proj.weight.dtype)
        )

    def _encode_vision(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine visual streams according to ``use_dino`` and I-MoF strategy."""
        siglip_proj = self._encode_siglip(pixel_values_siglip)
        if not self.use_dino or pixel_values_dino is None:
            return siglip_proj

        dino_proj = self._encode_dino(pixel_values_dino)

        # Align token count if needed (SigLIP-384 -> 729, DINOv2-Large -> 256
        # for 224, scales differently for 448). Interpolate to SigLIP length.
        if dino_proj.shape[1] != siglip_proj.shape[1]:
            dino_proj = self._resize_tokens(dino_proj, siglip_proj.shape[1])

        return self.mof_fn(siglip_proj, dino_proj)

    @staticmethod
    def _resize_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        """Bilinearly interpolate token sequences assuming a square grid."""
        B, N, D = tokens.shape
        src_side = int(round(N ** 0.5))
        if src_side * src_side != N:
            # Non-square: fall back to linear interpolation on the sequence.
            t = tokens.transpose(1, 2)  # [B, D, N]
            t = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
            return t.transpose(1, 2).contiguous()

        tgt_side = int(round(target_len ** 0.5))
        if tgt_side * tgt_side != target_len:
            t = tokens.transpose(1, 2)
            t = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
            return t.transpose(1, 2).contiguous()

        grid = tokens.transpose(1, 2).reshape(B, D, src_side, src_side)
        grid = F.interpolate(
            grid.float(), size=(tgt_side, tgt_side),
            mode="bilinear", align_corners=False,
        ).to(tokens.dtype)
        return grid.flatten(2).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Forward & generate
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor | None,
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

        # Memory-optimized: compute lm_head only on text positions.
        # peft.PeftModel patches LoRA modules in-place on the base model,
        # so calling base.model directly still routes through LoRA.
        base = self.llm.base_model.model if self.use_lora else self.llm
        lm_head = base.lm_head
        base_model = base.model  # LlamaModel

        base_out = base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
        )
        hidden = (
            base_out.last_hidden_state
            if hasattr(base_out, "last_hidden_state")
            else base_out[0]
        )

        text_hidden = hidden[:, P - 1:, :]  # [B, T+1, D]
        del hidden, base_out

        text_logits = lm_head(text_hidden)
        del text_hidden

        shift_logits = text_logits[:, :-1, :].contiguous()
        del text_logits
        shift_labels = labels.contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        pixel_values_siglip: torch.Tensor,
        pixel_values_dino: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> torch.Tensor:
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

        output_ids = self.llm.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        if was_training:
            self.train()
        return output_ids

    # ------------------------------------------------------------------
    # Symmetry helpers with FaceGroundVLM
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """LLM hidden size, used by the LoRA-MoE router."""
        return self.llm.config.hidden_size

    def trainable_summary(self) -> str:
        components = []
        adapter_p = 0
        if self.dino_adapter is not None:
            adapter_p = sum(
                p.numel() for p in self.dino_adapter.parameters() if p.requires_grad
            )
            components.append(f"DINOAdapter: {adapter_p:,}")
        lora_p = 0
        if self.use_lora:
            lora_p = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
            components.append(f"LoRA: {lora_p:,}")
        total = sum(p.numel() for p in self.parameters())
        trainable = adapter_p + lora_p
        return (
            f"Trainable: {trainable:,} ({100*trainable/max(total,1):.2f}%) | "
            + " | ".join(components)
            + f" | Total: {total:,}"
        )
