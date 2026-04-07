"""Stage 1 model: visual-language alignment pre-training.

Composes SigLIP (frozen) + VisualAdapter (trainable) + TinyLlama (frozen).
No soft tokens, no MoE at this stage.

Forward pass:
    pixel_values  -->  SigLIP  -->  Adapter  -->  visual_embeds  ─┐
    input_ids     -->  LLM embed_tokens       -->  text_embeds   ─┤
                                                                  v
                                                    [cat along seq dim]
                                                          │
                                                     TinyLlama decoder
                                                          │
                                                       LM Head  -->  loss
                                                  (on text tokens only)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from .adapter import VisualAdapter


class SigLlamaForPretraining(nn.Module):
    """SigLIP (frozen) + Adapter (trainable) + TinyLlama (frozen).

    Only the adapter weights are updated.  Gradients flow through the
    frozen LLM layers (their params just don't accumulate ``.grad``),
    which is how the adapter receives its learning signal.
    """

    def __init__(
        self,
        siglip_model: str = "google/siglip-base-patch16-224",
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        visual_dim: int = 768,
        llm_dim: int = 2048,
    ):
        super().__init__()

        self.siglip = SiglipVisionModel.from_pretrained(siglip_model)
        self.siglip.requires_grad_(False)

        self.adapter = VisualAdapter(visual_dim=visual_dim, llm_dim=llm_dim)

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm.requires_grad_(False)

    @property
    def num_visual_tokens(self) -> int:
        cfg = self.siglip.config
        return (cfg.image_size // cfg.patch_size) ** 2

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        device = pixel_values.device

        with torch.no_grad():
            visual_features = self.siglip(pixel_values=pixel_values).last_hidden_state

        visual_embeds = self.adapter(visual_features)          # [B, P, D]
        P = visual_embeds.shape[1]

        text_embeds = self.llm.model.embed_tokens(input_ids)   # [B, T, D]

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attn = torch.ones(B, P, dtype=attention_mask.dtype, device=device)
        full_attn = torch.cat([visual_attn, attention_mask], dim=1)

        visual_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([visual_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    # ------------------------------------------------------------------
    # Greedy generation (for validation samples)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer: AutoTokenizer,
        prompt: str | None = None,
        max_new_tokens: int = 64,
    ) -> list[str]:
        was_training = self.training
        self.eval()

        # Match the dtype of the LLM weights (needed when training with mixed precision)
        llm_dtype = self.llm.model.embed_tokens.weight.dtype

        visual_features = self.siglip(pixel_values=pixel_values).last_hidden_state
        visual_embeds = self.adapter(visual_features).to(llm_dtype)

        B = pixel_values.shape[0]
        device = pixel_values.device

        if prompt:
            prompt_enc = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
            prompt_ids = prompt_enc["input_ids"].to(device)          # [1, L]
            prompt_ids = prompt_ids.expand(B, -1)                    # [B, L]
            cur_ids = prompt_ids
        else:
            cur_ids = torch.full(
                (B, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
            )

        for _ in range(max_new_tokens):
            text_embeds = self.llm.model.embed_tokens(cur_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            logits = self.llm(inputs_embeds=inputs_embeds, use_cache=False).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)

            if (next_token == tokenizer.eos_token_id).all():
                break

        texts = tokenizer.batch_decode(cur_ids[:, 1:], skip_special_tokens=True)

        if was_training:
            self.train()
        return texts
