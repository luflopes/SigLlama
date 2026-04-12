"""Stage 2a model: SigLIP (frozen) + Adapter (trainable) + TinyLlama (LoRA).

No soft tokens or MoE at this stage.  LoRA is applied to the LLM
so that it can learn deepfake-specific language while keeping most
parameters frozen.

Forward pass::

    pixel_values  -->  SigLIP  -->  Adapter  -->  visual_embeds  ──┐
    input_ids     -->  LLM embed_tokens       -->  text_embeds    ──┤
                                                                    v
                                                     [concat along seq dim]
                                                           │
                                                      TinyLlama (LoRA)
                                                           │
                                                        LM Head  -->  loss
                                                    (on answer tokens only)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

from .adapter import VisualAdapter


class SigLlamaForFinetune(nn.Module):
    """SigLIP (frozen) + Adapter (trainable) + TinyLlama (LoRA).

    Parameters
    ----------
    siglip_model : str
        HuggingFace SigLIP model identifier.
    llm_model : str
        HuggingFace TinyLlama model identifier.
    visual_dim : int
        SigLIP hidden size.
    llm_dim : int
        TinyLlama hidden size.
    adapter_checkpoint : str or None
        Path to Stage 1 adapter checkpoint to load.
    lora_rank : int
        LoRA rank for TinyLlama.
    lora_alpha : int
        LoRA alpha scaling factor.
    lora_target_modules : list[str]
        Modules to apply LoRA to.
    lora_dropout : float
        Dropout for LoRA layers.
    """

    def __init__(
        self,
        siglip_model: str = "google/siglip-base-patch16-224",
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        visual_dim: int = 768,
        llm_dim: int = 2048,
        adapter_checkpoint: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # --- SigLIP (frozen) ---
        self.siglip = SiglipVisionModel.from_pretrained(siglip_model)
        self.siglip.requires_grad_(False)

        # --- Adapter (trainable) ---
        self.adapter = VisualAdapter(visual_dim=visual_dim, llm_dim=llm_dim)
        if adapter_checkpoint:
            ckpt = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
            state = ckpt["adapter"] if isinstance(ckpt, dict) and "adapter" in ckpt else ckpt
            self.adapter.load_state_dict(state)

        # --- TinyLlama + LoRA ---
        base_llm = AutoModelForCausalLM.from_pretrained(llm_model)

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(base_llm, lora_config)

    @property
    def num_visual_tokens(self) -> int:
        cfg = self.siglip.config
        return (cfg.image_size // cfg.patch_size) ** 2

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

        visual_embeds = self.adapter(visual_features)
        P = visual_embeds.shape[1]

        text_embeds = self.llm.get_base_model().model.embed_tokens(input_ids)

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

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer: AutoTokenizer,
        prompt: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ) -> list[str]:
        was_training = self.training
        self.eval()

        llm_dtype = self.llm.get_base_model().model.embed_tokens.weight.dtype

        visual_features = self.siglip(pixel_values=pixel_values).last_hidden_state
        visual_embeds = self.adapter(visual_features).to(llm_dtype)

        B = pixel_values.shape[0]
        device = pixel_values.device

        if prompt:
            prompt_enc = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
            cur_ids = prompt_enc["input_ids"].to(device).expand(B, -1)
        else:
            cur_ids = torch.full(
                (B, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
            )

        for _ in range(max_new_tokens):
            text_embeds = self.llm.get_base_model().model.embed_tokens(cur_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            logits = self.llm(inputs_embeds=inputs_embeds, use_cache=False).logits
            next_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(B):
                    prev_tokens = cur_ids[b].tolist()
                    for tok_id in set(prev_tokens):
                        if next_logits[b, tok_id] > 0:
                            next_logits[b, tok_id] /= repetition_penalty
                        else:
                            next_logits[b, tok_id] *= repetition_penalty

            if do_sample and temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove_mask] = float("-inf")
                    next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            cur_ids = torch.cat([cur_ids, next_token], dim=1)

            if (next_token == tokenizer.eos_token_id).all():
                break

        texts = tokenizer.batch_decode(cur_ids[:, 1:], skip_special_tokens=True)

        if was_training:
            self.train()
        return texts

    def print_trainable_parameters(self) -> str:
        adapter_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel() for n, p in self.llm.named_parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.parameters())
        trainable = adapter_params + lora_params
        msg = (
            f"Trainable: {trainable:,} ({100 * trainable / total_params:.2f}%) | "
            f"Adapter: {adapter_params:,} | LoRA: {lora_params:,} | "
            f"Total: {total_params:,}"
        )
        return msg
