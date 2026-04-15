"""Stage 4: LoRA-MoE fine-tuning with per-technique experts.

Each LoRA expert is initialised from the Phase 3 LoRA checkpoint.
A learnable router selects expert weights based on visual features.
Loss = LM loss + lambda * router auxiliary loss.

Usage::

    python training/train_stage4.py --config configs/stage4_moe.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from data.ddvqa_dataset import DDVQADataset, collate_ddvqa
from models.face_ground_vlm import FaceGroundVLM
from models.lora_moe import LoRAMoERouter
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


# ── Technique label ↔ expert index mapping ──
TECHNIQUE_TO_IDX: dict[str, int] = {
    "Deepfakes": 0,
    "Face2Face": 1,
    "FaceSwap": 2,
    "NeuralTextures": 3,
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="FaceGroundVLM Stage 4 (LoRA-MoE)")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    return p.parse_args()


class FaceGroundVLM_MoE(nn.Module):
    """Wraps FaceGroundVLM with LoRA-MoE: multiple LoRA state dicts
    blended at inference via a router."""

    def __init__(
        self,
        base_model: FaceGroundVLM,
        num_experts: int,
        router_hidden_dim: int,
        base_lora_sd: dict,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_experts = num_experts

        gemma_dim = base_model.paligemma.config.text_config.hidden_size
        self.router = LoRAMoERouter(gemma_dim, num_experts, router_hidden_dim)

        self.expert_lora_params = nn.ParameterList()
        flat_base = {k: v.clone() for k, v in base_lora_sd.items()}
        for _ in range(num_experts):
            expert_params = nn.ParameterDict()
            for k, v in flat_base.items():
                safe_key = k.replace(".", "__")
                expert_params[safe_key] = nn.Parameter(v.clone())
            self.expert_lora_params.append(expert_params)

        self._param_keys = [k.replace(".", "__") for k in flat_base.keys()]
        self._orig_keys = list(flat_base.keys())

    def _blend_lora(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Blend expert LoRA parameters using router weights [B, E] -> averaged."""
        avg_weights = weights.mean(dim=0)
        blended = {}
        for i, (safe_k, orig_k) in enumerate(zip(self._param_keys, self._orig_keys)):
            blended[orig_k] = sum(
                avg_weights[e] * self.expert_lora_params[e][safe_k]
                for e in range(self.num_experts)
            )
        return blended

    def forward(
        self,
        pixel_values_siglip,
        pixel_values_dino,
        input_ids,
        attention_mask,
        labels,
        method: list[str] | None = None,
    ):
        visual_embeds = self.base_model._encode_vision(
            pixel_values_siglip, pixel_values_dino
        )
        router_weights = self.router(visual_embeds)

        blended_sd = self._blend_lora(router_weights)
        set_peft_model_state_dict(
            self.base_model.language_model, blended_sd
        )

        out = self.base_model(
            pixel_values_siglip=pixel_values_siglip,
            pixel_values_dino=pixel_values_dino,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        router_loss = torch.tensor(0.0, device=visual_embeds.device)
        if method is not None:
            target_indices = []
            for m in method:
                idx = TECHNIQUE_TO_IDX.get(m, -1)
                if idx >= 0:
                    target_indices.append(idx)
            if target_indices:
                gt = torch.tensor(target_indices, device=router_weights.device)
                valid_weights = router_weights[: len(target_indices)]
                router_loss = F.cross_entropy(valid_weights, gt)

        return {
            "loss": out["loss"],
            "router_loss": router_loss,
            "router_weights": router_weights,
        }


def build_dataloaders(cfg, processor, dino_transform):
    train_ds = DDVQADataset(
        metadata_path=cfg["train_metadata"],
        image_root=cfg["image_root"],
        processor=processor,
        dino_transform=dino_transform,
        max_length=cfg.get("max_text_length", 384),
    )
    val_ds = DDVQADataset(
        metadata_path=cfg["val_metadata"],
        image_root=cfg["image_root"],
        processor=processor,
        dino_transform=dino_transform,
        max_length=cfg.get("max_text_length", 384),
    )

    nw = cfg["num_workers"]
    dl_common = dict(
        num_workers=nw, collate_fn=collate_ddvqa, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
    )

    if cfg.get("balanced_sampling", False):
        n_real = sum(1 for s in train_ds.samples if s.get("label", "").lower() == "real")
        n_fake = len(train_ds.samples) - n_real
        wr = 1.0 / max(n_real, 1)
        wf = 1.0 / max(n_fake, 1)
        weights = [wr if s.get("label", "").lower() == "real" else wf for s in train_ds.samples]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler, **dl_common,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, **dl_common,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, **dl_common,
    )
    return train_loader, val_loader


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg.get("mixed_precision", "no"),
    )

    processor = AutoProcessor.from_pretrained(cfg["paligemma_model"])
    tokenizer = processor.tokenizer

    dino_transform = Compose([
        Resize((448, 448)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_model = FaceGroundVLM(
        paligemma_model=cfg["paligemma_model"],
        dinov2_model=cfg["dinov2_model"],
        mof_strategy=cfg.get("mof_strategy", "interleave"),
        use_lora=True,
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_target_modules=cfg.get("lora_target_modules"),
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )
    base_model.enable_input_require_grads()

    ckpt = torch.load(cfg["adapter_checkpoint"], map_location="cpu")
    base_model.dino_adapter.load_state_dict(ckpt["adapter"])
    if "lora" in ckpt:
        set_peft_model_state_dict(base_model.language_model, ckpt["lora"])
    base_lora_sd = get_peft_model_state_dict(base_model.language_model)

    if cfg.get("gradient_checkpointing", True):
        base_model.enable_gradient_checkpointing()

    model = FaceGroundVLM_MoE(
        base_model=base_model,
        num_experts=cfg["num_experts"],
        router_hidden_dim=cfg.get("router_hidden_dim", 512),
        base_lora_sd=base_lora_sd,
    )
    logger.info("Created MoE model with %d experts", cfg["num_experts"])

    train_loader, val_loader = build_dataloaders(cfg, processor, dino_transform)

    adapter_params = list(model.base_model.dino_adapter.parameters())
    router_params = list(model.router.parameters())
    expert_params = list(model.expert_lora_params.parameters())

    optimizer = AdamW([
        {"params": adapter_params, "lr": float(cfg.get("adapter_lr", 5e-5))},
        {"params": expert_params, "lr": float(cfg.get("lora_lr", 1e-5))},
        {"params": router_params, "lr": float(cfg.get("router_lr", 1e-4))},
    ])

    grad_accum = int(cfg["gradient_accumulation_steps"])
    max_epochs = int(cfg["max_epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    max_train_steps = steps_per_epoch * max_epochs
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_train_steps * warmup_ratio),
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler,
    )

    router_loss_weight = float(cfg.get("router_loss_weight", 0.1))
    log_interval = int(cfg.get("log_interval", 50))
    save_interval = int(cfg.get("save_interval", 500))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{max_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            if batch is None:
                continue

            methods = batch.pop("method", None)
            batch.pop("label_str", None)

            with accelerator.accumulate(model):
                out = model(**batch, method=methods)
                total_loss = out["loss"] + router_loss_weight * out["router_loss"]
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    all_params = []
                    for pg in optimizer.param_groups:
                        all_params.extend(pg["params"])
                    accelerator.clip_grad_norm_(all_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    li = total_loss.detach().float().item()
                    ri = out["router_loss"].detach().float().item()
                    epoch_loss += li
                    n_batches += 1

                    pbar.set_postfix(
                        loss=f"{li:.4f}",
                        rloss=f"{ri:.4f}",
                        step=global_step,
                    )

                    if global_step % log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            "epoch=%d step=%d loss=%.4f router_loss=%.4f",
                            epoch, global_step, li, ri,
                        )

                    if save_interval > 0 and global_step % save_interval == 0 and accelerator.is_main_process:
                        um = accelerator.unwrap_model(model)
                        torch.save({
                            "step": global_step,
                            "adapter": um.base_model.dino_adapter.state_dict(),
                            "expert_lora_params": um.expert_lora_params.state_dict(),
                            "router": um.router.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }, output_dir / f"checkpoint-{global_step}.pt")
                        logger.info("Saved checkpoint-%d", global_step)

        if accelerator.is_main_process and n_batches > 0:
            logger.info("epoch %d done | mean loss=%.4f", epoch, epoch_loss / n_batches)

    if accelerator.is_main_process:
        um = accelerator.unwrap_model(model)
        torch.save({
            "step": global_step,
            "adapter": um.base_model.dino_adapter.state_dict(),
            "expert_lora_params": um.expert_lora_params.state_dict(),
            "router": um.router.state_dict(),
        }, output_dir / "checkpoint-final.pt")
        logger.info("Saved final checkpoint")


if __name__ == "__main__":
    main()
