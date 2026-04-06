"""Stage 1: Pre-train the Visual Adapter on LCS-558K.

Only the Adapter (MLP projection from SigLIP space to TinyLlama space)
is trained.  Both SigLIP and TinyLlama remain frozen.

Usage::

    # Single GPU
    python training/pretrain.py --config configs/pretraining.yaml

    # Multi-GPU via accelerate
    accelerate launch training/pretrain.py --config configs/pretraining.yaml

    # Resume from checkpoint
    python training/pretrain.py --config configs/pretraining.yaml \
        --resume outputs/pretraining/adapter_step_4000.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.pretrain_dataset import PretrainDataset, collate_skip_none  # noqa: E402
from models.sigllama_pretrain import SigLlamaForPretraining  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("pretrain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Adapter pre-training")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--resume", default=None, help="Adapter checkpoint to resume from")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("output_dir", "outputs/pretraining")

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )
    set_seed(cfg.get("seed", 0))

    # ---- tokenizer & processor ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoImageProcessor.from_pretrained(cfg["siglip_model"])

    # ---- dataset ----
    full_dataset = PretrainDataset(
        metadata_path=cfg["metadata_path"],
        image_root=cfg["image_root"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=cfg.get("max_text_length", 128),
    )

    val_size = min(cfg.get("val_size", 1000), int(0.01 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 0)),
    )
    accelerator.print(f"Train: {train_size}  |  Val: {val_size}")

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=collate_skip_none,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=collate_skip_none,
        pin_memory=True,
    )

    # ---- model ----
    model = SigLlamaForPretraining(
        siglip_model=cfg["siglip_model"],
        llm_model=cfg["llm_model"],
        visual_dim=cfg.get("visual_dim", 768),
        llm_dim=cfg.get("llm_dim", 2048),
    )

    if cfg.get("gradient_checkpointing", True):
        model.llm.gradient_checkpointing_enable()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "adapter" in ckpt:
            model.adapter.load_state_dict(ckpt["adapter"])
            accelerator.print(f"Resumed adapter from {args.resume} (step {ckpt.get('step', '?')})")
        else:
            model.adapter.load_state_dict(ckpt)
            accelerator.print(f"Resumed adapter from {args.resume}")

    trainable = sum(p.numel() for p in model.adapter.parameters())
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(
        f"Parameters — trainable: {trainable:,}  |  total: {total:,}  "
        f"({100 * trainable / total:.2f}%)"
    )

    # ---- optimiser & scheduler ----
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
        betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
    )

    num_epochs = cfg.get("max_epochs", 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = (
        cfg["warmup_steps"]
        if "warmup_steps" in cfg
        else int(cfg.get("warmup_ratio", 0.03) * total_steps)
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    accelerator.print(
        f"Schedule — epochs: {num_epochs}  |  steps/epoch: {steps_per_epoch}  "
        f"|  total: {total_steps}  |  warmup: {warmup_steps}"
    )

    # ---- accelerate prepare ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---- intervals ----
    log_interval = cfg.get("log_interval", 100)
    save_interval = cfg.get("save_interval", 2000)
    val_interval = cfg.get("val_interval", 2000)
    sample_interval = cfg.get("sample_interval", 5000)

    # ---- training loop ----
    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            if batch is None:
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients and cfg.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg["max_grad_norm"]
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            running_count += 1
            global_step += 1

            # -- log --
            if global_step % log_interval == 0:
                avg = running_loss / running_count
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", avg=f"{avg:.4f}", lr=f"{lr:.2e}"
                )
                if accelerator.is_main_process:
                    _log(output_dir, {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": loss.item(), "avg_loss": avg, "lr": lr,
                        "elapsed_min": (time.time() - t0) / 60,
                    })

            # -- save (overwrites previous → only 2 files on disk: latest + best) --
            if global_step % save_interval == 0:
                _save_adapter(accelerator, model, output_dir,
                              "latest_adapter.pt", step=global_step)

            # -- validate --
            if global_step % val_interval == 0:
                vl = _validate(model, val_loader, accelerator)
                accelerator.print(
                    f"  [step {global_step}] val_loss={vl:.4f}"
                )
                if accelerator.is_main_process and vl < best_val_loss:
                    best_val_loss = vl
                    _save_adapter(accelerator, model, output_dir,
                                  "best_adapter.pt", step=global_step)
                    accelerator.print(f"  -> new best saved (val_loss={vl:.4f})")
                model.train()

            # -- sample generation --
            if global_step % sample_interval == 0 and accelerator.is_main_process:
                _generate_samples(
                    model, val_ds, processor, tokenizer,
                    accelerator, output_dir, global_step,
                )
                model.train()

        # end of epoch
        _save_adapter(accelerator, model, output_dir,
                      "latest_adapter.pt", step=global_step)
        avg = running_loss / max(running_count, 1)
        accelerator.print(
            f"Epoch {epoch + 1} done | avg_loss={avg:.4f} | "
            f"elapsed={((time.time() - t0) / 60):.1f} min"
        )

    accelerator.print("Training complete.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _validate(model, loader, accelerator) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            total += model(**batch)["loss"].item()
            count += 1
    return total / max(count, 1)


def _save_adapter(accelerator, model, output_dir, filename, step: int = 0) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    path = os.path.join(output_dir, filename)
    torch.save({"step": step, "adapter": unwrapped.adapter.state_dict()}, path)
    accelerator.print(f"  checkpoint: {path}  (step {step})")


@torch.no_grad()
def _generate_samples(
    model, val_ds, processor, tokenizer, accelerator, output_dir, step,
    n: int = 5,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    device = accelerator.device

    lines: list[str] = [f"\n--- step {step} ---"]
    for i in range(min(n, len(val_ds))):
        sample = val_ds[i]
        if sample is None:
            continue
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        gen = unwrapped.generate(pv, tokenizer, max_new_tokens=64)
        ref = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        lines.append(f"[{i}] ref : {ref}")
        lines.append(f"     gen : {gen[0]}")

    text = "\n".join(lines) + "\n"
    accelerator.print(text)
    with open(os.path.join(output_dir, "samples.txt"), "a") as f:
        f.write(text)


def _log(output_dir: str, entry: dict) -> None:
    with open(os.path.join(output_dir, "train_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
