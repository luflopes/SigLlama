"""Stage 2a: Fine-tune SigLlama for deepfake VQA (no MoE, no soft tokens).

Trains the Adapter + LoRA on TinyLlama using the DD-VQA dataset.

Usage::

    # Single GPU
    python training/finetune.py --config configs/finetuning.yaml

    # Multi-GPU via accelerate
    accelerate launch training/finetune.py --config configs/finetuning.yaml

    # Resume from checkpoint
    python training/finetune.py --config configs/finetuning.yaml \
        --resume outputs/finetuning/latest_checkpoint
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.deepfake_vqa_dataset import DeepfakeVQADataset, collate_vqa  # noqa: E402
from models.sigllama_finetune import SigLlamaForFinetune  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("finetune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2a: Deepfake VQA fine-tuning")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("output_dir", "outputs/finetuning")

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
    )
    set_seed(cfg.get("seed", 42))

    # ---- tokenizer & processor ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoImageProcessor.from_pretrained(cfg["siglip_model"])

    # ---- datasets ----
    train_ds = DeepfakeVQADataset(
        metadata_path=cfg["train_metadata"],
        image_root=cfg["image_root"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=cfg.get("max_text_length", 256),
    )
    val_ds = DeepfakeVQADataset(
        metadata_path=cfg["val_metadata"],
        image_root=cfg["image_root"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=cfg.get("max_text_length", 256),
    )
    accelerator.print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    bs = cfg.get("batch_size", 16)
    nw = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=collate_vqa,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=collate_vqa,
        pin_memory=True,
    )

    # ---- model ----
    lora_target = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
    if isinstance(lora_target, str):
        lora_target = [m.strip() for m in lora_target.split(",")]

    model = SigLlamaForFinetune(
        siglip_model=cfg["siglip_model"],
        llm_model=cfg["llm_model"],
        visual_dim=cfg.get("visual_dim", 768),
        llm_dim=cfg.get("llm_dim", 2048),
        adapter_checkpoint=cfg.get("adapter_checkpoint"),
        lora_rank=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_target_modules=lora_target,
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )

    if cfg.get("gradient_checkpointing", True):
        model.llm.gradient_checkpointing_enable()

    accelerator.print(model.print_trainable_parameters())

    # ---- optimizer ----
    trainable_params = [
        {"params": model.adapter.parameters(), "lr": cfg.get("adapter_lr", cfg.get("learning_rate", 2e-4))},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": cfg.get("learning_rate", 2e-4)},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
    )

    # ---- scheduler ----
    num_epochs = cfg.get("max_epochs", 5)
    grad_accum = cfg.get("gradient_accumulation_steps", 2)
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

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isdir(args.resume):
        accelerator.load_state(args.resume)
        meta_path = os.path.join(args.resume, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            global_step = meta.get("global_step", 0)
            start_epoch = meta.get("epoch", 0)
            best_val_loss = meta.get("best_val_loss", float("inf"))
        accelerator.print(f"Resumed from {args.resume} (step {global_step})")

    # ---- accelerate prepare ----
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---- intervals ----
    log_interval = cfg.get("log_interval", 50)
    save_interval = cfg.get("save_interval", 500)
    val_interval = cfg.get("val_interval", 500)
    sample_interval = cfg.get("sample_interval", 1000)

    # ---- training loop ----
    t0 = time.time()

    for epoch in range(start_epoch, num_epochs):
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

            if global_step % save_interval == 0:
                _save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    output_dir, "latest_checkpoint",
                    global_step=global_step, epoch=epoch,
                    best_val_loss=best_val_loss,
                )

            if global_step % val_interval == 0:
                vl = _validate(model, val_loader, accelerator)
                accelerator.print(f"  [step {global_step}] val_loss={vl:.4f}")
                if accelerator.is_main_process:
                    _log(output_dir, {
                        "step": global_step, "val_loss": vl,
                        "best_val_loss": min(best_val_loss, vl),
                    })
                if vl < best_val_loss:
                    best_val_loss = vl
                    _save_checkpoint(
                        accelerator, model, optimizer, scheduler,
                        output_dir, "best_checkpoint",
                        global_step=global_step, epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
                    accelerator.print(f"  -> new best saved (val_loss={vl:.4f})")
                model.train()

            if global_step % sample_interval == 0 and accelerator.is_main_process:
                _generate_samples(
                    model, val_ds, processor, tokenizer,
                    accelerator, output_dir, global_step,
                )
                model.train()

        # End of epoch
        _save_checkpoint(
            accelerator, model, optimizer, scheduler,
            output_dir, "latest_checkpoint",
            global_step=global_step, epoch=epoch + 1,
            best_val_loss=best_val_loss,
        )
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


def _save_checkpoint(
    accelerator, model, optimizer, scheduler,
    output_dir: str, name: str,
    global_step: int = 0, epoch: int = 0,
    best_val_loss: float = float("inf"),
) -> None:
    ckpt_dir = os.path.join(output_dir, name)

    unwrapped = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(
            {"adapter": unwrapped.adapter.state_dict()},
            os.path.join(ckpt_dir, "adapter.pt"),
        )

        unwrapped.llm.save_pretrained(os.path.join(ckpt_dir, "lora"))

        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, f)

        accelerator.print(f"  checkpoint: {ckpt_dir}  (step {global_step})")


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

        raw = val_ds.samples[i]
        question = raw["question"]

        gen = unwrapped.generate(
            pv, tokenizer, prompt=question, max_new_tokens=128,
            temperature=0.7, top_p=0.9, repetition_penalty=1.2,
        )

        ref_ids = sample["input_ids"]
        ref = tokenizer.decode(ref_ids, skip_special_tokens=True)

        lines.append(f"[{i}] Q: {question}")
        lines.append(f"     ref : {ref}")
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
