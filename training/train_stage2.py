"""Stage 2: DINO adapter + LoRA fine-tuning on DD-VQA (FaceGroundVLM)."""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from data.ddvqa_dataset import DDVQADataset, _sample_is_real, collate_ddvqa
from data.prompt_formats import build_generation_inputs
from models import build_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from training.factory import build_processor_and_transforms
from training.sample_utils import decode_ids, save_samples

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceGroundVLM stage 2 (adapter + LoRA)")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume (adapter, lora, optimizer state)",
    )
    return p.parse_args()


def build_train_sampler(
    train_ds: DDVQADataset, cfg: dict
) -> WeightedRandomSampler | None:
    if not cfg.get("balanced_sampling", False):
        logger.info("balanced_sampling disabled; using natural class distribution")
        return None
    n_real = sum(1 for s in train_ds.samples if _sample_is_real(s))
    n_fake = len(train_ds.samples) - n_real
    if n_real == 0 or n_fake == 0:
        logger.warning(
            "balanced_sampling requested but dataset has only one class "
            "(real=%d, fake=%d); sampler disabled.",
            n_real, n_fake,
        )
        return None
    wr = 1.0 / n_real
    wf = 1.0 / n_fake
    weights = [
        wr if _sample_is_real(s) else wf for s in train_ds.samples
    ]
    logger.info(
        "balanced_sampling enabled: n_real=%d, n_fake=%d "
        "(weights: real=%.2e, fake=%.2e). Expected per-epoch ratio = 50/50.",
        n_real, n_fake, wr, wf,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(cfg: dict, tokenizer, image_processor, dino_transform):
    backbone = cfg.get("backbone", "paligemma")
    max_len = cfg.get("max_text_length", 256)
    train_ds = DDVQADataset(
        metadata_path=cfg["train_metadata"],
        image_root=cfg["image_root"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        dino_transform=dino_transform,
        backbone=backbone,
        max_length=max_len,
    )
    val_ds = DDVQADataset(
        metadata_path=cfg["val_metadata"],
        image_root=cfg["image_root"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        dino_transform=dino_transform,
        backbone=backbone,
        max_length=max_len,
    )
    sampler = build_train_sampler(train_ds, cfg)
    nw = cfg["num_workers"]
    train_kwargs: dict = {
        "batch_size": cfg["batch_size"],
        "num_workers": nw,
        "collate_fn": collate_ddvqa,
        "pin_memory": True,
        "persistent_workers": nw > 0,
        "prefetch_factor": 2 if nw > 0 else None,
    }
    if sampler is not None:
        train_kwargs["sampler"] = sampler
        train_kwargs["shuffle"] = False
    else:
        train_kwargs["shuffle"] = True
    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_ddvqa,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    return train_loader, val_loader


def validation_loss(model, val_loader, accelerator) -> float | None:
    if val_loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    pbar = tqdm(
        val_loader,
        desc="Validating",
        leave=False,
        disable=not accelerator.is_main_process,
    )
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            batch = {
                k: v
                for k, v in batch.items()
                if k in ("pixel_values_siglip", "pixel_values_dino", "input_ids", "attention_mask", "labels")
            }
            outputs = model(**batch)
            loss = outputs["loss"]
            total += loss.detach().float().item()
            count += 1
            if count > 0:
                pbar.set_postfix(avg_loss=f"{total / count:.4f}")
    model.train()
    if count == 0:
        return None
    t = torch.tensor(
        [total, float(count)], device=accelerator.device, dtype=torch.float32
    )
    t = accelerator.reduce(t, reduction="sum")
    return (t[0] / t[1]).item() if t[1] > 0 else None


def maybe_sample(
    model,
    tokenizer,
    batch,
    accelerator,
    output_dir: Path,
    global_step: int,
    backbone: str,
    max_new_tokens: int = 128,
) -> None:
    if batch is None:
        return
    model_unwrap = accelerator.unwrap_model(model)
    questions = batch.get("question", [])
    if not questions:
        # Fallback: fall back to the train input (shouldn't happen with the
        # updated DDVQADataset but keeps the function robust).
        references = decode_ids(batch.get("labels", batch["input_ids"]), tokenizer)
        logger.warning("maybe_sample: no 'question' field; skipping generation")
        return

    gen_inputs = build_generation_inputs(questions, tokenizer, backbone)
    device = batch["pixel_values_siglip"].device
    with torch.no_grad():
        gen_ids = model_unwrap.generate(
            pixel_values_siglip=batch["pixel_values_siglip"],
            pixel_values_dino=batch["pixel_values_dino"],
            input_ids=gen_inputs["input_ids"].to(device),
            attention_mask=gen_inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
    generated = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    references = list(batch.get("answer", []))

    logger.info(
        "sample gen[0]: Q=%r -> %r (ref=%r)",
        questions[0][:80] if questions else "",
        generated[0][:200] if generated else "",
        references[0][:120] if references else "",
    )

    save_samples(
        batch["pixel_values_dino"], generated, references,
        output_dir, global_step,
    )


def load_adapter_checkpoint(model, path: str) -> None:
    if model.dino_adapter is None:
        logger.info("use_dino=False: skipping adapter checkpoint load")
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["adapter"] if isinstance(ckpt, dict) and "adapter" in ckpt else ckpt
    model.dino_adapter.load_state_dict(state)


def load_lora_checkpoint(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    lora_sd = ckpt.get("lora", ckpt)
    set_peft_model_state_dict(model.language_model, lora_sd)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
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

    proc = build_processor_and_transforms(cfg)
    tokenizer = proc["tokenizer"]
    image_processor = proc["image_processor"]
    dino_transform = proc["dino_transform"]
    backbone = cfg.get("backbone", "paligemma")

    model = build_model(cfg, use_lora=True)
    model.enable_input_require_grads()

    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    if cfg.get("adapter_checkpoint"):
        load_adapter_checkpoint(model, cfg["adapter_checkpoint"])
    if cfg.get("lora_checkpoint"):
        load_lora_checkpoint(model, cfg["lora_checkpoint"])

    train_loader, val_loader = build_dataloaders(
        cfg, tokenizer, image_processor, dino_transform,
    )

    adapter_params = (
        list(model.dino_adapter.parameters()) if model.dino_adapter is not None else []
    )
    connector_params: list = []
    if bool(cfg.get("train_connector", False)) and hasattr(model, "connector"):
        # Only include params that were actually unfrozen by the model
        # (TinyLLaVAGroundVLM does this when ``train_connector=True``).
        connector_params = [
            p for p in model.connector.parameters() if p.requires_grad
        ]
    lora_params = [
        p for p in model.language_model.parameters() if p.requires_grad
    ]
    opt_groups: list[dict] = []
    group_names: list[str] = []
    if adapter_params:
        opt_groups.append({"params": adapter_params, "lr": float(cfg["adapter_lr"])})
        group_names.append("adapter")
    if connector_params:
        opt_groups.append(
            {"params": connector_params, "lr": float(cfg.get("connector_lr", 5.0e-6))}
        )
        group_names.append("connector")
    opt_groups.append({"params": lora_params, "lr": float(cfg["lora_lr"])})
    group_names.append("lora")
    optimizer = AdamW(opt_groups)
    logger.info(
        "Optimizer groups: %s",
        [(n, sum(p.numel() for p in g["params"]), g["lr"]) for n, g in zip(group_names, opt_groups)],
    )

    grad_accum = int(cfg["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    max_epochs = int(cfg["max_epochs"])
    max_train_steps = steps_per_epoch * max_epochs
    warmup_ratio = float(cfg["warmup_ratio"])
    warmup_steps = int(max_train_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if model.dino_adapter is not None and "adapter" in ckpt:
            model.dino_adapter.load_state_dict(ckpt["adapter"])
        if "connector" in ckpt and hasattr(model, "connector"):
            model.connector.load_state_dict(ckpt["connector"])
        if "lora" in ckpt:
            set_peft_model_state_dict(
                model.language_model, ckpt["lora"]
            )
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        logger.info("Resumed from step %s", start_step)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    log_interval = int(cfg["log_interval"])
    save_interval = int(cfg["save_interval"])
    val_interval = int(cfg["val_interval"])
    sample_interval = int(cfg["sample_interval"])
    max_grad_norm = float(cfg["max_grad_norm"])

    global_step = start_step
    sample_batch = None
    for sb in val_loader:
        if sb is not None:
            sample_batch = sb
            break
    if sample_batch is None:
        for sb in train_loader:
            if sb is not None:
                sample_batch = sb
                break

    clip_params: list = []
    for group in optimizer.param_groups:
        clip_params.extend(group["params"])

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
            batch = {
                k: v
                for k, v in batch.items()
                if k
                in (
                    "pixel_values_siglip",
                    "pixel_values_dino",
                    "input_ids",
                    "attention_mask",
                    "labels",
                )
            }
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(clip_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    li = loss.detach().float().item()
                    epoch_loss += li
                    n_batches += 1

                    lrs = scheduler.get_last_lr()
                    postfix = {"loss": f"{li:.4f}", "step": global_step}
                    for name, lr in zip(group_names, lrs):
                        postfix[f"lr_{name[:1]}"] = f"{lr:.2e}"
                    pbar.set_postfix(**postfix)

                    if global_step % log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            "epoch=%s step=%s loss=%.6f lr=%s",
                            epoch,
                            global_step,
                            li,
                            [f"{x:.2e}" for x in lrs],
                        )

                    # Save BEFORE sampling/validation: a crash in
                    # generate() must never lose progress.
                    if (
                        save_interval > 0
                        and global_step % save_interval == 0
                        and accelerator.is_main_process
                    ):
                        um = accelerator.unwrap_model(model)
                        ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                        state = {
                            "step": global_step,
                            "lora": get_peft_model_state_dict(um.language_model),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }
                        if um.dino_adapter is not None:
                            state["adapter"] = um.dino_adapter.state_dict()
                        if getattr(um, "train_connector", False) and hasattr(um, "connector"):
                            state["connector"] = um.connector.state_dict()
                        torch.save(state, ckpt_path)
                        logger.info("Saved %s", ckpt_path)

                    if (
                        val_interval > 0
                        and global_step % val_interval == 0
                        and accelerator.is_main_process
                    ):
                        try:
                            vloss = validation_loss(model, val_loader, accelerator)
                            if vloss is not None:
                                logger.info("validation loss: %.6f", vloss)
                        except Exception as e:
                            logger.warning(
                                "Validation at step %s failed (skipped): %s",
                                global_step, e,
                            )

                    if (
                        sample_interval > 0
                        and global_step % sample_interval == 0
                        and accelerator.is_main_process
                    ):
                        try:
                            maybe_sample(
                                model, tokenizer, sample_batch, accelerator,
                                output_dir, global_step, backbone=backbone,
                            )
                        except Exception as e:
                            logger.warning(
                                "Sampling at step %s failed (skipped): %s",
                                global_step, e,
                            )

        if accelerator.is_main_process and n_batches > 0:
            logger.info(
                "epoch %s done | mean train loss=%.6f",
                epoch,
                epoch_loss / n_batches,
            )

        # End-of-epoch fallback: ensures at least one validation + one sample
        # dump per epoch regardless of how val/sample_interval were configured.
        if accelerator.is_main_process:
            try:
                vloss = validation_loss(model, val_loader, accelerator)
                if vloss is not None:
                    logger.info("[epoch %s end] validation loss: %.6f", epoch, vloss)
            except Exception as e:
                logger.warning("End-of-epoch validation failed (skipped): %s", e)
            try:
                maybe_sample(
                    model, tokenizer, sample_batch, accelerator,
                    output_dir, global_step, backbone=backbone,
                )
            except Exception as e:
                logger.warning("End-of-epoch sampling failed (skipped): %s", e)

    if accelerator.is_main_process:
        um = accelerator.unwrap_model(model)
        final_path = output_dir / "checkpoint-final.pt"
        state = {
            "step": global_step,
            "lora": get_peft_model_state_dict(um.language_model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if um.dino_adapter is not None:
            state["adapter"] = um.dino_adapter.state_dict()
        if getattr(um, "train_connector", False) and hasattr(um, "connector"):
            state["connector"] = um.connector.state_dict()
        torch.save(state, final_path)
        logger.info("Saved %s", final_path)


if __name__ == "__main__":
    main()
