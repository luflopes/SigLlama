"""Stage 1: DINOv2 adapter pre-training on LCS-558K (FaceGroundVLM)."""
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
from data.lcs558k_dataset import LCS558KDataset, collate_skip_none
from models.face_ground_vlm import FaceGroundVLM
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from training.sample_utils import decode_ids, save_samples

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceGroundVLM stage 1 (DINO adapter)")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume (adapter + optimizer state)",
    )
    return p.parse_args()


def build_dataloaders(cfg: dict, processor, dino_transform):
    full = LCS558KDataset(
        metadata_path=cfg["metadata_path"],
        image_root=cfg["image_root"],
        processor=processor,
        dino_transform=dino_transform,
        max_length=cfg.get("max_text_length", 128),
    )
    val_size = int(cfg["val_size"])
    if val_size <= 0:
        train_ds = full
        val_ds = None
    else:
        n = len(full)
        val_size = min(val_size, n - 1) if n > 1 else 0
        if val_size <= 0:
            train_ds = full
            val_ds = None
        else:
            g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
            train_ds, val_ds = random_split(
                full, [n - val_size, val_size], generator=g
            )

    nw = cfg["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_skip_none,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_skip_none,
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
    max_new_tokens: int = 128,
) -> None:
    if batch is None:
        return
    model_unwrap = accelerator.unwrap_model(model)
    gen_batch = {
        k: v
        for k, v in batch.items()
        if k
        in ("pixel_values_siglip", "pixel_values_dino", "input_ids", "attention_mask")
    }
    with torch.no_grad():
        gen_ids = model_unwrap.generate(
            **gen_batch,
            max_new_tokens=max_new_tokens,
        )
    generated = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    references = decode_ids(batch.get("labels", batch["input_ids"]), tokenizer)

    logger.info("sample generation: %s", generated[: min(2, len(generated))])

    save_samples(
        batch["pixel_values_dino"], generated, references,
        output_dir, global_step,
    )


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

    processor = AutoProcessor.from_pretrained(cfg["paligemma_model"])
    tokenizer = processor.tokenizer

    dino_transform = Compose(
        [
            Resize((448, 448)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = FaceGroundVLM(
        paligemma_model=cfg["paligemma_model"],
        dinov2_model=cfg["dinov2_model"],
        mof_strategy=cfg["mof_strategy"],
        use_lora=False,
    )
    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    train_loader, val_loader = build_dataloaders(cfg, processor, dino_transform)

    lr = float(cfg.get("learning_rate", 1e-4))
    optimizer = SGD(
        model.dino_adapter.parameters(),
        lr=lr,
        momentum=0.9,
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
        model.dino_adapter.load_state_dict(ckpt["adapter"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        logger.info("Resumed from step %s", start_step)

    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    log_interval = int(cfg["log_interval"])
    save_interval = int(cfg["save_interval"])
    val_interval = int(cfg["val_interval"])
    sample_interval = int(cfg["sample_interval"])
    max_grad_norm = float(cfg["max_grad_norm"])

    global_step = start_step
    sample_batch = None
    if val_loader is not None:
        for sb in val_loader:
            if sb is not None:
                sample_batch = sb
                break
    if sample_batch is None:
        for sb in train_loader:
            if sb is not None:
                sample_batch = sb
                break

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
                    accelerator.clip_grad_norm_(
                        accelerator.unwrap_model(model).dino_adapter.parameters(),
                        max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    li = loss.detach().float().item()
                    epoch_loss += li
                    n_batches += 1

                    pbar.set_postfix(
                        loss=f"{li:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=global_step,
                    )

                    if global_step % log_interval == 0 and accelerator.is_main_process:
                        logger.info(
                            "epoch=%s step=%s loss=%.6f lr=%.2e",
                            epoch,
                            global_step,
                            li,
                            scheduler.get_last_lr()[0],
                        )

                    if (
                        sample_interval > 0
                        and global_step % sample_interval == 0
                        and accelerator.is_main_process
                    ):
                        maybe_sample(
                            model, tokenizer, sample_batch, accelerator,
                            output_dir, global_step,
                        )

                    if (
                        val_loader is not None
                        and val_interval > 0
                        and global_step % val_interval == 0
                        and accelerator.is_main_process
                    ):
                        vloss = validation_loss(model, val_loader, accelerator)
                        if vloss is not None:
                            logger.info("validation loss: %.6f", vloss)

                    if (
                        save_interval > 0
                        and global_step % save_interval == 0
                        and accelerator.is_main_process
                    ):
                        ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                        torch.save(
                            {
                                "step": global_step,
                                "adapter": accelerator.unwrap_model(
                                    model
                                ).dino_adapter.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                            },
                            ckpt_path,
                        )
                        logger.info("Saved %s", ckpt_path)

        if accelerator.is_main_process and n_batches > 0:
            logger.info(
                "epoch %s done | mean train loss=%.6f",
                epoch,
                epoch_loss / n_batches,
            )

    if accelerator.is_main_process:
        final_path = output_dir / "checkpoint-final.pt"
        torch.save(
            {
                "step": global_step,
                "adapter": accelerator.unwrap_model(model).dino_adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            final_path,
        )
        logger.info("Saved %s", final_path)


if __name__ == "__main__":
    main()
