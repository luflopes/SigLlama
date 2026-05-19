"""Stage A: Train DINOv2 with LoRA (single or MoE) + dual supervision.

Fine-tunes LoRA adapters on DINOv2 for binary deepfake detection
and forgery-type classification simultaneously.  The trained LoRA
weights are then used downstream by the VLM adapter (Stage 1').

Usage::

    # Single LoRA
    python training/train_dino_lora.py --config configs/dino_lora_classifier.yaml

    # LoRA-MoE
    python training/train_dino_lora.py --config configs/dino_lora_moe_classifier.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from tqdm import tqdm

from peft import get_peft_model_state_dict, set_peft_model_state_dict

from models.dino_lora_classifier import DINOv2LoRAClassifier, METHOD_TO_IDX

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("train_dino_lora")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Dataset ──────────────────────────────────────────────────────────

class FFDualSupDataset(Dataset):
    """FF++ dataset returning pixel_values, binary label, and method index."""

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        image_size: int = 384,
        augmentation: bool = False,
    ):
        self.image_root = image_root
        if augmentation:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])

        self.samples: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_real = sum(1 for s in self.samples if s["label"] == 0)
        n_fake = len(self.samples) - n_real
        methods = {}
        for s in self.samples:
            m = s.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        logger.info(
            "Loaded %d samples from %s (real=%d, fake=%d, methods=%s)",
            len(self.samples), metadata_path, n_real, n_fake, methods,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img_path = os.path.join(self.image_root, row["image"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        binary_label = int(row["label"])
        method = row.get("method", "unknown")
        method_idx = METHOD_TO_IDX.get(method, -1)
        return pixel_values, binary_label, method_idx


# ── Dataloaders ──────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    image_root = os.path.join(cfg["data_root"], "frames")
    image_size = int(cfg.get("image_size", 384))
    augmentation = bool(cfg.get("augmentation", True))

    train_ds = FFDualSupDataset(
        os.path.join(cfg["data_root"], "train.jsonl"), image_root, image_size,
        augmentation=augmentation,
    )
    val_ds = FFDualSupDataset(
        os.path.join(cfg["data_root"], "val.jsonl"), image_root, image_size,
        augmentation=False,
    )

    max_train = cfg.get("max_train_samples")
    if max_train and max_train < len(train_ds):
        train_ds.samples = train_ds.samples[:max_train]
        logger.info("Sub-sampled training set to %d samples", max_train)

    max_val = cfg.get("max_val_samples")
    if max_val and max_val < len(val_ds):
        val_ds.samples = val_ds.samples[:max_val]
        logger.info("Sub-sampled validation set to %d samples", max_val)

    n_real = sum(1 for s in train_ds.samples if s["label"] == 0)
    n_fake = len(train_ds.samples) - n_real
    if n_real > 0 and n_fake > 0:
        wr = 1.0 / n_real
        wf = 1.0 / n_fake
        weights = [wr if s["label"] == 0 else wf for s in train_ds.samples]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    nw = int(cfg.get("num_workers", 4))
    bs = int(cfg["batch_size"])

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=train_shuffle, sampler=sampler,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
    )
    return train_loader, val_loader


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: DINOv2LoRAClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    bin_correct = 0
    ft_correct = 0
    total_bin_loss = 0.0
    total_ft_loss = 0.0
    tp = fp = tn = fn = 0

    for pixels, bin_labels, method_labels in tqdm(loader, desc="Evaluating", leave=False):
        pixels = pixels.to(device)
        bin_labels = bin_labels.to(device)
        method_labels = method_labels.to(device)

        out = model(pixels, binary_labels=bin_labels, method_labels=method_labels)

        bs = bin_labels.size(0)
        total += bs

        total_bin_loss += out["binary_loss"].item() * bs
        bin_preds = out["binary_logits"].argmax(dim=-1)
        bin_correct += (bin_preds == bin_labels).sum().item()

        tp += ((bin_preds == 1) & (bin_labels == 1)).sum().item()
        fp += ((bin_preds == 1) & (bin_labels == 0)).sum().item()
        tn += ((bin_preds == 0) & (bin_labels == 0)).sum().item()
        fn += ((bin_preds == 0) & (bin_labels == 1)).sum().item()

        valid = method_labels >= 0
        if valid.any():
            total_ft_loss += out["forgery_loss"].item() * valid.sum().item()
            ft_preds = out["forgery_logits"][valid].argmax(dim=-1)
            ft_correct += (ft_preds == method_labels[valid]).sum().item()

    acc = bin_correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "binary_accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_loss": total_bin_loss / max(total, 1),
        "forgery_accuracy": ft_correct / max(total, 1),
        "forgery_loss": total_ft_loss / max(total, 1),
        "total": total,
    }


# ── LR Scheduler ─────────────────────────────────────────────────────

def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_type = cfg.get("lr_scheduler")
    if not sched_type:
        return None

    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    max_epochs = int(cfg.get("max_epochs", 20))
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch

    if sched_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 1e-2)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-2)

        scheduler = LambdaLR(optimizer, lr_lambda)
        logger.info(
            "LR scheduler: cosine (warmup=%d steps, total=%d steps)",
            warmup_steps, total_steps,
        )
        return scheduler

    logger.warning("Unknown lr_scheduler '%s', skipping", sched_type)
    return None


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 LoRA classifier (Stage A)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    use_moe = bool(cfg.get("use_moe", False))

    model = DINOv2LoRAClassifier(
        dino_model=cfg.get("dinov2_model", "facebook/dinov2-large"),
        lora_rank=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        lora_target_modules=cfg.get("lora_target_modules"),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 256)),
        head_dropout=float(cfg.get("head_dropout", 0.3)),
        use_moe=use_moe,
        num_experts=int(cfg.get("num_experts", 6)),
        router_hidden_dim=int(cfg.get("router_hidden_dim", 256)),
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.binary_head.load_state_dict(ckpt["binary_head"])
        model.forgery_head.load_state_dict(ckpt["forgery_head"])
        if "lora" in ckpt:
            set_peft_model_state_dict(model.dinov2, ckpt["lora"])
        if use_moe and "expert_lora_params" in ckpt:
            model.expert_lora_params.load_state_dict(ckpt["expert_lora_params"])
        if use_moe and "router" in ckpt:
            model.router.load_state_dict(ckpt["router"])
        logger.info("Resumed from %s", args.resume)

    model.to(device)

    # Collect trainable parameters
    lora_params = [p for p in model.dinov2.parameters() if p.requires_grad]
    head_params = list(model.binary_head.parameters()) + list(model.forgery_head.parameters())
    opt_groups = [
        {"params": lora_params, "lr": float(cfg.get("lora_lr", 2e-4))},
        {"params": head_params, "lr": float(cfg.get("head_lr", 1e-3))},
    ]

    if use_moe:
        expert_params = list(model.expert_lora_params.parameters())
        router_params = list(model.router.parameters())
        opt_groups.append({"params": expert_params, "lr": float(cfg.get("lora_lr", 2e-4))})
        opt_groups.append({"params": router_params, "lr": float(cfg.get("router_lr", 1e-4))})

    total_trainable = sum(p.numel() for g in opt_groups for p in g["params"])
    logger.info("Total trainable parameters: %d (~%.2fM)", total_trainable, total_trainable / 1e6)

    optimizer = AdamW(opt_groups, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    train_loader, val_loader = build_dataloaders(cfg)

    max_epochs = int(cfg.get("max_epochs", 20))
    patience = int(cfg.get("early_stopping_patience", 5))
    forgery_loss_weight = float(cfg.get("forgery_loss_weight", 0.2))
    router_loss_weight = float(cfg.get("router_loss_weight", 0.1))

    best_val_acc = 0.0
    best_epoch = -1
    epochs_without_improvement = 0

    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    for epoch in range(max_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_bin_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for pixels, bin_labels, method_labels in pbar:
            pixels = pixels.to(device)
            bin_labels = bin_labels.to(device)
            method_labels = method_labels.to(device)

            out = model(pixels, binary_labels=bin_labels, method_labels=method_labels)

            loss = out["binary_loss"] + forgery_loss_weight * out.get("forgery_loss", 0.0)
            if use_moe:
                loss = loss + router_loss_weight * out.get("router_loss", 0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt_groups for p in g["params"]], 1.0
            )
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            bs = bin_labels.size(0)
            epoch_loss += loss.item() * bs
            bin_preds = out["binary_logits"].argmax(dim=-1)
            epoch_bin_correct += (bin_preds == bin_labels).sum().item()
            epoch_total += bs

            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{epoch_bin_correct / max(epoch_total, 1):.3f}",
                lr=f"{current_lr:.2e}",
            )

        train_acc = epoch_bin_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        val_metrics = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "epoch=%d lr=%.2e train_loss=%.4f train_acc=%.4f | "
            "val_bin_acc=%.4f val_f1=%.4f val_ft_acc=%.4f",
            epoch, current_lr, train_loss, train_acc,
            val_metrics["binary_accuracy"], val_metrics["f1"],
            val_metrics["forgery_accuracy"],
        )

        if val_metrics["binary_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["binary_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_dict = {
                "epoch": epoch,
                "binary_head": model.binary_head.state_dict(),
                "forgery_head": model.forgery_head.state_dict(),
                "lora": get_peft_model_state_dict(model.dinov2),
                "val_metrics": val_metrics,
                "config": {
                    "dinov2_model": cfg.get("dinov2_model", "facebook/dinov2-large"),
                    "lora_rank": cfg.get("lora_rank", 16),
                    "lora_alpha": cfg.get("lora_alpha", 32),
                    "lora_target_modules": cfg.get("lora_target_modules", ["query", "value"]),
                    "use_moe": use_moe,
                    "num_experts": cfg.get("num_experts", 6),
                },
            }
            if use_moe:
                save_dict["expert_lora_params"] = model.expert_lora_params.state_dict()
                save_dict["router"] = model.router.state_dict()

            torch.save(save_dict, output_dir / "best.pt")
            logger.info("  -> New best val_bin_acc=%.4f (saved best.pt)", best_val_acc)
        else:
            epochs_without_improvement += 1

        # Always save last
        last_dict = {
            "epoch": epoch,
            "binary_head": model.binary_head.state_dict(),
            "forgery_head": model.forgery_head.state_dict(),
            "lora": get_peft_model_state_dict(model.dinov2),
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        if use_moe:
            last_dict["expert_lora_params"] = model.expert_lora_params.state_dict()
            last_dict["router"] = model.router.state_dict()
        torch.save(last_dict, output_dir / "last.pt")

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping: no improvement for %d epochs (best=%.4f at epoch %d)",
                patience, best_val_acc, best_epoch,
            )
            break

    logger.info(
        "Training complete. Best val_bin_acc=%.4f at epoch %d (ran %d/%d epochs)",
        best_val_acc, best_epoch, epoch + 1, max_epochs,
    )

    results = {
        "best_val_binary_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs_run": epoch + 1,
        "max_epochs": max_epochs,
        "use_moe": use_moe,
        "early_stopped": epochs_without_improvement >= patience if patience > 0 else False,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
