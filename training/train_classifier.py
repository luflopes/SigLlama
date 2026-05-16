"""Train the DINOv2 binary classification head on FaceForensics++.

Only the MLP head (~66K parameters) is trained; the DINOv2 backbone
stays frozen. Training is fast (minutes on a single GPU).

Usage::

    python training/train_classifier.py \
        --config configs/dino_classifier.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from tqdm import tqdm

from models.dino_classifier import DINOv2Classifier

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("train_classifier")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_train_transform(image_size: int, augmentation: bool) -> T.Compose:
    """Build transform for training; includes geometric + color augmentation."""
    if augmentation:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class FFClassificationDataset(Dataset):
    """JSONL dataset with ``image``, ``label`` (0=real, 1=fake), ``method``."""

    def __init__(self, metadata_path: str, image_root: str, image_size: int = 384,
                 augmentation: bool = False):
        self.image_root = image_root
        self.transform = _build_train_transform(image_size, augmentation) if augmentation else _build_eval_transform(image_size)

        self.samples: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_real = sum(1 for s in self.samples if s["label"] == 0)
        n_fake = len(self.samples) - n_real
        logger.info(
            "Loaded %d samples from %s (real=%d, fake=%d)",
            len(self.samples), metadata_path, n_real, n_fake,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img_path = os.path.join(self.image_root, row["image"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        label = int(row["label"])
        return pixel_values, label


def build_dataloaders(cfg: dict):
    image_root = os.path.join(cfg["data_root"], "frames")
    image_size = int(cfg.get("image_size", 384))
    augmentation = bool(cfg.get("augmentation", False))

    train_ds = FFClassificationDataset(
        os.path.join(cfg["data_root"], "train.jsonl"), image_root, image_size,
        augmentation=augmentation,
    )
    val_ds = FFClassificationDataset(
        os.path.join(cfg["data_root"], "val.jsonl"), image_root, image_size,
        augmentation=False,
    )

    max_train = cfg.get("max_train_samples")
    if max_train and max_train < len(train_ds):
        train_ds.samples = train_ds.samples[:max_train]
        logger.info("Sub-sampled training set to %d samples", max_train)

    if augmentation:
        logger.info("Classifier augmentation enabled (geometric + color)")

    # Balanced sampling for training
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    tp = fp = tn = fn = 0

    for pixels, labels in tqdm(loader, desc="Evaluating", leave=False):
        pixels = pixels.to(device)
        labels = labels.to(device)
        logits = model(pixels)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": total_loss / max(total, 1),
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 classification head")
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

    model = DINOv2Classifier(
        dino_model=cfg.get("dinov2_model", "facebook/dinov2-large"),
        num_classes=int(cfg.get("num_classes", 2)),
        hidden_dim=int(cfg.get("head_hidden_dim", 256)),
        dropout=float(cfg.get("head_dropout", 0.1)),
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.head.load_state_dict(ckpt["head"])
        logger.info("Resumed head from %s", args.resume)

    model.to(device)

    head_params = list(model.head.parameters())
    total_params = sum(p.numel() for p in head_params)
    logger.info("Trainable head parameters: %d (~%.1fK)", total_params, total_params / 1e3)

    optimizer = AdamW(
        head_params,
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = build_dataloaders(cfg)

    max_epochs = int(cfg.get("max_epochs", 15))
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(max_epochs):
        model.train()
        model.dinov2.eval()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for pixels, labels in pbar:
            pixels = pixels.to(device)
            labels = labels.to(device)

            logits = model(pixels)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            epoch_loss += loss.item() * bs
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += bs
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{epoch_correct / max(epoch_total, 1):.3f}",
            )

        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch, train_loss, train_acc,
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"],
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "head": model.head.state_dict(),
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
            logger.info("  -> New best val_acc=%.4f (saved best.pt)", best_val_acc)

        torch.save(
            {
                "epoch": epoch,
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
            },
            output_dir / "last.pt",
        )

    logger.info(
        "Training complete. Best val_acc=%.4f at epoch %d",
        best_val_acc, best_epoch,
    )

    # Final evaluation summary
    results = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": max_epochs,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
