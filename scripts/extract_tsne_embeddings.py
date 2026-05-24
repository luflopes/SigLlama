"""Extract DINOv2 CLS-token embeddings for t-SNE visualisation.

Produces two sets of embeddings (with LoRA / frozen) so the notebook can
compare how the LoRA-finetuned backbone separates real vs fake in the
embedding space.

Supports two data formats:

1. **FF++ classification** (``--format ff``)::

       python scripts/extract_tsne_embeddings.py \\
           --checkpoint outputs/dino_lora_classifier/best.pt \\
           --metadata /datasets/deepfake/ff_classification/val.jsonl \\
           --images-dir /datasets/deepfake/ff_classification/frames \\
           --format ff --output outputs/analysis/tsne_embeddings.npz

2. **DD-VQA** (``--format ddvqa``, default) — uses the test split and
   deduplicates by image so each frame appears once::

       python scripts/extract_tsne_embeddings.py \\
           --checkpoint outputs/dino_lora_classifier/best.pt \\
           --metadata /datasets/deepfake/ddvqa_prepared/test.jsonl \\
           --images-dir /datasets/deepfake/ddvqa_prepared/frames \\
           --output outputs/analysis/tsne_embeddings.npz

The output ``.npz`` contains:
    embeddings_lora   : float32 [N, D]   – CLS tokens with LoRA active
    embeddings_frozen : float32 [N, D]   – CLS tokens with LoRA disabled
    labels            : str     [N]      – "real" / "fake"
    methods           : str     [N]      – manipulation method name
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dino_lora_classifier import DINOv2LoRAClassifier, METHOD_TO_IDX

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

IDX_TO_METHOD = {v: k for k, v in METHOD_TO_IDX.items()}


def _load_samples(metadata_path: str, fmt: str) -> list[dict]:
    """Load JSONL and return normalised dicts with keys: image, label, method.

    For DD-VQA format, deduplicates by image name (each image has multiple
    QA pairs but we only need one embedding per image).
    """
    raw = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    if fmt == "ff":
        return [
            {"image": r["image"], "label": int(r["label"]), "method": r.get("method", "unknown")}
            for r in raw
        ]

    # DD-VQA: deduplicate by image, derive label from is_real / label field
    seen: set[str] = set()
    samples: list[dict] = []
    for r in raw:
        img = r["image"]
        if img in seen:
            continue
        seen.add(img)

        if "is_real" in r:
            label = 0 if r["is_real"] else 1
        elif "label" in r:
            lbl = str(r["label"]).lower()
            label = 0 if lbl in ("real", "0", "true") else 1
        else:
            label = -1

        method = r.get("method", "unknown")
        samples.append({"image": img, "label": label, "method": method})
    return samples


class ImageListDataset(Dataset):
    """Lightweight dataset over a list of {image, label, method} dicts."""

    def __init__(self, samples: list[dict], image_root: str, image_size: int = 384):
        self.samples = samples
        self.image_root = image_root
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img_path = os.path.join(self.image_root, row["image"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        return pixel_values, int(row["label"]), row["method"]


def extract_embeddings(
    model: DINOv2LoRAClassifier,
    loader: DataLoader,
    device: torch.device,
    use_lora: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Forward-pass through the dataset collecting CLS tokens."""
    all_emb = []
    all_labels = []
    all_methods = []

    if not use_lora:
        model.dinov2.disable_adapter_layers()
    else:
        model.dinov2.enable_adapter_layers()

    model.eval()
    with torch.no_grad():
        for pixel_values, binary_labels, methods in tqdm(loader, desc="Extracting"):
            pixel_values = pixel_values.to(device)
            out = model.dinov2(pixel_values=pixel_values)
            cls_tokens = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(cls_tokens)

            for bl, m in zip(binary_labels.tolist(), methods):
                all_labels.append("real" if bl == 0 else "fake")
                all_methods.append(m)

    if not use_lora:
        model.dinov2.enable_adapter_layers()

    return np.concatenate(all_emb, axis=0), all_labels, all_methods


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings for t-SNE")
    parser.add_argument("--checkpoint", required=True, help="Path to DINOv2 LoRA classifier .pt")
    parser.add_argument("--metadata", required=True,
                        help="Path to JSONL metadata (e.g. test.jsonl)")
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing the images referenced in metadata")
    parser.add_argument("--format", default="ddvqa", choices=["ddvqa", "ff"],
                        help="Metadata format: ddvqa (default) or ff (FF++ classification)")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/analysis/tsne_embeddings.npz")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    ckpt_cfg = ckpt.get("config", {})
    model = DINOv2LoRAClassifier(
        dino_model=ckpt_cfg.get("dinov2_model", "facebook/dinov2-large"),
        lora_rank=ckpt_cfg.get("lora_rank", 16),
        lora_alpha=ckpt_cfg.get("lora_alpha", 32),
        use_moe=ckpt_cfg.get("use_moe", False),
    )

    from peft import set_peft_model_state_dict
    if "lora" in ckpt:
        set_peft_model_state_dict(model.dinov2, ckpt["lora"])
    model.binary_head.load_state_dict(ckpt["binary_head"])
    model.forgery_head.load_state_dict(ckpt["forgery_head"])

    model = model.to(device)
    model.eval()

    samples = _load_samples(args.metadata, args.format)
    print(f"Loaded {len(samples)} unique images from {args.metadata} (format={args.format})")
    dataset = ImageListDataset(samples, args.images_dir, args.image_size)

    if args.max_samples and args.max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Extracting embeddings (LoRA active) from {len(dataset)} samples...")
    emb_lora, labels, methods = extract_embeddings(model, loader, device, use_lora=True)

    print(f"Extracting embeddings (LoRA disabled / frozen)...")
    emb_frozen, _, _ = extract_embeddings(model, loader, device, use_lora=False)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings_lora=emb_lora,
        embeddings_frozen=emb_frozen,
        labels=np.array(labels),
        methods=np.array(methods),
    )
    print(f"Saved: {args.output}  (shape: {emb_lora.shape})")


if __name__ == "__main__":
    main()
