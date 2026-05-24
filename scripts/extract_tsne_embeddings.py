"""Extract DINOv2 CLS-token embeddings for t-SNE visualisation.

Produces two sets of embeddings (with LoRA / frozen) so the notebook can
compare how the LoRA-finetuned backbone separates real vs fake in the
embedding space.

Usage (on the GPU server)::

    python scripts/extract_tsne_embeddings.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --data-root /datasets/deepfake/ff_classification \
        --split val \
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


class SimpleFFDataset(Dataset):
    """Minimal FF++ dataset that returns pixel_values, label, method."""

    def __init__(self, metadata_path: str, image_root: str, image_size: int = 384):
        self.image_root = image_root
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img_path = os.path.join(self.image_root, row["image"])
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        binary_label = int(row["label"])
        method = row.get("method", "unknown")
        return pixel_values, binary_label, method


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
    parser.add_argument("--data-root", default="/datasets/deepfake/ff_classification",
                        help="FF++ classification data root")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Which split to use")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/analysis/tsne_embeddings.npz")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = DINOv2LoRAClassifier(use_moe=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device)
    model.eval()

    metadata_path = os.path.join(args.data_root, f"{args.split}.jsonl")
    image_root = os.path.join(args.data_root, "frames")
    dataset = SimpleFFDataset(metadata_path, image_root, args.image_size)

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
