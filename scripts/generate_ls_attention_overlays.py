#!/usr/bin/env python3
"""Generate DINOv2 attention overlays for Label Studio DD-VQA frames.

Computes last-layer CLS→patch attention (avg heads) with the LoRA deepfake
classifier, blends a jet heatmap onto the original frame, and writes JPEGs::

    <output-dir>/<same_basename>.jpg

Usage on tarkin (GPU)::

    python scripts/generate_ls_attention_overlays.py \\
        --frames-dir /path/to/frames \\
        --output-dir /path/to/attention_out \\
        --checkpoint outputs/dino_lora_classifier/best.pt \\
        --batch-size 16

Use --max-images 4 for a smoke test. Re-runs skip existing files by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from models.dino_lora_classifier import DINOv2LoRAClassifier  # noqa: E402
from extract_attention_maps import extract_attention  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Map [0,1] float array to RGB uint8 using a simple jet-like LUT."""
    try:
        import matplotlib.cm as cm

        rgba = cm.jet(np.clip(x, 0.0, 1.0))
        return (rgba[..., :3] * 255).astype(np.uint8)
    except Exception:
        # Fallback without matplotlib: blue → cyan → yellow → red
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(1.5 - abs(4.0 * x - 3.0), 0, 1)
        g = np.clip(1.5 - abs(4.0 * x - 2.0), 0, 1)
        b = np.clip(1.5 - abs(4.0 * x - 1.0), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def attn_to_overlay(
    rgb: np.ndarray,
    attn_flat: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Resize CLS→patch attention to image size and blend with jet colormap."""
    h, w = rgb.shape[:2]
    n = int(attn_flat.shape[0])
    side = int(round(n ** 0.5))
    if side * side != n:
        # pad/crop to nearest square
        side = int(np.floor(n ** 0.5))
        attn_flat = attn_flat[: side * side]
    grid = attn_flat.reshape(side, side).astype(np.float32)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heat = np.array(
        Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    color = jet_colormap(heat).astype(np.float32)
    base = rgb.astype(np.float32)
    blend = (1.0 - alpha) * base + alpha * color
    return np.clip(blend, 0, 255).astype(np.uint8)


def load_model(checkpoint: str, device: torch.device) -> DINOv2LoRAClassifier:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
    return model


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frames-dir",
        default=str(repo / "label_studio" / "data" / "frames"),
    )
    p.add_argument(
        "--output-dir",
        default=str(repo / "label_studio" / "data" / "attention"),
    )
    p.add_argument(
        "--checkpoint",
        default=str(repo / "outputs" / "dino_lora_classifier" / "best.pt"),
    )
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skip_existing = args.skip_existing and not args.no_skip_existing
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if skip_existing:
        pending = [p for p in paths if not (out_dir / p.name).is_file()]
        print(f"Frames: {len(paths)} total, {len(pending)} pending, "
              f"{len(paths) - len(pending)} already done")
        paths = pending
    else:
        print(f"Frames: {len(paths)}")

    if not paths:
        print("Nothing to do.")
        return

    print("Loading DINOv2 LoRA classifier...")
    model = load_model(args.checkpoint, device)
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    bs = max(1, args.batch_size)
    if device.type == "cpu":
        bs = min(bs, 2)

    for i in tqdm(range(0, len(paths), bs), desc="Attention overlays"):
        batch_paths = paths[i : i + bs]
        rgbs = []
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            rgbs.append(np.array(img))
            tensors.append(transform(img))
        pixel_values = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            attn = extract_attention(model, pixel_values)
        for p, rgb, a in zip(batch_paths, rgbs, attn):
            overlay = attn_to_overlay(rgb, a, alpha=args.alpha)
            Image.fromarray(overlay).save(out_dir / p.name, quality=92)

    print(f"Wrote overlays -> {out_dir}")


if __name__ == "__main__":
    main()
