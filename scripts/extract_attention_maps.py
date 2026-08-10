"""Extract DINOv2 attention maps (CLS → patches) for visualisation.

Loads the DINOv2 LoRA classifier and extracts last-layer CLS→patch
attention (average over heads). Optionally writes jet overlays for
Label Studio.

Usage (sample from predictions, as before)::

    python scripts/extract_attention_maps.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --images-dir /datasets/deepfake/ddvqa_prepared/frames \
        --predictions outputs/ablation/g4_lora_loc/evaluation/best_test/predictions.jsonl \
        --output outputs/analysis/attention_maps.npz \
        --n-samples 300

Usage (all frames + Label Studio overlays on tarkin)::

    python scripts/extract_attention_maps.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --images-dir /datasets/deepfake/ddvqa_prepared/frames \
        --all-frames \
        --overlay-dir outputs/analysis/overlays/ \
        --batch-size 16

Output ``.npz`` keys (when ``--output`` is set):
    images      : uint8  [N, H, W, 3]  – resized RGB (omitted if --no-store-images)
    attn_maps   : float32 [N, P]
    labels      : str [N]
    filenames   : str [N]
    predictions : str [N]
    references  : str [N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dino_lora_classifier import DINOv2LoRAClassifier

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_predictions_sample(
    predictions_path: str,
    n_samples: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Load predictions JSONL and sample diverse examples."""
    records = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    seen = set()
    unique = []
    for r in records:
        img = r["image"]
        if img not in seen:
            seen.add(img)
            unique.append(r)

    rng = np.random.RandomState(seed)
    methods: dict[str, list] = {}
    for r in unique:
        m = r.get("method", "unknown")
        methods.setdefault(m, []).append(r)

    sampled = []
    per_method = max(1, n_samples // max(len(methods), 1))
    for _method, recs in methods.items():
        rng.shuffle(recs)
        sampled.extend(recs[:per_method])

    rng.shuffle(sampled)
    return sampled[:n_samples]


def list_all_frame_samples(images_dir: str, max_images: int | None = None) -> list[dict]:
    """One synthetic sample dict per image file in ``images_dir``."""
    root = Path(images_dir)
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if max_images is not None:
        paths = paths[:max_images]
    return [{"image": p.name, "true_label": "unknown", "generated": "", "reference_answer": ""} for p in paths]


def extract_attention(
    model: DINOv2LoRAClassifier,
    pixel_values: torch.Tensor,
) -> np.ndarray:
    """Extract last-layer CLS→patch attention, averaged across heads.

    Returns shape [B, num_patches].
    """
    model.eval()

    base_model = model.dinov2.base_model.model
    last_self_attn = base_model.encoder.layer[-1].attention.attention

    captured = {}

    def _pre_hook(module, args):
        captured["hidden"] = args[0].detach()

    handle = last_self_attn.register_forward_pre_hook(_pre_hook)

    with torch.no_grad():
        model.dinov2(pixel_values=pixel_values)

    handle.remove()

    hidden = captured["hidden"]

    with torch.no_grad():
        q = last_self_attn.query(hidden)
        k = last_self_attn.key(hidden)

    head_dim = q.shape[-1] // last_self_attn.num_attention_heads
    num_heads = last_self_attn.num_attention_heads
    batch_size = q.size(0)

    q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    cls_attn = attn_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]
    avg_attn = cls_attn.mean(dim=1)        # [B, num_patches]
    return avg_attn.cpu().numpy()


def jet_colormap(x: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm

        rgba = cm.jet(np.clip(x, 0.0, 1.0))
        return (rgba[..., :3] * 255).astype(np.uint8)
    except Exception:
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
    """Blend jet heatmap of CLS→patch attention onto an RGB image."""
    h, w = rgb.shape[:2]
    n = int(attn_flat.shape[0])
    side = int(round(n ** 0.5))
    if side * side != n:
        side = int(np.floor(n ** 0.5))
        attn_flat = attn_flat[: side * side]
    grid = attn_flat.reshape(side, side).astype(np.float32)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heat = np.array(
        Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    color = jet_colormap(heat).astype(np.float32)
    blend = (1.0 - alpha) * rgb.astype(np.float32) + alpha * color
    return np.clip(blend, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 attention maps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument(
        "--predictions",
        default=None,
        help="predictions.jsonl (sample mode). Omit when using --all-frames.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process every image in --images-dir (for Label Studio overlays).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .npz output (default: outputs/analysis/attention_maps.npz "
             "in sample mode; omitted in --all-frames unless set).",
    )
    parser.add_argument(
        "--overlay-dir",
        default="outputs/analysis/overlays/",
        help="Write jet overlays as JPEG with the same basename as each frame.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--skip-existing-overlays",
        action="store_true",
        default=True,
        help="Skip frames whose overlay JPEG already exists (default: on).",
    )
    parser.add_argument("--no-skip-existing-overlays", action="store_true")
    parser.add_argument(
        "--no-store-images",
        action="store_true",
        help="Do not store resized RGB arrays in the npz (saves a lot of disk).",
    )
    args = parser.parse_args()

    if not args.all_frames and not args.predictions:
        parser.error("Provide --predictions (sample mode) or --all-frames")
    if args.all_frames and not args.overlay_dir and not args.output:
        parser.error("With --all-frames, set --overlay-dir and/or --output")

    skip_existing = args.skip_existing_overlays and not args.no_skip_existing_overlays
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

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    if args.all_frames:
        samples = list_all_frame_samples(args.images_dir, args.max_images)
    else:
        samples = load_predictions_sample(args.predictions, args.n_samples)
        if args.max_images is not None:
            samples = samples[: args.max_images]

    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
        if skip_existing:
            before = len(samples)
            samples = [
                s for s in samples
                if not os.path.isfile(os.path.join(args.overlay_dir, os.path.basename(s["image"])))
            ]
            print(f"Overlays: {before - len(samples)} already exist, {len(samples)} pending")

    print(f"Processing {len(samples)} images...")
    if not samples:
        print("Nothing to do.")
        return

    all_images = []
    all_attn = []
    all_labels = []
    all_filenames = []
    all_predictions = []
    all_references = []

    store_npz = args.output is not None or not args.all_frames
    if store_npz and args.output is None:
        args.output = "outputs/analysis/attention_maps.npz"

    bs = max(1, args.batch_size)
    if device.type == "cpu":
        bs = min(bs, 2)

    for i in tqdm(range(0, len(samples), bs), desc="Attention maps"):
        batch = samples[i : i + bs]
        tensors = []
        originals = []  # RGB at native resolution for overlays
        keep = []
        for sample in batch:
            img_path = os.path.join(args.images_dir, sample["image"])
            if not os.path.exists(img_path):
                print(f"  skip (not found): {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            originals.append(np.array(img))
            tensors.append(transform(img))
            keep.append(sample)
        if not tensors:
            continue

        pixel_values = torch.stack(tensors, dim=0).to(device)
        attn = extract_attention(model, pixel_values)

        for sample, rgb, a in zip(keep, originals, attn):
            fname = os.path.basename(sample["image"])
            if args.overlay_dir:
                overlay = attn_to_overlay(rgb, a, alpha=args.overlay_alpha)
                Image.fromarray(overlay).save(
                    os.path.join(args.overlay_dir, fname),
                    quality=92,
                )
            if store_npz:
                img_resized = np.array(
                    Image.fromarray(rgb).resize((args.image_size, args.image_size))
                )
                if not args.no_store_images:
                    all_images.append(img_resized)
                all_attn.append(a)
                all_labels.append(sample.get("true_label", "unknown"))
                all_filenames.append(fname)
                all_predictions.append(sample.get("generated", ""))
                all_references.append(sample.get("reference_answer", ""))

    if store_npz and args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "attn_maps": np.array(all_attn),
            "labels": np.array(all_labels),
            "filenames": np.array(all_filenames),
            "predictions": np.array(all_predictions),
            "references": np.array(all_references),
        }
        if not args.no_store_images:
            payload["images"] = np.array(all_images)
        np.savez_compressed(args.output, **payload)
        print(f"Saved: {args.output}  ({len(all_filenames)} images)")

    if args.overlay_dir:
        n_out = len([
            p for p in Path(args.overlay_dir).iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        ])
        print(f"Overlays in {args.overlay_dir}: {n_out} files")


if __name__ == "__main__":
    main()
