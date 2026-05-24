"""Extract DINOv2 attention maps (CLS → patches) for visualisation.

Loads the DINOv2 LoRA classifier and, for a set of selected images,
extracts the attention weights from the last transformer layer
(CLS token attending to all patch tokens). These serve as a proxy
for GradCAM-like saliency — showing which image regions the
deepfake-aware backbone focuses on.

Usage (on the GPU server)::

    python scripts/extract_attention_maps.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --images-dir /datasets/deepfake/ddvqa_prepared/frames \
        --predictions outputs/ablation/g4_lora_loc/evaluation/best_test/predictions.jsonl \
        --output outputs/analysis/attention_maps.npz \
        --n-samples 50

Output ``.npz`` keys:
    images      : uint8  [N, H, W, 3]  – resized RGB images
    attn_maps   : float32 [N, P]       – CLS→patch attention (last layer, avg heads)
    labels      : str [N]
    filenames   : str [N]
    predictions : str [N]  – generated text from predictions file
    references  : str [N]  – reference text from predictions file
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dino_lora_classifier import DINOv2LoRAClassifier

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


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

    # Deduplicate by image (keep first question per image)
    seen = set()
    unique = []
    for r in records:
        img = r["image"]
        if img not in seen:
            seen.add(img)
            unique.append(r)

    rng = np.random.RandomState(seed)
    methods = {}
    for r in unique:
        m = r.get("method", "unknown")
        methods.setdefault(m, []).append(r)

    sampled = []
    per_method = max(1, n_samples // len(methods))
    for method, recs in methods.items():
        rng.shuffle(recs)
        sampled.extend(recs[:per_method])

    rng.shuffle(sampled)
    return sampled[:n_samples]


def extract_attention(
    model: DINOv2LoRAClassifier,
    pixel_values: torch.Tensor,
) -> np.ndarray:
    """Extract last-layer CLS→patch attention, averaged across heads.

    Uses a forward pre-hook to capture the hidden states entering the last
    self-attention module during the real forward pass (respects PEFT/LoRA
    and SDPA). Q and K are then computed from the captured input.

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


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 attention maps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--predictions", required=True,
                        help="predictions.jsonl from G4 evaluation")
    parser.add_argument("--output", default="outputs/analysis/attention_maps.npz")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=384)
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

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    samples = load_predictions_sample(args.predictions, args.n_samples)
    print(f"Processing {len(samples)} images...")

    all_images = []
    all_attn = []
    all_labels = []
    all_filenames = []
    all_predictions = []
    all_references = []

    for sample in tqdm(samples, desc="Attention maps"):
        img_path = os.path.join(args.images_dir, sample["image"])
        if not os.path.exists(img_path):
            print(f"  skip (not found): {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((args.image_size, args.image_size))
        img_array = np.array(img_resized)

        pixel_values = transform(img).unsqueeze(0).to(device)
        attn = extract_attention(model, pixel_values)

        all_images.append(img_array)
        all_attn.append(attn[0])
        all_labels.append(sample.get("true_label", "unknown"))
        all_filenames.append(sample["image"])
        all_predictions.append(sample.get("generated", ""))
        all_references.append(sample.get("reference_answer", ""))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        images=np.array(all_images),
        attn_maps=np.array(all_attn),
        labels=np.array(all_labels),
        filenames=np.array(all_filenames),
        predictions=np.array(all_predictions),
        references=np.array(all_references),
    )
    print(f"Saved: {args.output}  ({len(all_images)} images)")


if __name__ == "__main__":
    main()
