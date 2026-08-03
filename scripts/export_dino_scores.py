"""Export per-frame binary scores from the DINOv2 LoRA classifier.

Runs the trained DINOv2 LoRA classifier over an FF++ classification split
(or any JSONL with the FF++ schema) and writes, for every frame, the
softmax probability of the *fake* class. These scores feed the threshold
analysis in ``notebooks/analyze_evaluation.ipynb``.

The binary head outputs 2 logits (index 0 = Real, 1 = Fake). We export
``score = softmax(logits)[:, 1]`` (probability of fake, in [0, 1]) so the
optimal cut-off can be tuned on the validation split and applied to test.

Usage (run on the server where the checkpoint and datasets live)::

    # Validation split (for threshold tuning)
    python scripts/export_dino_scores.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --metadata /datasets/deepfake/ff_classification/val.jsonl \
        --images-dir /datasets/deepfake/ff_classification/frames \
        --output outputs/dino_lora_classifier/scores_val.jsonl

    # Test split (for final evaluation at the tuned threshold)
    python scripts/export_dino_scores.py \
        --checkpoint outputs/dino_lora_classifier/best.pt \
        --metadata /datasets/deepfake/ff_classification/test.jsonl \
        --images-dir /datasets/deepfake/ff_classification/frames \
        --output outputs/dino_lora_classifier/scores_test.jsonl

Each output row::

    {"image": "Deepfakes_135_880_f00.jpg", "true_label": "fake",
     "score": 0.9873, "logit_real": -2.1, "logit_fake": 2.2,
     "method": "Deepfakes", "video_id": "135_880"}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dino_lora_classifier import DINOv2LoRAClassifier, METHOD_TO_IDX  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

IDX_TO_METHOD = {v: k for k, v in METHOD_TO_IDX.items()}

# Strip a trailing frame suffix like _f00.jpg to recover the source video id.
_FRAME_SUFFIX_RE = re.compile(r"_f\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def _derive_video_id(row: dict) -> str:
    """Return video_id from the row, or derive it from the image name."""
    vid = row.get("video_id")
    if vid:
        return str(vid)
    image = row.get("image", "")
    return _FRAME_SUFFIX_RE.sub("", image)


def _row_label(row: dict) -> int:
    """Return 0 (real) / 1 (fake) from an FF++/DD-VQA style row."""
    if "label" in row and row["label"] is not None:
        try:
            return int(row["label"])
        except (TypeError, ValueError):
            pass
    if "is_real" in row:
        return 0 if bool(row["is_real"]) else 1
    method = row.get("method", "")
    return 0 if method == "Original" else 1


def load_samples(metadata_path: str, dedup_by_image: bool = False) -> list[dict]:
    """Load JSONL rows, normalising to {image, label, method, video_id}.

    When ``dedup_by_image`` is set (DD-VQA, where each frame has multiple
    QA rows), only the first occurrence of each image is kept so every
    frame is scored exactly once.
    """
    samples: list[dict] = []
    seen: set[str] = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            image = r["image"]
            if dedup_by_image:
                if image in seen:
                    continue
                seen.add(image)
            samples.append({
                "image": image,
                "label": _row_label(r),
                "method": r.get("method", "unknown"),
                "video_id": _derive_video_id(r),
            })
    return samples


class FrameDataset(Dataset):
    """Dataset over {image, label, method, video_id} dicts (no augmentation)."""

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
        return self.transform(img), idx


def load_classifier(checkpoint_path: str, device: torch.device) -> DINOv2LoRAClassifier:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_cfg = ckpt.get("config", {})
    model = DINOv2LoRAClassifier(
        dino_model=ckpt_cfg.get("dinov2_model", "facebook/dinov2-large"),
        lora_rank=int(ckpt_cfg.get("lora_rank", 16)),
        lora_alpha=int(ckpt_cfg.get("lora_alpha", 32)),
        lora_target_modules=ckpt_cfg.get("lora_target_modules", ["query", "value"]),
        use_moe=bool(ckpt_cfg.get("use_moe", False)),
        num_experts=int(ckpt_cfg.get("num_experts", 6)),
    )

    from peft import set_peft_model_state_dict
    if "lora" in ckpt:
        set_peft_model_state_dict(model.dinov2, ckpt["lora"])
    model.binary_head.load_state_dict(ckpt["binary_head"])
    model.forgery_head.load_state_dict(ckpt["forgery_head"])
    if model.use_moe and "expert_lora_params" in ckpt:
        model.expert_lora_params.load_state_dict(ckpt["expert_lora_params"])
    if model.use_moe and "router" in ckpt:
        model.router.load_state_dict(ckpt["router"])

    val_acc = ckpt.get("val_metrics", {}).get("binary_accuracy", 0.0)
    print(
        f"Loaded DINOv2 LoRA classifier from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, val_bin_acc={val_acc:.4f}, "
        f"moe={model.use_moe})"
    )

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DINOv2 per-frame fake scores")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DINOv2 LoRA classifier .pt")
    parser.add_argument("--metadata", required=True,
                        help="Path to FF++ classification JSONL (val.jsonl / test.jsonl)")
    parser.add_argument("--images-dir", required=True,
                        help="Directory with the frames referenced in metadata")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for per-frame scores")
    parser.add_argument("--format", default="ff", choices=["ff", "ddvqa"],
                        help="Metadata format. 'ddvqa' deduplicates by image "
                             "(one score per frame instead of per QA row).")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_classifier(args.checkpoint, device)

    samples = load_samples(args.metadata, dedup_by_image=(args.format == "ddvqa"))
    print(
        f"Loaded {len(samples)} frames from {args.metadata} "
        f"(format={args.format})"
    )
    dataset = FrameDataset(samples, args.images_dir, args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    results: list[dict] = []
    with torch.no_grad():
        for pixel_values, indices in tqdm(loader, desc="Scoring"):
            pixel_values = pixel_values.to(device)
            out = model(pixel_values=pixel_values)
            logits = out["binary_logits"].float()
            probs = F.softmax(logits, dim=-1)
            fake_probs = probs[:, 1].cpu().tolist()
            logits_cpu = logits.cpu().tolist()

            for j, idx in enumerate(indices.tolist()):
                row = samples[idx]
                results.append({
                    "image": row["image"],
                    "true_label": "real" if row["label"] == 0 else "fake",
                    "score": round(float(fake_probs[j]), 6),
                    "logit_real": round(float(logits_cpu[j][0]), 6),
                    "logit_fake": round(float(logits_cpu[j][1]), 6),
                    "method": row["method"],
                    "video_id": row["video_id"],
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_fake = sum(1 for r in results if r["true_label"] == "fake")
    n_real = len(results) - n_fake
    print(
        f"Saved {len(results)} scores to {args.output} "
        f"(real={n_real}, fake={n_fake})"
    )


if __name__ == "__main__":
    main()
