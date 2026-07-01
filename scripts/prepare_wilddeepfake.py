#!/usr/bin/env python3
"""Download WildDeepfake from HuggingFace and prepare test.jsonl for evaluation.

The dataset (xingjunm/WildDeepfake) is stored in webdataset format (tar.gz).
Images are 224x224 PNG face crops. Keys encode the label via path structure:
``./video_id/fake_or_real/subfolder/frame_id``.

This script:
  1. Downloads the test split via HuggingFace datasets (streaming or full).
  2. Saves individual face images to ``<output_dir>/frames/``.
  3. Writes ``test.jsonl`` compatible with DDVQADataset.

Usage::

    python scripts/prepare_wilddeepfake.py \
        --output-dir /datasets/deepfake/wilddeepfake_prepared \
        --max-samples 0

    # Quick test with a subset:
    python scripts/prepare_wilddeepfake.py \
        --output-dir /datasets/deepfake/wilddeepfake_prepared \
        --max-samples 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("prepare_wilddeepfake")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare WildDeepfake for cross-dataset eval")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for frames/ and test.jsonl")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to download (0 = all, useful for debugging)")
    p.add_argument("--question", default="Is this image real or fake?",
                   help="Question text for the VQA prompt")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip frames that already exist on disk")
    p.add_argument("--dataset-name", default="xingjunm/WildDeepfake",
                   help="HuggingFace dataset identifier")
    return p.parse_args()


def parse_label_from_key(key: str) -> bool | None:
    """Determine if a sample is real or fake from its __key__ field.

    Keys follow the pattern: ./video_id/real_or_fake/subfolder/frame_id
    e.g. ./1/fake/131/1057 -> fake
         ./3/real/45/200   -> real
    """
    parts = key.replace("\\", "/").split("/")
    for part in parts:
        if part.lower() == "fake":
            return False
        if part.lower() == "real":
            return True
    return None


def extract_video_id(key: str) -> str:
    """Extract a video-level identifier from the key for aggregation.

    Key format: ./video_id/label/subfolder/frame_id
    Video ID = video_id/label/subfolder (groups all frames from same source).
    """
    parts = key.replace("\\", "/").strip("./").split("/")
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return key


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    logger.info("Loading dataset %s (test split)...", args.dataset_name)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset(args.dataset_name, split="test", trust_remote_code=True)
    total = len(ds)
    logger.info("Test split has %d samples", total)

    if args.max_samples > 0:
        total = min(total, args.max_samples)
        logger.info("Limiting to %d samples", total)

    all_samples = []
    saved = 0
    skipped = 0

    for idx in tqdm(range(total), desc="Processing WildDeepfake test"):
        sample = ds[idx]

        key = sample.get("__key__", f"sample_{idx:06d}")
        image = sample.get("png") or sample.get("image") or sample.get("jpg")

        if image is None:
            logger.warning("No image found for sample %d (key=%s)", idx, key)
            continue

        is_real = parse_label_from_key(key)
        if is_real is None:
            logger.warning("Cannot determine label for key: %s", key)
            continue

        safe_name = key.replace("/", "_").replace(".", "").strip("_") + ".png"
        frame_path = os.path.join(frames_dir, safe_name)

        if args.skip_existing and os.path.isfile(frame_path):
            skipped += 1
        else:
            image.save(frame_path)
            saved += 1

        video_id = extract_video_id(key)

        all_samples.append({
            "image": safe_name,
            "question": args.question,
            "answer": "Real." if is_real else "Fake.",
            "is_real": is_real,
            "method": "wilddeepfake",
            "video_id": video_id,
            "frame_idx": idx,
            "split": "test",
        })

    metadata_path = os.path.join(output_dir, "test.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_real = sum(1 for s in all_samples if s["is_real"])
    n_fake = len(all_samples) - n_real
    logger.info(
        "Done! Wrote %d samples to %s (real=%d, fake=%d). "
        "Saved %d images, skipped %d existing.",
        len(all_samples), metadata_path, n_real, n_fake, saved, skipped,
    )


if __name__ == "__main__":
    main()
