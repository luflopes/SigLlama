"""Extract MediaPipe face landmarks from all DD-VQA frames.

Produces a JSONL file mapping each frame path to its 478 normalized
landmarks.  Used by ``scripts/create_loc_annotations.py`` to build
<loc>-enriched annotations.

Usage::

    python scripts/extract_landmarks.py \
        --frames-dir /datasets/deepfake/ddvqa_prepared/frames \
        --output /datasets/deepfake/ddvqa_prepared/landmarks.jsonl \
        --max-faces 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.extractors.landmark_extractor import LandmarkExtractor  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True, help="Directory with DD-VQA frame images")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--min-confidence", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)

    image_paths = sorted(
        p for p in frames_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    logger.info("Found %d images in %s", len(image_paths), frames_dir)

    extractor = LandmarkExtractor(
        max_num_faces=args.max_faces,
        min_detection_confidence=args.min_confidence,
    )

    success = 0
    failed = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        for i, img_path in enumerate(image_paths):
            result = extractor.extract(str(img_path))
            if result is None:
                failed += 1
                continue

            rel_path = str(img_path.relative_to(frames_dir))
            record = {
                "image": rel_path,
                "landmarks": result["normalized"],
                "confidence": result["confidence"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1

            if (i + 1) % 5000 == 0:
                logger.info("Processed %d / %d (success=%d, failed=%d)", i + 1, len(image_paths), success, failed)

    extractor.close()
    logger.info("Done. success=%d, failed=%d, output=%s", success, failed, args.output)


if __name__ == "__main__":
    main()
