"""Enrich DD-VQA annotations with textual ``[y1,x1,y2,x2]`` bounding boxes.

Reads:
  - Original DD-VQA JSONL (train.jsonl / val.jsonl)
  - Landmarks JSONL (from scripts/extract_landmarks.py)

Writes:
  - Enriched JSONL (train_loc.jsonl / val_loc.jsonl) where answers
    include ``[y1,x1,y2,x2]`` bounding boxes (integer coords in [0, 1000])
    grounding each explanation to a facial region.

This textual format works consistently for both PaliGemma2 and TinyLLaVA
backbones without requiring additional tokens in the vocabulary.

Usage::

    python scripts/create_loc_annotations.py \
        --ddvqa-jsonl /datasets/deepfake/ddvqa_prepared/train.jsonl \
        --landmarks-jsonl /datasets/deepfake/ddvqa_prepared/landmarks.jsonl \
        --output /datasets/deepfake/ddvqa_prepared/train_loc.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── MediaPipe 478 landmark → semantic facial region mapping ──
FACIAL_REGIONS: dict[str, list[int]] = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "nose": [1, 2, 3, 4, 5, 6, 195, 197, 45, 275, 440, 220],
    "mouth": [
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 95,
        146, 178, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317,
        318, 324, 375, 402, 405,
    ],
    "jawline": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
    "left_cheek": [116, 117, 118, 119, 120, 121, 128, 36, 205, 206, 207],
    "right_cheek": [345, 346, 347, 348, 349, 350, 357, 266, 425, 426, 427],
    "forehead": [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 68, 104, 69, 108],
}

# Which regions are most affected per manipulation technique
TECHNIQUE_REGIONS: dict[str, list[str]] = {
    "Deepfakes": ["jawline", "left_cheek", "right_cheek", "forehead"],
    "FaceSwap": ["jawline", "left_cheek", "right_cheek", "forehead"],
    "Face2Face": ["mouth", "left_eye", "right_eye"],
    "NeuralTextures": ["nose", "left_cheek", "right_cheek", "mouth"],
}

TECHNIQUE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Deepfakes": {
        "jawline": "shows blending artifacts along the face boundary",
        "left_cheek": "displays unnatural skin texture from face replacement",
        "right_cheek": "displays unnatural skin texture from face replacement",
        "forehead": "reveals identity swap artifacts in the forehead region",
    },
    "FaceSwap": {
        "jawline": "shows visible boundary artifacts from face swap",
        "left_cheek": "presents color inconsistency from face swap",
        "right_cheek": "presents color inconsistency from face swap",
        "forehead": "reveals identity replacement artifacts",
    },
    "Face2Face": {
        "mouth": "shows unnatural expression artifacts from reenactment",
        "left_eye": "displays gaze inconsistencies typical of face reenactment",
        "right_eye": "displays gaze inconsistencies typical of face reenactment",
    },
    "NeuralTextures": {
        "nose": "reveals neural rendering artifacts in surface texture",
        "left_cheek": "shows synthetic texture patterns",
        "right_cheek": "shows synthetic texture patterns",
        "mouth": "displays rendering inconsistencies",
    },
}

REAL_DESCRIPTIONS: dict[str, str] = {
    "left_eye": "appears natural with consistent lighting",
    "right_eye": "appears natural with consistent lighting",
    "nose": "shows natural skin texture without artifacts",
    "mouth": "displays natural expression without anomalies",
    "jawline": "has a smooth, natural boundary without blending artifacts",
    "left_cheek": "shows consistent skin texture",
    "right_cheek": "shows consistent skin texture",
}


BBOX_COORD_SCALE = 1000


def landmarks_to_bbox_text(
    landmarks: list[list[float]],
    region_indices: list[int],
) -> str | None:
    """Convert landmark indices to a textual bounding box ``[y1,x1,y2,x2]``.

    Coordinates are integers in ``[0, BBOX_COORD_SCALE]`` so the format is
    backbone-agnostic: both PaliGemma2 and TinyLlama can read/emit it as
    plain tokens without extending the vocabulary.
    """
    valid = [i for i in region_indices if i < len(landmarks)]
    if not valid:
        return None

    pts = [landmarks[i] for i in valid]
    x_min = min(p[0] for p in pts)
    y_min = min(p[1] for p in pts)
    x_max = max(p[0] for p in pts)
    y_max = max(p[1] for p in pts)

    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    ly1 = int(round(y_min * BBOX_COORD_SCALE))
    lx1 = int(round(x_min * BBOX_COORD_SCALE))
    ly2 = int(round(y_max * BBOX_COORD_SCALE))
    lx2 = int(round(x_max * BBOX_COORD_SCALE))

    return f"[{ly1},{lx1},{ly2},{lx2}]"


def build_enriched_answer_fake(
    method: str,
    landmarks: list[list[float]],
    original_answer: str,
) -> str:
    """Build a bbox-enriched explanation for a fake image."""
    regions = TECHNIQUE_REGIONS.get(method, ["mouth", "jawline"])
    descs = TECHNIQUE_DESCRIPTIONS.get(method, {})

    parts = ["This image is fake."]
    for region_name in regions:
        indices = FACIAL_REGIONS.get(region_name)
        if indices is None:
            continue
        bbox_str = landmarks_to_bbox_text(landmarks, indices)
        if bbox_str is None:
            continue
        desc = descs.get(region_name, "shows manipulation artifacts")
        parts.append(
            f"The {region_name.replace('_', ' ')} region {bbox_str} {desc}."
        )

    return " ".join(parts)


def build_enriched_answer_real(
    landmarks: list[list[float]],
    original_answer: str,
) -> str:
    """Build a bbox-enriched description for a real image."""
    inspect_regions = ["left_eye", "mouth", "jawline"]

    parts = ["This image is real."]
    for region_name in inspect_regions:
        indices = FACIAL_REGIONS.get(region_name)
        if indices is None:
            continue
        bbox_str = landmarks_to_bbox_text(landmarks, indices)
        if bbox_str is None:
            continue
        desc = REAL_DESCRIPTIONS.get(region_name, "appears natural")
        parts.append(
            f"The {region_name.replace('_', ' ')} region {bbox_str} {desc}."
        )

    return " ".join(parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ddvqa-jsonl", required=True, help="Original DD-VQA JSONL")
    p.add_argument("--landmarks-jsonl", required=True, help="Landmarks JSONL from extract_landmarks.py")
    p.add_argument("--output", required=True, help="Output enriched JSONL")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading landmarks from %s", args.landmarks_jsonl)
    landmarks_db: dict[str, list[list[float]]] = {}
    with open(args.landmarks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            landmarks_db[rec["image"]] = rec["landmarks"]
    logger.info("Loaded landmarks for %d images", len(landmarks_db))

    logger.info("Processing %s", args.ddvqa_jsonl)
    enriched = 0
    skipped = 0
    total = 0

    with open(args.ddvqa_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line.strip())
            total += 1

            img_rel = row.get("image", "")
            landmarks = landmarks_db.get(img_rel)

            if landmarks is None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            label = row.get("label", "fake").lower()
            method = row.get("method", "unknown")
            original_answer = row.get("answer", "")

            if label == "real":
                new_answer = build_enriched_answer_real(landmarks, original_answer)
            else:
                new_answer = build_enriched_answer_fake(method, landmarks, original_answer)

            row["answer"] = new_answer
            row["answer_original"] = original_answer
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            enriched += 1

    logger.info(
        "Done. total=%d, enriched=%d, skipped=%d (no landmarks). Output: %s",
        total, enriched, skipped, args.output,
    )


if __name__ == "__main__":
    main()
