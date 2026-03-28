#!/usr/bin/env python3
"""CLI for extracting soft tokens (YOLO detections + MediaPipe face landmarks)
from the LCS-558K dataset (or any dataset with the same metadata format).

Supports:
  - Configurable paths for metadata, images, YOLO weights, and output
  - Automatic resume: skips image IDs already present in the output NDJSON
  - Optional annotated-image visualisation

Usage examples:

    # Full LCS-558K extraction on GPU server
    python scripts/extract_soft_tokens.py \
        --metadata /data/lcs-558k/blip_laion_cc_sbu_558k_meta.json \
        --image-root /data/lcs-558k \
        --output /data/lcs-558k/lcs_softtokens.ndjson \
        --yolo-model yolov8l.pt

    # Quick test on local samples
    python scripts/extract_soft_tokens.py \
        --metadata sample/metadata.json \
        --image-root sample \
        --output sample_softtokens.ndjson \
        --visualize --vis-dir annotated
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.extractors import SoftTokenExtractor
from data.visualization import plot_and_save
from data.writers import NDJSONWriter


def fix_seed(seed: int = 0) -> None:
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract soft tokens (detections + landmarks) from images."
    )
    p.add_argument(
        "--metadata", required=True,
        help="Path to the JSON metadata file (list of {id, image, blip_caption}).",
    )
    p.add_argument(
        "--image-root", required=True,
        help="Root directory that image paths in metadata are relative to.",
    )
    p.add_argument(
        "--output", default="lcs_softtokens.ndjson",
        help="Output NDJSON path (default: lcs_softtokens.ndjson).",
    )
    p.add_argument(
        "--yolo-model", default="yolov8l.pt",
        help="Path to YOLOv8 weights (default: yolov8l.pt).",
    )
    p.add_argument(
        "--yolo-conf", type=float, default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0).",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume (overwrite output instead of appending).",
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Save annotated images with detections and landmarks.",
    )
    p.add_argument(
        "--vis-dir", default="annotated",
        help="Directory for annotated images (default: annotated).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)

    with open(args.metadata, "r", encoding="utf-8") as f:
        samples = json.load(f)

    processed_ids: set[str] = set()
    if not args.no_resume:
        processed_ids = NDJSONWriter.load_processed_ids(args.output)
        if processed_ids:
            print(f"[resume] {len(processed_ids)} imagens já processadas, pulando.")

    if args.no_resume and os.path.isfile(args.output):
        os.remove(args.output)

    extractor = SoftTokenExtractor(
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
    )

    with NDJSONWriter(args.output) as writer:
        for sample in tqdm(samples, desc="Extracting soft tokens"):
            image_id = sample["id"]
            if image_id in processed_ids:
                continue

            image_path = os.path.join(args.image_root, sample["image"])
            if not os.path.isfile(image_path):
                tqdm.write(f"[WARN] Imagem não encontrada: {image_path}")
                continue

            caption = sample.get("blip_caption", "")
            entry = extractor.process_image(image_id, image_path, caption)
            writer.write(entry)

            if args.visualize:
                plot_and_save(
                    image_id=image_id,
                    image_path=image_path,
                    detections=entry["detection_tokens"],
                    landmarks=entry["landmark_tokens"],
                    class_names=extractor.detector.class_names,
                    out_dir=args.vis_dir,
                )

    print(f"Extração concluída. Saída em: {args.output}")


if __name__ == "__main__":
    main()
