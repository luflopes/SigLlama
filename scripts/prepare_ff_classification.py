#!/usr/bin/env python3
"""Prepare the FaceForensics++ dataset for DINOv2 binary classification.

Extracts one face-cropped frame per video for every method (Original +
manipulations) and generates a JSONL metadata file with train/val/test
splits that respect the official FF++ split lists.

Usage::

    python scripts/prepare_ff_classification.py \
        --ff-root /datasets/deepfake/faceforensics \
        --output-dir /datasets/deepfake/ff_classification \
        --compression c23

The ``--ddvqa-test-ids`` flag accepts a path to a newline-delimited file
of video IDs.  Any video present in that file is **excluded** from the
classifier's train/val sets to prevent leakage into DD-VQA evaluation.

Output layout::

    <output-dir>/
      frames/Original_000.jpg
      frames/Deepfakes_135_880.jpg
      ...
      train.jsonl
      val.jsonl
      test.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("prepare_ff_cls")

METHODS_FAKE = [
    ("Deepfakes", "manipulated_sequences/Deepfakes"),
    ("Face2Face", "manipulated_sequences/Face2Face"),
    ("FaceSwap", "manipulated_sequences/FaceSwap"),
    ("NeuralTextures", "manipulated_sequences/NeuralTextures"),
]

METHODS_REAL = [
    ("Original", "original_sequences/youtube"),
]

ALL_METHODS = METHODS_REAL + METHODS_FAKE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare FF++ for DINOv2 classification")
    p.add_argument("--ff-root", required=True, help="FaceForensics++ root dir")
    p.add_argument("--output-dir", required=True, help="Output dir for frames + metadata")
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    p.add_argument(
        "--ddvqa-test-ids", default=None,
        help="Path to file with DD-VQA test video IDs (one per line) to exclude from train/val",
    )
    p.add_argument("--num-frames", type=int, default=30,
                    help="Number of frames to extract per video (evenly spaced)")
    p.add_argument("--face-margin", type=float, default=0.3)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def extract_frames(video_path: str, num_frames: int = 30) -> list[tuple[int, any]]:
    """Extract evenly-spaced frames from a video.

    Returns a list of (frame_index, BGR numpy array) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(num_frames, total)
    if n == 1:
        indices = [total // 2]
    else:
        step = max(total / (n + 1), 1)
        indices = [int(step * (i + 1)) for i in range(n)]
        indices = [min(idx, total - 1) for idx in indices]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def crop_face_simple(frame, margin: float = 0.3):
    """Crop largest face using Haar cascade. Falls back to full frame."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    areas = [w * h for (_, _, w, h) in faces]
    best = faces[areas.index(max(areas))]
    x, y, w, h = best
    H, W = frame.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    return frame[y1:y2, x1:x2]


def load_ff_splits(ff_root: str) -> dict[str, set[str]]:
    """Load official FF++ train/val/test splits (JSON lists of video-ID pairs).

    Returns a dict mapping split name -> set of individual video IDs.
    """
    split_dir = os.path.join(ff_root, "splits")
    mapping: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    split_files = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }
    for split_name, fname in split_files.items():
        fpath = os.path.join(split_dir, fname)
        if not os.path.isfile(fpath):
            logger.warning("Split file not found: %s", fpath)
            continue
        with open(fpath) as f:
            pairs = json.load(f)
        for pair in pairs:
            for vid in pair:
                mapping[split_name].add(str(vid))
    for s, ids in mapping.items():
        logger.info("FF++ split '%s': %d video IDs", s, len(ids))
    return mapping


def video_id_to_split(vid: str, ff_splits: dict[str, set[str]]) -> str:
    """Map a video ID to its split; defaults to 'train' if not found."""
    for split_name in ("test", "val", "train"):
        if vid in ff_splits[split_name]:
            return split_name
    return "train"


def main() -> None:
    args = parse_args()
    ff_root = args.ff_root
    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    ff_splits = load_ff_splits(ff_root)

    ddvqa_test_ids: set[str] = set()
    if args.ddvqa_test_ids:
        with open(args.ddvqa_test_ids) as f:
            for line in f:
                line = line.strip()
                if line:
                    ddvqa_test_ids.add(line)
        logger.info("Loaded %d DD-VQA test video IDs to exclude from train/val", len(ddvqa_test_ids))

    all_samples: list[dict] = []
    total_skipped = 0

    for method_name, method_subdir in ALL_METHODS:
        is_real = method_name == "Original"
        label = 0 if is_real else 1

        video_dir = os.path.join(ff_root, method_subdir, args.compression, "videos")
        if not os.path.isdir(video_dir):
            logger.warning("Video dir not found: %s", video_dir)
            continue

        videos = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        ])
        logger.info("Processing %s: %d videos in %s", method_name, len(videos), video_dir)

        for vfile in tqdm(videos, desc=method_name, leave=False):
            video_id = os.path.splitext(vfile)[0]
            base_vid = video_id.split("_")[0]
            split = video_id_to_split(base_vid, ff_splits)

            # Check if frames for this video already exist (skip_existing)
            first_frame_name = f"{method_name}_{video_id}_f00.jpg"
            if args.skip_existing and os.path.isfile(os.path.join(frames_dir, first_frame_name)):
                for fi in range(args.num_frames):
                    fname = f"{method_name}_{video_id}_f{fi:02d}.jpg"
                    if os.path.isfile(os.path.join(frames_dir, fname)):
                        all_samples.append({
                            "image": fname,
                            "label": label,
                            "is_real": is_real,
                            "method": method_name,
                            "video_id": video_id,
                            "split": split,
                        })
                continue

            vpath = os.path.join(video_dir, vfile)
            extracted = extract_frames(vpath, num_frames=args.num_frames)
            if not extracted:
                total_skipped += 1
                continue

            for fi, (frame_idx, frame) in enumerate(extracted):
                frame_name = f"{method_name}_{video_id}_f{fi:02d}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                face = crop_face_simple(frame, margin=args.face_margin)
                cv2.imwrite(frame_path, face)

                all_samples.append({
                    "image": frame_name,
                    "label": label,
                    "is_real": is_real,
                    "method": method_name,
                    "video_id": video_id,
                    "split": split,
                })

    logger.info(
        "Extracted %d samples total (skipped %d videos)",
        len(all_samples), total_skipped,
    )

    # Enforce DD-VQA test leakage prevention
    if ddvqa_test_ids:
        for s in all_samples:
            base_vid = s["video_id"].split("_")[0]
            if base_vid in ddvqa_test_ids and s["split"] != "test":
                s["split"] = "test"

    for split_name in ("train", "val", "test"):
        split_samples = [s for s in all_samples if s["split"] == split_name]
        n_real = sum(1 for s in split_samples if s["is_real"])
        n_fake = len(split_samples) - n_real
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(json.dumps(s) + "\n")
        logger.info(
            "  %s: %d samples (real=%d, fake=%d) -> %s",
            split_name, len(split_samples), n_real, n_fake, out_path,
        )

    logger.info("Done. Frames saved to: %s", frames_dir)


if __name__ == "__main__":
    main()
