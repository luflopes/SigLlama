#!/usr/bin/env python3
"""Prepare the FaceForensics++ dataset for DINOv2 binary classification.

Extracts face-cropped frames per video for every method (Original +
manipulations) and generates JSONL metadata files with train/val/test
splits that respect the official FF++ split lists.

Usage::

    python scripts/prepare_ff_classification.py \
        --ff-root /datasets/deepfake/faceforensics \
        --output-dir /datasets/deepfake/ff_classification \
        --compression c23 \
        --splits-dir /datasets/deepfake/Research-DD-VQA/DQ_FF++/split

The ``--splits-dir`` flag points to a directory containing
train.json / val.json / test.json — the FF++ official splits stored as
lists of [video_id, video_id] pairs.  Using the **same** split directory
as DD-VQA (``DQ_FF++/split/``) guarantees that the classifier's
train/val/test partition is identical to the one used by the VQA dataset,
preventing any information leakage.

Output layout::

    <output-dir>/
      frames/Original_000_f00.jpg
      frames/Deepfakes_135_880_f00.jpg
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
    ("FaceShifter", "manipulated_sequences/FaceShifter"),
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
        "--splits-dir", default=None,
        help=(
            "Directory containing train.json/val.json/test.json split files. "
            "These are the FF++ official splits (lists of video-ID pairs). "
            "Use the same dir as DD-VQA to guarantee no leakage: "
            "e.g. /datasets/deepfake/Research-DD-VQA/DQ_FF++/split. "
            "Falls back to <ff-root>/splits/ if not provided."
        ),
    )
    p.add_argument("--num-frames", type=int, default=30,
                    help="Number of frames to extract per video (evenly spaced)")
    p.add_argument("--face-margin", type=float, default=0.3)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--methods", nargs="*", default=None,
        help=(
            "Only process these methods (e.g. --methods FaceShifter). "
            "When set, uses --append mode: extracts frames for the "
            "specified methods and merges into existing JSONL files."
        ),
    )
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


def load_ff_splits(splits_dir: str) -> dict[str, set[str]]:
    """Load FF++ train/val/test splits (JSON lists of video-ID pairs).

    The split files are typically lists of pairs: [[id1, id2], [id3, id4], ...]
    Each individual ID is added to the corresponding split set.

    Returns a dict mapping split name -> set of individual video IDs.
    """
    mapping: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    split_files = {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }
    for split_name, fname in split_files.items():
        fpath = os.path.join(splits_dir, fname)
        if not os.path.isfile(fpath):
            logger.warning("Split file not found: %s", fpath)
            continue
        with open(fpath) as f:
            pairs = json.load(f)
        for pair in pairs:
            if isinstance(pair, list):
                for vid in pair:
                    mapping[split_name].add(str(vid))
            else:
                mapping[split_name].add(str(pair))
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

    # Determine splits directory
    splits_dir = args.splits_dir
    if splits_dir is None:
        splits_dir = os.path.join(ff_root, "splits")
    if not os.path.isdir(splits_dir):
        logger.error(
            "Splits directory not found: %s. "
            "Please provide --splits-dir pointing to the directory with "
            "train.json/val.json/test.json (e.g. the DD-VQA DQ_FF++/split dir).",
            splits_dir,
        )
        sys.exit(1)

    ff_splits = load_ff_splits(splits_dir)

    total_ids = sum(len(v) for v in ff_splits.values())
    if total_ids == 0:
        logger.error(
            "No video IDs loaded from splits. Check the contents of %s", splits_dir
        )
        sys.exit(1)

    # Filter methods if requested
    if args.methods:
        valid_names = {name for name, _ in ALL_METHODS}
        for m in args.methods:
            if m not in valid_names:
                logger.error("Unknown method '%s'. Valid: %s", m, sorted(valid_names))
                sys.exit(1)
        methods_to_process = [(n, p) for n, p in ALL_METHODS if n in args.methods]
        logger.info("Processing only: %s (append mode)", [n for n, _ in methods_to_process])
    else:
        methods_to_process = ALL_METHODS

    # In append mode, load existing samples first
    all_samples: list[dict] = []
    if args.methods:
        for split_name in ("train", "val", "test"):
            jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
            if os.path.isfile(jsonl_path):
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            sample = json.loads(line)
                            if sample.get("method") not in args.methods:
                                all_samples.append(sample)
                logger.info(
                    "Loaded existing %s (kept %d samples, excluding %s)",
                    jsonl_path,
                    sum(1 for s in all_samples if s.get("split") == split_name),
                    args.methods,
                )

    total_skipped = 0

    for method_name, method_subdir in methods_to_process:
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
            # For fake videos like "135_880", both IDs may be in splits.
            # For originals like "000", just use the ID directly.
            # We check each part of underscore-split IDs against the splits.
            parts = video_id.split("_")
            split = None
            for part in parts:
                s = video_id_to_split(part, ff_splits)
                if s != "train" or part in ff_splits["train"]:
                    split = s
                    break
            if split is None:
                split = "train"

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
