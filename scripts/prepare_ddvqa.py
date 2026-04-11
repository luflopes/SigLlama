#!/usr/bin/env python3
"""Prepare the DD-VQA dataset from FaceForensics++ videos.

Pipeline:
  1. Walk FF++ video directories for each manipulation method
  2. Extract the middle frame from each video
  3. Detect and crop the largest face (via YOLO or simple heuristic)
  4. Merge with DD-VQA annotation JSONs
  5. Write a flat JSONL metadata ready for training

Expected FF++ layout (after download with c23 compression)::

    <ff_root>/
      original_sequences/youtube/c23/videos/000.mp4 ...
      manipulated_sequences/Deepfakes/c23/videos/000_003.mp4 ...
      manipulated_sequences/Face2Face/c23/videos/...
      manipulated_sequences/FaceSwap/c23/videos/...
      manipulated_sequences/FaceShifter/c23/videos/...
      manipulated_sequences/NeuralTextures/c23/videos/...

Expected DD-VQA layout (clone of Research-DD-VQA)::

    <ddvqa_root>/
      DQ_FF++/total.json
      DQ_FF++/split/train.json
      DQ_FF++/split/val.json
      DQ_FF++/split/test.json

Usage::

    python scripts/prepare_ddvqa.py \\
        --ff-root /datasets/deepfake/faceforensics \\
        --ddvqa-root /datasets/deepfake/Research-DD-VQA \\
        --output-dir /datasets/deepfake/ddvqa_prepared \\
        --compression c23
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
logger = logging.getLogger("prepare_ddvqa")

MANIP_ID_TO_METHOD = {
    "0": "Deepfakes",
    "1": "Face2Face",
    "2": "FaceShifter",
    "3": "FaceSwap",
    "5": "Original",
    "6": "NeuralTextures",
}

METHOD_TO_FF_PATH = {
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
    "Original": "original_sequences/youtube",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DD-VQA dataset")
    p.add_argument("--ff-root", required=True, help="FaceForensics++ root dir")
    p.add_argument("--ddvqa-root", required=True, help="Research-DD-VQA repo root")
    p.add_argument("--output-dir", required=True, help="Output dir for frames + metadata")
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    p.add_argument("--num-frames", type=int, default=1,
                    help="Frames to extract per video (sampled evenly)")
    p.add_argument("--face-margin", type=float, default=0.3,
                    help="Margin around detected face (fraction of bbox size)")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose frames already exist")
    return p.parse_args()


def extract_frames(video_path: str, num_frames: int = 1) -> list:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    if num_frames == 1:
        indices = [total // 2]
    else:
        step = max(total // (num_frames + 1), 1)
        indices = [step * (i + 1) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def crop_face_simple(frame, margin: float = 0.3):
    """Crop face using OpenCV's Haar cascade (no GPU dependency).

    Falls back to returning the full frame if no face is found.
    """
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
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    return frame[y1:y2, x1:x2]


def resolve_video_path(
    ff_root: str, method: str, video_id: str, compression: str
) -> str | None:
    """Find the FF++ video file for a given method and video ID."""
    ff_subdir = METHOD_TO_FF_PATH.get(method)
    if not ff_subdir:
        return None

    video_dir = os.path.join(ff_root, ff_subdir, compression, "videos")
    video_file = os.path.join(video_dir, f"{video_id}.mp4")
    if os.path.isfile(video_file):
        return video_file
    return None


def parse_ddvqa_key(key: str) -> tuple[str, str]:
    """Parse DD-VQA key like '0_135_880' into (manip_id, video_id).

    Original keys look like '5_000' (manip_id=5, video_id=000).
    Fake keys look like '0_135_880' (manip_id=0, video_id=135_880).
    """
    parts = key.split("_", 1)
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    ff_root = args.ff_root
    ddvqa_root = args.ddvqa_root
    output_dir = args.output_dir

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Load DD-VQA annotations
    annot_path = os.path.join(ddvqa_root, "DQ_FF++", "total.json")
    if not os.path.isfile(annot_path):
        logger.error("DD-VQA annotations not found: %s", annot_path)
        sys.exit(1)

    with open(annot_path, "r") as f:
        annotations = json.load(f)
    logger.info("Loaded %d annotation entries from DD-VQA", len(annotations))

    # Load splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(ddvqa_root, "DQ_FF++", "split", f"{split_name}.json")
        if os.path.isfile(split_path):
            with open(split_path, "r") as f:
                splits[split_name] = json.load(f)
            logger.info("Split '%s': %d video pairs", split_name, len(splits[split_name]))

    # Build a set of video IDs per split for quick lookup
    split_video_ids: dict[str, set[str]] = {}
    for split_name, pairs in splits.items():
        ids = set()
        for pair in pairs:
            ids.add("_".join(pair))
            ids.add("_".join(pair[::-1]))
            for vid in pair:
                ids.add(vid)
        split_video_ids[split_name] = ids

    def get_split(video_id: str) -> str:
        for split_name, ids in split_video_ids.items():
            if video_id in ids:
                return split_name
        return "train"

    # Process each annotation entry
    all_samples = []
    skipped = 0

    for ddvqa_key, questions in tqdm(annotations.items(), desc="Processing"):
        manip_id, video_id = parse_ddvqa_key(ddvqa_key)
        method = MANIP_ID_TO_METHOD.get(manip_id)
        if method is None:
            skipped += 1
            continue

        frame_filename = f"{method}_{video_id}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)

        # Extract frame if needed
        if not os.path.isfile(frame_path) or not args.skip_existing:
            video_path = resolve_video_path(ff_root, method, video_id, args.compression)
            if video_path is None:
                skipped += 1
                continue

            extracted = extract_frames(video_path, args.num_frames)
            if not extracted:
                skipped += 1
                continue

            _, frame = extracted[0]
            face_crop = crop_face_simple(frame, margin=args.face_margin)
            cv2.imwrite(frame_path, face_crop)

        is_real = method == "Original"
        split = get_split(video_id)

        for q_id, qa in questions.items():
            question = qa["question"]
            answers = qa["answer"]
            answer = answers[0]

            all_samples.append({
                "image": frame_filename,
                "question": question,
                "answer": answer,
                "method": method,
                "is_real": is_real,
                "video_id": video_id,
                "split": split,
                "ddvqa_key": ddvqa_key,
            })

    logger.info("Generated %d QA samples (skipped %d entries)", len(all_samples), skipped)

    # Write per-split JSONL files
    for split_name in ["train", "val", "test"]:
        split_samples = [s for s in all_samples if s["split"] == split_name]
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(json.dumps(s) + "\n")
        logger.info("  %s: %d samples -> %s", split_name, len(split_samples), out_path)

    # Also write a combined file
    combined_path = os.path.join(output_dir, "all.jsonl")
    with open(combined_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    logger.info("  all: %d samples -> %s", len(all_samples), combined_path)

    logger.info("Done. Frames saved to: %s", frames_dir)


if __name__ == "__main__":
    main()
