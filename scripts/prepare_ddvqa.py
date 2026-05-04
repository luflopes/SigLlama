#!/usr/bin/env python3
"""Prepare the DD-VQA dataset for training.

Two modes of operation:

**Mode A – Original author frames (recommended)**:
  Uses the exact face-cropped frames provided by the DD-VQA paper authors,
  guaranteeing pixel-perfect alignment with the human annotations.

  Usage::

      python scripts/prepare_ddvqa.py \\
          --ddvqa-root /datasets/deepfake/Research-DD-VQA \\
          --output-dir /datasets/deepfake/ddvqa_prepared \\
          --original-frames /path/to/ddvqa_frames

**Mode B – Extract from FF++ videos (legacy)**:
  Extracts the middle frame from each FF++ video and crops the largest
  face via Haar cascade.  Frame may NOT match what annotators saw.

  Usage::

      python scripts/prepare_ddvqa.py \\
          --ff-root /datasets/deepfake/faceforensics \\
          --ddvqa-root /datasets/deepfake/Research-DD-VQA \\
          --output-dir /datasets/deepfake/ddvqa_prepared \\
          --compression c23

Expected DD-VQA layout (clone of Research-DD-VQA)::

    <ddvqa_root>/
      DQ_FF++/total.json
      DQ_FF++/split/train.json
      DQ_FF++/split/val.json
      DQ_FF++/split/test.json

Original author frames layout::

    <original-frames>/
      0_135_880.jpg      # {manip_id}_{video_id}.jpg
      5_000.jpg
      ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

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
    p.add_argument("--ddvqa-root", required=True, help="Research-DD-VQA repo root")
    p.add_argument("--output-dir", required=True, help="Output dir for frames + metadata")
    p.add_argument("--original-frames", default=None,
                    help="Path to author-provided frames ({manip_id}_{video_id}.jpg). "
                         "When set, frames are copied from here instead of extracting from FF++ videos.")
    p.add_argument("--ff-root", default=None, help="FaceForensics++ root dir (only needed without --original-frames)")
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    p.add_argument("--num-frames", type=int, default=1,
                    help="Frames to extract per video (sampled evenly)")
    p.add_argument("--face-margin", type=float, default=0.3,
                    help="Margin around detected face (fraction of bbox size)")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose frames already exist")
    args = p.parse_args()
    if args.original_frames is None and args.ff_root is None:
        p.error("Either --original-frames or --ff-root must be provided")
    return args


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


def _provision_original_frame(
    ddvqa_key: str, frames_dir: str, original_frames_dir: str
) -> str | None:
    """Copy an author-provided frame into the output frames dir.

    Author frames are named ``{ddvqa_key}.jpg`` (e.g. ``0_135_880.jpg``).
    We rename to ``{Method}_{video_id}.jpg`` for consistency with the rest
    of the pipeline.
    """
    manip_id, video_id = parse_ddvqa_key(ddvqa_key)
    method = MANIP_ID_TO_METHOD.get(manip_id)
    if method is None:
        return None

    src = os.path.join(original_frames_dir, f"{ddvqa_key}.jpg")
    if not os.path.isfile(src):
        return None

    dst_name = f"{method}_{video_id}.jpg"
    dst = os.path.join(frames_dir, dst_name)
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
    return dst_name


def _provision_extracted_frame(
    ddvqa_key: str,
    frames_dir: str,
    ff_root: str,
    compression: str,
    num_frames: int,
    face_margin: float,
) -> str | None:
    """Extract a frame from the FF++ video (legacy mode)."""
    import cv2  # lazy import – only needed in legacy mode

    manip_id, video_id = parse_ddvqa_key(ddvqa_key)
    method = MANIP_ID_TO_METHOD.get(manip_id)
    if method is None:
        return None

    dst_name = f"{method}_{video_id}.jpg"
    dst = os.path.join(frames_dir, dst_name)
    if os.path.isfile(dst):
        return dst_name

    video_path = resolve_video_path(ff_root, method, video_id, compression)
    if video_path is None:
        return None

    extracted = extract_frames(video_path, num_frames)
    if not extracted:
        return None

    _, frame = extracted[0]
    face_crop = crop_face_simple(frame, margin=face_margin)
    cv2.imwrite(dst, face_crop)
    return dst_name


def main() -> None:
    args = parse_args()
    ddvqa_root = args.ddvqa_root
    output_dir = args.output_dir
    use_original = args.original_frames is not None

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if use_original:
        logger.info("Using author-provided frames from: %s", args.original_frames)
    else:
        logger.info("Extracting frames from FF++ videos at: %s", args.ff_root)

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

        if use_original:
            frame_filename = _provision_original_frame(
                ddvqa_key, frames_dir, args.original_frames
            )
        else:
            frame_filename = _provision_extracted_frame(
                ddvqa_key, frames_dir, args.ff_root,
                args.compression, args.num_frames, args.face_margin,
            )

        if frame_filename is None:
            skipped += 1
            continue

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
