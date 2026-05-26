#!/usr/bin/env python3
"""Prepare Celeb-DF-v2 test set for cross-dataset evaluation.

Reads the official test split from ``List_of_testing_videos.txt``, extracts
N evenly-spaced frames per video, crops the largest face via Haar cascade,
and writes a ``test.jsonl`` metadata file compatible with DDVQADataset.

Usage::

    python scripts/prepare_celebdf.py \
        --celebdf-root /datasets/deepfake/Celeb-DF-v2 \
        --output-dir /datasets/deepfake/celebdf_prepared \
        --num-frames 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("prepare_celebdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Celeb-DF-v2 for cross-dataset eval")
    p.add_argument("--celebdf-root", required=True,
                   help="Root of the extracted Celeb-DF-v2 dataset")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for frames/ and test.jsonl")
    p.add_argument("--num-frames", type=int, default=32,
                   help="Number of frames to extract per video (default: 32)")
    p.add_argument("--face-margin", type=float, default=0.3,
                   help="Margin around detected face (fraction of bbox size)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip videos whose frames already exist")
    p.add_argument("--question", default="Does the image look real/fake?",
                   help="Question text for the VQA prompt")
    return p.parse_args()


def extract_frames(video_path: str, num_frames: int = 1) -> list[tuple[int, any]]:
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
    """Crop face using Haar cascade. Falls back to full frame if no face found."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

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


def parse_test_list(celebdf_root: str) -> list[dict]:
    """Parse List_of_testing_videos.txt into structured entries.

    Each line: ``{label} {folder}/{filename}.mp4``
    label=1 means real, label=0 means fake.
    """
    list_path = os.path.join(celebdf_root, "List_of_testing_videos.txt")
    if not os.path.isfile(list_path):
        logger.error("Test list not found: %s", list_path)
        sys.exit(1)

    entries = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            label_int = int(parts[0])
            rel_path = parts[1].strip()

            folder = os.path.dirname(rel_path)
            filename = os.path.splitext(os.path.basename(rel_path))[0]
            is_real = label_int == 1

            entries.append({
                "rel_path": rel_path,
                "folder": folder,
                "filename": filename,
                "is_real": is_real,
            })

    return entries


def main() -> None:
    args = parse_args()
    celebdf_root = args.celebdf_root
    output_dir = args.output_dir
    num_frames = args.num_frames

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    entries = parse_test_list(celebdf_root)
    logger.info(
        "Parsed %d test videos (real=%d, fake=%d)",
        len(entries),
        sum(1 for e in entries if e["is_real"]),
        sum(1 for e in entries if not e["is_real"]),
    )

    all_samples = []
    skipped_videos = 0
    failed_videos = 0

    for entry in tqdm(entries, desc="Processing videos"):
        video_path = os.path.join(celebdf_root, entry["rel_path"])
        if not os.path.isfile(video_path):
            logger.warning("Video not found: %s", video_path)
            failed_videos += 1
            continue

        folder = entry["folder"].replace("/", "_").replace("\\", "_")
        filename = entry["filename"]
        video_id = filename
        base_name = f"{folder}_{filename}"

        if args.skip_existing:
            first_frame = os.path.join(frames_dir, f"{base_name}_f00.jpg")
            if os.path.isfile(first_frame):
                skipped_videos += 1
                for fi in range(num_frames):
                    frame_name = f"{base_name}_f{fi:02d}.jpg"
                    if os.path.isfile(os.path.join(frames_dir, frame_name)):
                        all_samples.append({
                            "image": frame_name,
                            "question": args.question,
                            "answer": "Real." if entry["is_real"] else "Fake.",
                            "is_real": entry["is_real"],
                            "method": entry["folder"],
                            "video_id": video_id,
                            "frame_idx": fi,
                            "split": "test",
                        })
                continue

        extracted = extract_frames(video_path, num_frames)
        if not extracted:
            logger.warning("Could not extract frames from: %s", video_path)
            failed_videos += 1
            continue

        for fi, (frame_idx, frame) in enumerate(extracted):
            face_crop = crop_face_simple(frame, margin=args.face_margin)
            frame_name = f"{base_name}_f{fi:02d}.jpg"
            dst_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(dst_path, face_crop)

            all_samples.append({
                "image": frame_name,
                "question": args.question,
                "answer": "Real." if entry["is_real"] else "Fake.",
                "is_real": entry["is_real"],
                "method": entry["folder"],
                "video_id": video_id,
                "frame_idx": fi,
                "split": "test",
            })

    logger.info(
        "Generated %d frame samples from %d videos "
        "(skipped=%d existing, failed=%d)",
        len(all_samples),
        len(entries),
        skipped_videos,
        failed_videos,
    )

    out_path = os.path.join(output_dir, "test.jsonl")
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    logger.info("Metadata written to: %s", out_path)

    n_real = sum(1 for s in all_samples if s["is_real"])
    n_fake = len(all_samples) - n_real
    logger.info("  Total frames: %d (real=%d, fake=%d)", len(all_samples), n_real, n_fake)
    logger.info("  Frames dir: %s", frames_dir)


if __name__ == "__main__":
    main()
