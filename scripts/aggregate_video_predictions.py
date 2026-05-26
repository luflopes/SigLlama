#!/usr/bin/env python3
"""Aggregate frame-level predictions into video-level metrics.

Reads ``predictions.jsonl`` produced by ``evaluation/evaluate.py`` and
computes metrics at two granularities:

- **Frame-level**: each frame is an independent sample (Acc, F1, AUC).
- **Video-level**: frames are grouped by video_id via majority voting,
  with AUC computed using the proportion of fake-predicted frames as a
  continuous score per video.

Usage::

    python scripts/aggregate_video_predictions.py \
        --predictions outputs/cross_dataset/celebdf_g3_lora/evaluation/predictions.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("aggregate_predictions")

# Matches frame suffix like _f00, _f01, _f31
_FRAME_SUFFIX_RE = re.compile(r"_f\d+\.jpg$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate predictions to video-level")
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.jsonl from evaluate.py")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (defaults to same dir as predictions)")
    return p.parse_args()


def extract_video_id(image_name: str) -> str:
    """Extract video identifier from frame filename.

    Filenames follow the pattern: {folder}_{videoname}_f{NN}.jpg
    We strip the _fNN.jpg suffix to get a unique video identifier.
    """
    return _FRAME_SUFFIX_RE.sub("", image_name)


def compute_metrics(true_labels: list[str], pred_labels: list[str],
                    scores: list[float] | None = None) -> dict:
    """Compute detection metrics (binary: real=0, fake=1)."""
    y_true = np.array([1 if l == "fake" else 0 for l in true_labels])
    y_pred = np.array([1 if l == "fake" else 0 for l in pred_labels])

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }

    if scores is not None and len(set(y_true)) > 1:
        try:
            results["auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            results["auc"] = None
    else:
        if scores is not None:
            results["auc"] = None
            results["auc_note"] = "Only one class present, AUC undefined"

    return results


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions

    if not os.path.isfile(predictions_path):
        logger.error("Predictions file not found: %s", predictions_path)
        return

    output_dir = args.output_dir or os.path.dirname(predictions_path)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    logger.info("Loaded %d frame predictions from %s", len(rows), predictions_path)

    # --- Frame-level metrics ---
    frame_true = [r["true_label"] for r in rows]
    frame_pred = [r["pred_label"] for r in rows]

    # Use verdict_score if available, otherwise binary prediction
    frame_scores = None
    if "verdict_score" in rows[0]:
        frame_scores = [float(r["verdict_score"]) for r in rows]
    else:
        frame_scores = [1.0 if r["pred_label"] == "fake" else 0.0 for r in rows]

    frame_metrics = compute_metrics(frame_true, frame_pred, frame_scores)
    frame_metrics["level"] = "frame"
    logger.info("=== Frame-level metrics ===")
    for k, v in frame_metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # --- Video-level aggregation ---
    video_frames: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        vid_id = extract_video_id(r["image"])
        video_frames[vid_id].append(r)

    logger.info("Grouped into %d videos", len(video_frames))

    video_true_labels = []
    video_pred_labels = []
    video_scores = []
    video_details = []

    for vid_id, frames in sorted(video_frames.items()):
        true_label = frames[0]["true_label"]
        preds = [f["pred_label"] for f in frames]
        counter = Counter(preds)
        voted_label = counter.most_common(1)[0][0]

        n_fake = counter.get("fake", 0)
        n_real = counter.get("real", 0)
        fake_ratio = n_fake / len(frames)

        video_true_labels.append(true_label)
        video_pred_labels.append(voted_label)
        video_scores.append(fake_ratio)

        video_details.append({
            "video_id": vid_id,
            "true_label": true_label,
            "voted_label": voted_label,
            "correct": voted_label == true_label,
            "n_frames": len(frames),
            "n_fake_preds": n_fake,
            "n_real_preds": n_real,
            "fake_ratio": round(fake_ratio, 4),
            "method": frames[0].get("method", "unknown"),
        })

    video_metrics = compute_metrics(video_true_labels, video_pred_labels, video_scores)
    video_metrics["level"] = "video"
    video_metrics["n_videos"] = len(video_frames)

    logger.info("=== Video-level metrics ===")
    for k, v in video_metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    # --- Per-method breakdown (frame-level) ---
    method_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        method_groups[r.get("method", "unknown")].append(r)

    method_metrics = {}
    for method, method_rows in sorted(method_groups.items()):
        m_true = [r["true_label"] for r in method_rows]
        m_pred = [r["pred_label"] for r in method_rows]
        m_scores = [1.0 if r["pred_label"] == "fake" else 0.0 for r in method_rows]
        method_metrics[method] = compute_metrics(m_true, m_pred, m_scores)
        logger.info(
            "  [%s] acc=%.4f f1=%.4f n=%d",
            method,
            method_metrics[method]["accuracy"],
            method_metrics[method]["f1"],
            method_metrics[method]["n_samples"],
        )

    # --- Save outputs ---
    frame_results_path = os.path.join(output_dir, "frame_results.json")
    with open(frame_results_path, "w") as f:
        json.dump({"frame_level": frame_metrics, "per_method": method_metrics}, f, indent=2)
    logger.info("Frame results saved to: %s", frame_results_path)

    video_results_path = os.path.join(output_dir, "video_results.json")
    with open(video_results_path, "w") as f:
        json.dump(video_metrics, f, indent=2)
    logger.info("Video results saved to: %s", video_results_path)

    video_csv_path = os.path.join(output_dir, "video_predictions.csv")
    fieldnames = [
        "video_id", "method", "true_label", "voted_label", "correct",
        "n_frames", "n_fake_preds", "n_real_preds", "fake_ratio",
    ]
    with open(video_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in video_details:
            writer.writerow(row)
    logger.info("Video predictions CSV saved to: %s", video_csv_path)

    # Combined summary
    summary = {
        "frame_level": frame_metrics,
        "video_level": video_metrics,
        "per_method": method_metrics,
    }
    summary_path = os.path.join(output_dir, "cross_dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Combined summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
