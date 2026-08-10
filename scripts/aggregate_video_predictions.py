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

# Sufixos de frame por dataset. FF++/Celeb-DF usam ``_fNN.jpg``; o WildDeepfake
# salva ``video_id/label/subfolder/frame`` como ``1_fake_1_168.png`` (sufixo
# ``_<frame>.png`` sem o ``f``). A ordem importa: o padrão específico de FF++
# é tentado antes do genérico para não sobre-remover.
_FRAME_SUFFIX_RES = [
    re.compile(r"_f\d+\.(jpg|jpeg|png)$", re.IGNORECASE),
    re.compile(r"_\d+\.(jpg|jpeg|png)$", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate predictions to video-level")
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.jsonl from evaluate.py")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (defaults to same dir as predictions)")
    p.add_argument("--metadata", default=None,
                   help="JSONL opcional (image, video_id) para mapear frame->vídeo "
                        "de forma exata; recomendado para o WildDeepfake. Sem ele, "
                        "o video_id é derivado do nome do arquivo.")
    return p.parse_args()


def load_video_id_map(metadata_path: str) -> dict:
    """Mapeia nome de imagem -> video_id a partir de um JSONL de metadata."""
    mapping: dict[str, str] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img, vid = r.get("image"), r.get("video_id")
            if img and vid is not None:
                mapping[img] = str(vid)
                mapping[os.path.basename(img)] = str(vid)
    return mapping


def extract_video_id(image_name: str, meta_map: dict | None = None) -> str:
    """Recupera o identificador de vídeo a partir do frame.

    Usa o ``video_id`` da metadata quando disponível (mapeamento exato); caso
    contrário, remove o sufixo de frame do nome do arquivo, cobrindo tanto o
    padrão ``_fNN.jpg`` (FF++/Celeb-DF) quanto ``_<frame>.png`` (WildDeepfake).
    """
    if meta_map:
        vid = meta_map.get(image_name) or meta_map.get(os.path.basename(image_name))
        if vid is not None:
            return vid
    for rex in _FRAME_SUFFIX_RES:
        if rex.search(image_name):
            return rex.sub("", image_name)
    return image_name


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


def _binary_metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Compute binary classification metrics at a given score threshold."""
    y_pred = (scores >= threshold).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
        "youden_j": round(rec - (1 - spec), 4),
    }


def threshold_sensitivity_analysis(
    rows: list[dict],
    video_frames: dict[str, list[dict]],
) -> dict:
    """Run threshold sweep on verdict_score for frame-level and video-level.

    Returns a dict with optimal thresholds and full sweep tables.
    """
    has_scores = "verdict_score" in rows[0]
    if not has_scores:
        return {}

    # --- Frame-level threshold analysis ---
    frame_y_true = np.array([1 if r["true_label"] == "fake" else 0 for r in rows])
    frame_scores = np.array([float(r["verdict_score"]) for r in rows])

    thresholds = [i * 0.25 for i in range(-24, 25)]  # -6.0 to +6.0
    frame_sweep = []
    best_frame_j = -1.0
    best_frame_thr = 0.0

    for thr in thresholds:
        m = _binary_metrics_at_threshold(frame_y_true, frame_scores, thr)
        frame_sweep.append(m)
        if m["youden_j"] > best_frame_j:
            best_frame_j = m["youden_j"]
            best_frame_thr = thr

    frame_default = _binary_metrics_at_threshold(frame_y_true, frame_scores, 0.0)
    frame_optimal = _binary_metrics_at_threshold(frame_y_true, frame_scores, best_frame_thr)

    # --- Video-level threshold analysis ---
    # For each threshold, re-do majority voting with threshold-based frame decisions
    video_ids = sorted(video_frames.keys())
    video_y_true = np.array([
        1 if video_frames[vid][0]["true_label"] == "fake" else 0
        for vid in video_ids
    ])

    video_sweep = []
    best_video_j = -1.0
    best_video_thr = 0.0

    for thr in thresholds:
        # For each video, count frames predicted fake at this threshold
        video_preds = []
        for vid in video_ids:
            frames = video_frames[vid]
            n_fake = sum(1 for f in frames if float(f["verdict_score"]) >= thr)
            video_preds.append(1 if n_fake > len(frames) / 2 else 0)
        video_preds_arr = np.array(video_preds)

        n = len(video_y_true)
        tp = int(((video_preds_arr == 1) & (video_y_true == 1)).sum())
        fp = int(((video_preds_arr == 1) & (video_y_true == 0)).sum())
        tn = int(((video_preds_arr == 0) & (video_y_true == 0)).sum())
        fn = int(((video_preds_arr == 0) & (video_y_true == 1)).sum())

        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = rec - (1 - spec)

        entry = {
            "threshold": round(thr, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "youden_j": round(j, 4),
        }
        video_sweep.append(entry)

        if j > best_video_j:
            best_video_j = j
            best_video_thr = thr

    # Recompute video metrics at default and optimal
    def _video_metrics_at(thr):
        preds = []
        for vid in video_ids:
            frames = video_frames[vid]
            n_fake = sum(1 for f in frames if float(f["verdict_score"]) >= thr)
            preds.append(1 if n_fake > len(frames) / 2 else 0)
        preds_arr = np.array(preds)
        n = len(video_y_true)
        tp = int(((preds_arr == 1) & (video_y_true == 1)).sum())
        fp = int(((preds_arr == 1) & (video_y_true == 0)).sum())
        tn = int(((preds_arr == 0) & (video_y_true == 0)).sum())
        fn = int(((preds_arr == 0) & (video_y_true == 1)).sum())
        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "threshold": round(thr, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "youden_j": round(rec - (1 - spec), 4),
        }

    video_default = _video_metrics_at(0.0)
    video_optimal = _video_metrics_at(best_video_thr)

    return {
        "frame_level": {
            "default_metrics": frame_default,
            "optimal_threshold": round(best_frame_thr, 4),
            "optimal_metrics": frame_optimal,
            "sweep": frame_sweep,
        },
        "video_level": {
            "default_metrics": video_default,
            "optimal_threshold": round(best_video_thr, 4),
            "optimal_metrics": video_optimal,
            "sweep": video_sweep,
        },
    }


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
    meta_map = load_video_id_map(args.metadata) if args.metadata else None
    if meta_map:
        logger.info("Mapeamento de video_id carregado da metadata: %d entradas", len(meta_map))
    video_frames: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        vid_id = extract_video_id(r["image"], meta_map)
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

    # --- Threshold sensitivity analysis ---
    threshold_results = threshold_sensitivity_analysis(rows, video_frames)
    if threshold_results:
        threshold_path = os.path.join(output_dir, "threshold_analysis.json")
        with open(threshold_path, "w") as f:
            json.dump(threshold_results, f, indent=2)
        logger.info("Threshold analysis saved to: %s", threshold_path)

        fl = threshold_results["frame_level"]
        vl = threshold_results["video_level"]
        logger.info("=== Threshold Sensitivity ===")
        logger.info(
            "  Frame: default(thr=0) acc=%.4f f1=%.4f | optimal(thr=%.2f) acc=%.4f f1=%.4f",
            fl["default_metrics"]["accuracy"], fl["default_metrics"]["f1"],
            fl["optimal_threshold"],
            fl["optimal_metrics"]["accuracy"], fl["optimal_metrics"]["f1"],
        )
        logger.info(
            "  Video: default(thr=0) acc=%.4f f1=%.4f | optimal(thr=%.2f) acc=%.4f f1=%.4f",
            vl["default_metrics"]["accuracy"], vl["default_metrics"]["f1"],
            vl["optimal_threshold"],
            vl["optimal_metrics"]["accuracy"], vl["optimal_metrics"]["f1"],
        )
    else:
        threshold_results = {}
        logger.info("No verdict_score found; skipping threshold analysis.")

    # Combined summary
    summary = {
        "frame_level": frame_metrics,
        "video_level": video_metrics,
        "per_method": method_metrics,
    }
    if threshold_results:
        summary["threshold_optimal"] = {
            "frame_level": {
                "threshold": threshold_results["frame_level"]["optimal_threshold"],
                **threshold_results["frame_level"]["optimal_metrics"],
            },
            "video_level": {
                "threshold": threshold_results["video_level"]["optimal_threshold"],
                **threshold_results["video_level"]["optimal_metrics"],
            },
        }
    summary_path = os.path.join(output_dir, "cross_dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Combined summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
