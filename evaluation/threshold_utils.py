"""Threshold calibration utilities for binary deepfake detection.

Shared by the analysis notebook and any script that needs to pick an
operating point on a validation split and evaluate it on test.

Score convention: **higher score => more likely fake**. This holds for
both the VLM ``verdict_score`` (log P(fake) - log P(real), default
cut-off 0.0) and the DINOv2 classifier ``score`` (softmax P(fake),
default cut-off 0.5).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

_METRIC_KEYS = {
    "youden": "youden_j",
    "f1": "f1",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def _to_binary(labels: Sequence) -> np.ndarray:
    """Map labels to {0 real, 1 fake}, accepting strings or ints."""
    out = []
    for l in labels:
        if isinstance(l, str):
            out.append(1 if l.strip().lower() == "fake" else 0)
        else:
            out.append(int(l))
    return np.asarray(out, dtype=int)


def metrics_at_threshold(
    labels: Sequence, scores: Sequence[float], threshold: float
) -> dict:
    """Compute binary detection metrics at a fixed decision threshold.

    A sample is predicted fake when ``score >= threshold``.
    """
    y_true = _to_binary(labels)
    s = np.asarray(scores, dtype=float)
    y_pred = (s >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + fp + tn + fn

    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal_acc = 0.5 * (rec + spec)

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "youden_j": rec + spec - 1.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
    }


def build_threshold_grid(
    scores: Sequence[float], grid: Iterable[float] | None = None, n: int = 201
) -> np.ndarray:
    """Return candidate thresholds spanning the score range (with padding)."""
    if grid is not None:
        return np.asarray(list(grid), dtype=float)
    s = np.asarray(scores, dtype=float)
    lo, hi = float(np.min(s)), float(np.max(s))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.02 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def sweep_thresholds(
    labels: Sequence,
    scores: Sequence[float],
    grid: Iterable[float] | None = None,
    n: int = 201,
) -> list[dict]:
    """Evaluate metrics across a grid of thresholds."""
    grid_arr = build_threshold_grid(scores, grid, n)
    return [metrics_at_threshold(labels, scores, t) for t in grid_arr]


def find_optimal_threshold(
    labels: Sequence,
    scores: Sequence[float],
    metric: str = "youden",
    grid: Iterable[float] | None = None,
    n: int = 201,
) -> tuple[float, dict, list[dict]]:
    """Find the threshold maximising ``metric`` on the given (val) data.

    Parameters
    ----------
    metric : one of "youden", "f1", "accuracy", "balanced_accuracy".

    Returns
    -------
    (best_threshold, best_metrics, full_sweep)
    """
    if metric not in _METRIC_KEYS:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from {sorted(_METRIC_KEYS)}."
        )
    key = _METRIC_KEYS[metric]
    sweep = sweep_thresholds(labels, scores, grid, n)
    best = max(sweep, key=lambda m: (m[key], m["accuracy"]))
    return best["threshold"], best, sweep


def roc_auc(labels: Sequence, scores: Sequence[float]) -> float | None:
    """ROC-AUC (higher score = fake). Returns None if only one class."""
    y_true = _to_binary(labels)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, np.asarray(scores, dtype=float)))
    except Exception:
        return None


def roc_curve_points(labels: Sequence, scores: Sequence[float]):
    """Return (fpr, tpr, thresholds) arrays for plotting a ROC curve."""
    from sklearn.metrics import roc_curve

    y_true = _to_binary(labels)
    fpr, tpr, thr = roc_curve(y_true, np.asarray(scores, dtype=float))
    return fpr, tpr, thr
