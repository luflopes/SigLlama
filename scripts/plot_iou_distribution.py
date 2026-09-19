#!/usr/bin/env python3
"""Gera o gráfico composto de distribuição de IoU (A4 vs A6).

Reproduz, fora do notebook, o painel duplo usado na tese (fig:iou_dist):
histograma de IoU por caixa de referência (apenas IoU>0), com faixas
coloridas por nível de sobreposição e linhas de corte HR@0,3 / HR@0,5.

A4 = outputs/ablation/g4_auto_clean/evaluation/cleantest
A6 = outputs/ablation/g6_full/evaluation/cleantest   (mesmo gerador + veredito
     do classificador; distribuições praticamente sobreponíveis)

Uso:
    python scripts/plot_iou_distribution.py
Saída: outputs/figures/iou_distribution.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.metrics import parse_loc_tokens, box_iou  # noqa: E402

FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("FaceGroundVLM-A4 (loc)", "g4_auto_clean"),
    ("FaceGroundVLM-A6 (full)", "g6_full"),
]


def load_preds(exp_dir: str) -> list[dict]:
    path = ROOT / "outputs" / "ablation" / exp_dir / "evaluation" / "cleantest" / "predictions.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def iou_distribution(rows: list[dict]) -> list[float]:
    ious: list[float] = []
    for r in rows:
        pred = parse_loc_tokens(r["generated"])
        ref = parse_loc_tokens(r["reference_answer"])
        if not ref:
            continue
        for rb in ref:
            ious.append(max((box_iou(pb, rb) for pb in pred), default=0.0))
    return ious


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
        "axes.titlesize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "axes.facecolor": "white",
    })

    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 4.5), squeeze=False)
    last_mean = 0.0
    for i, (label, exp_dir) in enumerate(PANELS):
        ious = [v for v in iou_distribution(load_preds(exp_dir)) if v > 0.0]
        ax = axes[0, i]
        counts, bins, patches = ax.hist(ious, bins=40, zorder=3, alpha=0.9)
        for j in range(len(patches)):
            c = (bins[j] + bins[j + 1]) / 2
            if c < 0.3:
                patches[j].set_facecolor("#ffcccc"); patches[j].set_edgecolor("#8b0000")
                patches[j].set_linewidth(1.0); patches[j].set_linestyle("--")
            elif c < 0.5:
                patches[j].set_facecolor("#8cb4f8"); patches[j].set_edgecolor("#00008b")
                patches[j].set_linewidth(1.2)
            else:
                patches[j].set_facecolor("#8fe38f"); patches[j].set_edgecolor("#006400")
                patches[j].set_linewidth(1.2)
        ax.axvline(0.3, color="#00008b", ls="--", lw=1.5, zorder=4)
        ax.axvline(0.5, color="#006400", ls="--", lw=1.5, zorder=4)
        ax.set_xlabel("IoU", weight="bold")
        ax.set_ylabel("Contagem", weight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.7, linestyle="-", color="#d3d3d3", zorder=0)
        mean_iou = float(np.mean(ious)) if ious else 0.0
        last_mean = mean_iou
        ax.text(0.97, 0.73, f"IoU Médio = {mean_iou:.3f}", transform=ax.transAxes,
                ha="right", va="center", fontsize=11, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))

    legend_elements = [
        Patch(facecolor="#ffcccc", edgecolor="#8b0000", lw=1.5, linestyle="--", label="IoU < 0.3 (Baixo)"),
        Patch(facecolor="#8cb4f8", edgecolor="#00008b", linestyle="-", label="0.3 ≤ IoU < 0.5 (Razoável)"),
        Patch(facecolor="#8fe38f", edgecolor="#006400", linestyle="-", label="IoU ≥ 0.5 (Excelente)"),
        Line2D([0], [0], color="#00008b", lw=1.5, linestyle="--", label="HR@0,3"),
        Line2D([0], [0], color="#006400", lw=1.5, linestyle="--", label="HR@0,5"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.03),
               ncol=3, frameon=True, edgecolor="gray", facecolor="white")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIGURES_DIR / "iou_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "iou_distribution.png", dpi=300, bbox_inches="tight")
    print(f"Figura salva em {FIGURES_DIR}/iou_distribution.pdf (e .png)")


if __name__ == "__main__":
    main()
