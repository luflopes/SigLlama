#!/usr/bin/env python
"""Figura de sensibilidade ao limiar de decisão (cross-dataset).

Gera o gráfico de acurácia em nível de vídeo em função do limiar de decisão para
Celeb-DF-v2 e WildDeepfake, comparando as configurações A3, A4 e o classificador
DINOv2. Marca o veredito nativo (limiar 0), o limiar de Youden calibrado na
validação do DD-VQA (~1,75) e o ótimo estimado por dataset (estrela).

Fontes de dados:
  - A3/A4: varredura em nível de vídeo já computada em
    ``outputs/cross_dataset/{dataset}_{exp}/evaluation/threshold_analysis.json``
    (campo ``video_level.sweep``: threshold -> accuracy).
  - DINOv2: varredura computada aqui a partir de
    ``outputs/dino_lora_classifier/scores_{dataset}_test.jsonl`` usando o escore
    contínuo ``logit_fake - logit_real`` e voto majoritário por vídeo.

Uso:
    python scripts/plot_threshold_sensitivity.py

Reproduz a figura ``threshold_sensitivity.{pdf,png}``. Este script foi
reconstruído a partir do snippet ad-hoc original (que não havia sido salvo).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Configuração ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CROSS_DIR = ROOT / "outputs" / "cross_dataset"
DINO_DIR = ROOT / "outputs" / "dino_lora_classifier"
OUT_DIRS = [ROOT / "outputs" / "figures", CROSS_DIR]

GRID = np.arange(-6, 6.01, 0.25)
YOUDEN_VAL = 1.75  # limiar de Youden calibrado na validação do DD-VQA
COL = {"A3": "#1f77b4", "A4": "#d62728", "DINO": "#7030a0"}

# Experimentos que fornecem o verdict score de cada configuração.
# A3 = g3_lora; A4 = g4_lora_loc (troque por "g4_auto_clean" para o A4 no
# protocolo limpo atual — o verdict score é o mesmo modelo de localização).
A3_EXP = "g3_lora"
A4_EXP = "g4_auto_clean"

# (rótulo do painel -> prefixo dos diretórios/arquivos)
DATASETS = {"Celeb-DF-v2": "celebdf", "WildDeepfake": "wild"}


def vlm_sweep(prefix: str, exp: str):
    """Varredura video-level (threshold -> accuracy) do threshold_analysis.json."""
    path = CROSS_DIR / f"{prefix}_{exp}" / "evaluation" / "threshold_analysis.json"
    vl = json.load(open(path))["video_level"]
    sweep = vl["sweep"]
    x = [p["threshold"] for p in sweep]
    acc = [p["accuracy"] for p in sweep]
    return x, acc, vl["optimal_threshold"], vl["optimal_metrics"]["accuracy"]


def dino_sweep(prefix: str):
    """Varredura video-level do DINOv2 a partir de logit_fake - logit_real."""
    path = DINO_DIR / f"scores_{prefix}_test.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    groups: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for r in rows:
        vid = r["video_id"]
        diff = float(r["logit_fake"]) - float(r["logit_real"])
        groups[vid].append(diff)
        labels[vid] = 1 if r["true_label"] == "fake" else 0
    vids = list(groups)
    y = np.array([labels[v] for v in vids])
    accs = []
    for t in GRID:
        vp = np.array([
            1 if np.mean([1 if d >= t else 0 for d in groups[v]]) >= 0.5 else 0
            for v in vids
        ])
        accs.append(float((vp == y).mean()))
    accs = np.array(accs)
    k = int(np.argmax(accs))
    return list(GRID), list(accs), float(GRID[k]), float(accs[k])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    opt_report: dict[tuple[str, str], tuple[float, float]] = {}

    for ax, (ds_label, prefix) in zip(axes, DATASETS.items()):
        for model, exp in [("A3", A3_EXP), ("A4", A4_EXP)]:
            x, acc, opt, oacc = vlm_sweep(prefix, exp)
            ax.plot(x, acc, color=COL[model], lw=2, label=model)
            ax.scatter([opt], [oacc], color=COL[model], marker="*", s=160,
                       zorder=5, edgecolor="white", lw=0.6)
            opt_report[(ds_label, model)] = (opt, oacc)

        x, acc, opt, oacc = dino_sweep(prefix)
        ax.plot(x, acc, color=COL["DINO"], lw=2, label="DINOv2")
        ax.scatter([opt], [oacc], color=COL["DINO"], marker="*", s=160,
                   zorder=5, edgecolor="white", lw=0.6)
        opt_report[(ds_label, "DINO")] = (opt, oacc)

        ax.axvline(0.0, color="0.35", ls=":", lw=1.6)
        ax.axvline(YOUDEN_VAL, color="#ff7f0e", ls="--", lw=1.6)
        ax.set_title(ds_label, fontsize=12)
        ax.set_xlabel("Limiar (verdict score / logit Fake $-$ logit Real)", fontsize=10.5)
        ax.set_xlim(-6, 6)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Acurácia (nível de vídeo)", fontsize=11)
    axes[0].set_ylim(0.30, 0.90)

    handles = [
        Line2D([0], [0], color=COL["A3"], lw=2, label="A3"),
        Line2D([0], [0], color=COL["A4"], lw=2, label="A4"),
        Line2D([0], [0], color=COL["DINO"], lw=2, label="DINOv2"),
        Line2D([0], [0], color="0.35", ls=":", lw=1.6, label="Veredito nativo (limiar 0)"),
        Line2D([0], [0], color="#ff7f0e", ls="--", lw=1.6,
               label="Youden calibrado (val. DD-VQA, $\\approx$1,75)"),
        Line2D([0], [0], color="0.3", marker="*", ls="none", ms=12, label="Ótimo por dataset"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "threshold_sensitivity.pdf", bbox_inches="tight")
        fig.savefig(d / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")

    print("Ótimos estimados por dataset (limiar de oráculo, apenas referência):")
    for (ds, m), (thr, acc) in opt_report.items():
        print(f"  {ds:12s} {m:6s} thr={thr:+.2f}  acc={acc:.3f}")
    print("\nFiguras salvas em:")
    for d in OUT_DIRS:
        print(f"  {d / 'threshold_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
