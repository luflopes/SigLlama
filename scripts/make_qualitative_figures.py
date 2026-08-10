#!/usr/bin/env python
"""Gera as figuras da seção de Análise Qualitativa da dissertação.

Produz três artefatos a partir das predições da configuração A4 no test set do
DD-VQA (deduplicado por imagem):

1. Nuvens de palavras das explicações, separadas por veredito (Fake vs Real).
2. Barras divergentes dos termos mais discriminativos por veredito (log-odds).
3. Painel de casos de falha, com as caixas delimitadoras desenhadas quando
   presentes na explicação gerada.

As caixas do DD-VQA seguem a ordem [y1, x1, y2, x2] em escala normalizada
[0, 1000], com x relativo à largura e y relativo à altura da imagem.

Uso:
    venv/bin/python scripts/make_qualitative_figures.py
"""
from __future__ import annotations

import json
import math
import os
import re
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from wordcloud import WordCloud

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
PRED_PATH = "outputs/ablation/g4_lora_loc/evaluation/best_test/predictions.jsonl"
FRAMES_DIR = "ddvqa/frames"
OUT_DIRS = ["notebooks/figures", "outputs/figures"]

FAKE_COLOR = "#c0392b"
REAL_COLOR = "#1e8449"
BOX_COLOR = "#f1c40f"

# Casos de falha exibidos no painel: (imagem, método, rótulo, predito).
FAILURE_CASES = [
    ("Face2Face_233_995.jpg", "Face2Face", "Fake", "Real"),
    ("NeuralTextures_707_705.jpg", "NeuralTextures", "Fake", "Real"),
    ("FaceSwap_190_176.jpg", "FaceSwap", "Fake", "Real"),
    ("Original_507.jpg", "Original", "Real", "Fake"),
]

STOPWORDS = set(
    "the a an is are it its of to on in with and or but as at looks look image "
    "person face real fake very this that has have there be no not do does "
    "looking appears appear seems bit other because area".split()
)
_BOX_RE = re.compile(r"\[(\d+),(\d+),(\d+),(\d+)\]")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def load_unique_predictions(path: str) -> list[dict]:
    """Carrega predições deduplicando por imagem (uma entrada por face)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by_image = {r["image"]: r for r in rows}
    return list(by_image.values())


def tokenize(text: str) -> list[str]:
    text = _BOX_RE.sub(" ", text.lower())
    return [w for w in re.findall(r"[a-zA-Z]+", text) if w not in STOPWORDS and len(w) > 2]


def verdict_counters(rows: list[dict]):
    """Retorna (freq_total_fake, freq_total_real, docfreq_fake, docfreq_real, n_fake, n_real)."""
    tf, tr, cf, cr = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        words = tokenize(r["generated"])
        if r["pred_label"] == "fake":
            tf.update(words)
            cf.update(set(words))
        else:
            tr.update(words)
            cr.update(set(words))
    n_fake = sum(1 for r in rows if r["pred_label"] == "fake")
    n_real = len(rows) - n_fake
    return tf, tr, cf, cr, n_fake, n_real


def save(fig, name: str, **kwargs):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.pdf"), **kwargs)
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=150, **kwargs)


# --------------------------------------------------------------------------- #
# 1) Nuvens de palavras
# --------------------------------------------------------------------------- #
def make_wordclouds(tf: Counter, tr: Counter):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, freqs, cmap, title, color in [
        (axes[0], tf, "Reds", "Veredito Fake", FAKE_COLOR),
        (axes[1], tr, "Greens", "Veredito Real", REAL_COLOR),
    ]:
        wc = WordCloud(
            width=650, height=520, background_color="white", colormap=cmap,
            prefer_horizontal=0.95, relative_scaling=0.5, min_font_size=10,
            max_words=45, random_state=1,
        ).generate_from_frequencies(freqs)
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=color, fontweight="bold")
    fig.tight_layout()
    save(fig, "verdict_wordclouds", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2) Barras divergentes (log-odds) — mantido para reuso futuro
# --------------------------------------------------------------------------- #
def make_lexicon_divergent(cf, cr, n_fake, n_real, min_count=8, top_k=12):
    import numpy as np

    vocab = {w for w in (set(cf) | set(cr)) if (cf[w] + cr[w]) >= min_count}
    log_odds = {}
    for w in vocab:
        a, b = cf[w] + 0.5, n_fake - cf[w] + 0.5
        c, d = cr[w] + 0.5, n_real - cr[w] + 0.5
        log_odds[w] = math.log((a / b) / (c / d))
    top_fake = sorted(log_odds.items(), key=lambda x: -x[1])[:top_k]
    top_real = sorted(log_odds.items(), key=lambda x: x[1])[:top_k]
    items = top_fake[::-1] + top_real
    labels = [w for w, _ in items]
    values = [v for _, v in items]
    colors = [FAKE_COLOR if v > 0 else REAL_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(items))
    ax.barh(y, values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel(
        "log-odds  (\u2190 mais associada a Real    |    mais associada a Fake \u2192)",
        fontsize=10,
    )
    ax.set_title("Termos mais discriminativos das explicações por veredito (A4)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "verdict_lexicon")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) Painel de casos de falha
# --------------------------------------------------------------------------- #
def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(int(v) for v in m) for m in _BOX_RE.findall(text)]


def make_failure_panel(rows: list[dict]):
    generated = {r["image"]: r["generated"] for r in rows}

    fig = plt.figure(figsize=(9.2, 8.6))
    outer = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10)

    for k, (img, method, true_lab, pred_lab) in enumerate(FAILURE_CASES):
        r, c = divmod(k, 2)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.02)
        ax_img = fig.add_subplot(inner[0])
        ax_txt = fig.add_subplot(inner[1])

        im = mpimg.imread(os.path.join(FRAMES_DIR, img))
        h, w = im.shape[:2]
        ax_img.imshow(im)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color(FAKE_COLOR)
            spine.set_linewidth(2.2)

        # As caixas do DD-VQA seguem a ordem [y1, x1, y2, x2] em escala [0, 1000].
        for (y1, x1, y2, x2) in parse_boxes(generated[img]):
            rx, ry = x1 / 1000 * w, y1 / 1000 * h
            rw, rh = (x2 - x1) / 1000 * w, (y2 - y1) / 1000 * h
            ax_img.add_patch(
                Rectangle((rx, ry), rw, rh, fill=False, edgecolor=BOX_COLOR, lw=2.6)
            )

        ax_img.set_title(method, fontsize=13, fontweight="bold", pad=5)

        # Painel textual
        ax_txt.axis("off")
        ax_txt.text(0.30, 0.98, f"Rótulo: {true_lab}", ha="center", va="top",
                    fontsize=10.5, fontweight="bold", color=REAL_COLOR,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.70, 0.98, f"Predito: {pred_lab}", ha="center", va="top",
                    fontsize=10.5, fontweight="bold", color=FAKE_COLOR,
                    transform=ax_txt.transAxes)
        snippet = generated[img]
        snippet = snippet[:150] + "\u2026" if len(snippet) > 150 else snippet
        wrapped = "\n".join(textwrap.wrap(snippet, 52))
        ax_txt.text(0.5, 0.68, wrapped, ha="center", va="top", fontsize=10,
                    style="italic", color="0.12", transform=ax_txt.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f6f6f6", ec="0.82", lw=0.8))

    save(fig, "failure_cases", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    rows = load_unique_predictions(PRED_PATH)
    tf, tr, cf, cr, n_fake, n_real = verdict_counters(rows)
    make_wordclouds(tf, tr)
    make_lexicon_divergent(cf, cr, n_fake, n_real)
    make_failure_panel(rows)
    print(f"Figuras geradas para {len(rows)} imagens (n_fake={n_fake}, n_real={n_real}).")


if __name__ == "__main__":
    main()
