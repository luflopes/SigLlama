#!/usr/bin/env python3
"""Qualitative side-by-side of the GLOBAL-question answer for two text models.

For each selected test image: the face on the left, and two text panels on the
right with the generated answer of each model (e.g. A3-ablation vs
A3-clean+gold), plus the human reference. Lets us *see* whether the revised
protocol produces less formulaic / more grounded descriptions.

Usage::

    python scripts/plot_text_compare.py \\
        --a outputs/ablation/g3_lora/evaluation/best_test/predictions.jsonl \\
        --b outputs/ablation/g3_clean_gold/evaluation/cleantest/predictions.jsonl \\
        --labels A3-ablation A3-clean+gold \\
        --out-dir outputs/analysis/a3_text_compare --per-label 4
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

GLOBAL_Q = "Does the image looks real/fake?"


def load_global(path: Path) -> dict:
    idx = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("question") == GLOBAL_Q:
            idx[r["image"]] = r
    return idx


def find_image(name: str, roots) -> Path | None:
    for root in roots:
        p = root / name
        if p.is_file():
            return p
    return None


def wrap(t, width=52):
    return textwrap.fill((t or "(vazio)").strip(), width=width)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default=str(repo / "outputs/ablation/g3_lora/evaluation/best_test/predictions.jsonl"))
    p.add_argument("--b", default=str(repo / "outputs/ablation/g3_clean_gold/evaluation/cleantest/predictions.jsonl"))
    p.add_argument("--labels", nargs=2, default=["A3-ablation", "A3-clean+gold"])
    p.add_argument("--out-dir", default=str(repo / "outputs/analysis/a3_text_compare"))
    p.add_argument("--frame-roots", nargs="+",
                   default=[str(repo / "label_studio/data/frames"), str(repo / "ddvqa/frames")])
    p.add_argument("--per-label", type=int, default=4,
                   help="quantas imagens de cada classe (real/fake)")
    p.add_argument("--only-diff", action="store_true",
                   help="só imagens onde as duas respostas geradas diferem")
    args = p.parse_args()

    roots = [Path(r) for r in args.frame_roots]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    A = load_global(Path(args.a))
    B = load_global(Path(args.b))
    common = [im for im in A if im in B]

    buckets = {"real": [], "fake": []}
    for im in sorted(common):
        lab = str(A[im].get("true_label", "")).lower()
        if lab not in buckets:
            continue
        if args.only_diff and A[im].get("generated", "").strip() == B[im].get("generated", "").strip():
            continue
        buckets[lab].append(im)

    selected = []
    for lab in ("fake", "real"):
        selected += buckets[lab][: args.per_label]

    made = 0
    for im in selected:
        fp = find_image(im, roots)
        if fp is None:
            print(f"  (imagem ausente) {im}")
            continue
        img = Image.open(fp).convert("RGB")
        ra, rb = A[im], B[im]
        true_lab = str(ra.get("true_label", "?"))

        fig = plt.figure(figsize=(13, 5.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 3, 3], wspace=0.08)

        axi = fig.add_subplot(gs[0, 0]); axi.imshow(img); axi.axis("off")
        axi.set_title(f"{ra.get('method','?')} / {true_lab}", fontsize=11, fontweight="bold")

        panels = [
            (gs[0, 1], args.labels[0], ra),
            (gs[0, 2], args.labels[1], rb),
        ]
        for cell, lbl, row in panels:
            ax = fig.add_subplot(cell); ax.axis("off")
            ok = bool(row.get("correct"))
            col = "#2ca02c" if ok else "#d62728"
            ax.set_title(f"{lbl}   [{'✓' if ok else '✗'} veredito]",
                         fontsize=11, color=col, fontweight="bold")
            ax.text(0.0, 0.98, wrap(row.get("generated", "")), va="top", ha="left",
                    fontsize=8.5, family="monospace")

        fig.suptitle(f"{im}   —   ref: {wrap(ra.get('reference_answer',''), 90)[:90]}...",
                     fontsize=9)
        dest = out / f"{true_lab}_{Path(im).stem}.png"
        fig.savefig(dest, dpi=115, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  -> {dest}")

    print(f"\n{made} figuras em {out}")


if __name__ == "__main__":
    main()
