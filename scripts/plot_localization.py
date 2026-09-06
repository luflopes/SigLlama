#!/usr/bin/env python3
"""Qualitative side-by-side of grounding: gold vs A4-manual vs A4-auto.

For each (fake) test image draws three panels with the boxes overlaid on
the frame and the full generated/reference text underneath, so the
localization *and* the description can be inspected together.

Boxes are ``[y1,x1,y2,x2]`` integers in ``[0,1000]`` (as emitted inline in
``generated`` / ``reference_answer``), rescaled to pixel coordinates.

Usage::

    python scripts/plot_localization.py \\
        --manual outputs/ablation/g4_a4_gold/evaluation/best_test/predictions.jsonl \\
        --auto   outputs/ablation/g4_lora_loc/evaluation/goldtest/predictions.jsonl \\
        --out-dir outputs/analysis/loc_plots --per-method 2
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

BOX_RE = re.compile(r"\[(\d+),(\d+),(\d+),(\d+)\]")


def parse_boxes(text: str):
    out = []
    for m in BOX_RE.finditer(text or ""):
        y1, x1, y2, x2 = (int(m.group(i)) for i in range(1, 5))
        y1, y2 = sorted((y1, y2))
        x1, x2 = sorted((x1, x2))
        if y2 > y1 and x2 > x1:
            out.append((y1, x1, y2, x2))
    return out


def load(path: Path):
    idx = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            idx[r["image"]] = r
    return idx


def find_image(name: str, roots) -> Path | None:
    for root in roots:
        p = root / name
        if p.is_file():
            return p
    return None


def draw(ax, img, boxes, color, title):
    W, H = img.size
    ax.imshow(img)
    for (y1, x1, y2, x2) in boxes:
        px, py = x1 / 1000 * W, y1 / 1000 * H
        w, h = (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H
        ax.add_patch(Rectangle((px, py), w, h, fill=False, edgecolor=color, linewidth=2.2))
    ax.set_title(f"{title}  (n={len(boxes)})", fontsize=11, color=color, fontweight="bold")
    ax.axis("off")


def wrap(t, width=48):
    return textwrap.fill(t or "(vazio)", width=width)


def single_panel_mode(jsonl: Path, out: Path, roots, per_method: int,
                      field: str, panel_title: str) -> None:
    """Render single-panel boxes+text from ``field`` of each row.

    field="answer"     -> gold annotations (dataset jsonl)
    field="generated"  -> model-generated annotations (predictions.jsonl)
    """
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r.get("method", "?"), []).append(r)

    selected = []
    for method, items in sorted(by_method.items()):
        selected += items if per_method == 0 else items[:per_method]

    made = 0
    for r in selected:
        im = r["image"]
        fp = find_image(im, roots)
        if fp is None:
            print(f"  (imagem ausente) {im}")
            continue
        img = Image.open(fp).convert("RGB")
        text = r.get(field, "")
        boxes = parse_boxes(text)
        is_real = bool(r.get("is_real")) or str(r.get("true_label", "")).lower() == "real"
        color = "#9467bd" if is_real else ("#2ca02c" if field == "answer" else "#1f77b4")
        fig = plt.figure(figsize=(6, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.04)
        draw(fig.add_subplot(gs[0, 0]), img, boxes, color, panel_title)
        axt = fig.add_subplot(gs[1, 0]); axt.axis("off")
        axt.text(0.02, 1.0, wrap(text, width=64), va="top", ha="left",
                 fontsize=8, family="monospace")
        tag = "real" if is_real else "fake"
        fig.suptitle(f"{im}  [{r.get('method','?')} / {tag}]", fontsize=12, fontweight="bold")
        dest = out / f"{Path(im).stem}.png"
        fig.savefig(dest, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  -> {dest}")
    print(f"\n{made} figuras ({panel_title}) em {out}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manual", default=str(repo / "outputs/ablation/g4_a4_gold/evaluation/best_test/predictions.jsonl"))
    p.add_argument("--auto", default=str(repo / "outputs/ablation/g4_lora_loc/evaluation/goldtest/predictions.jsonl"))
    p.add_argument("--out-dir", default=str(repo / "outputs/analysis/loc_plots"))
    # IMPORTANT: annotations (and training/eval) use the tarkin-aligned crop in
    # label_studio/data/frames (== ddvqa_prepared/frames). The local ddvqa/frames
    # copy is an OLDER extraction with a DIFFERENT crop -> boxes would be displaced.
    p.add_argument("--frame-roots", nargs="+", default=[str(repo / "label_studio/data/frames"), str(repo / "ddvqa/frames")])
    p.add_argument("--per-method", type=int, default=2, help="images per method (0 = all)")
    p.add_argument("--label", choices=["fake", "real", "all"], default="fake",
                   help="which images to plot in comparison mode")
    p.add_argument("--gold-only", default=None,
                   help="path to a dataset jsonl (e.g. train_loc_gold.jsonl); renders "
                        "single-panel GOLD boxes+text (field 'answer')")
    p.add_argument("--pred-only", default=None,
                   help="path to a predictions.jsonl (e.g. the pool run); renders "
                        "single-panel MODEL boxes+text (field 'generated')")
    args = p.parse_args()

    roots = [Path(r) for r in args.frame_roots]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.gold_only:
        single_panel_mode(Path(args.gold_only), out, roots, args.per_method,
                          field="answer", panel_title="GOLD")
        return
    if args.pred_only:
        single_panel_mode(Path(args.pred_only), out, roots, args.per_method,
                          field="generated", panel_title="A4 (gerado)")
        return

    man = load(Path(args.manual))
    aut = load(Path(args.auto))

    def keep(im):
        lab = str(man[im].get("true_label", "")).lower()
        return args.label == "all" or lab == args.label

    common = [im for im in man if im in aut and keep(im)]
    by_method: dict[str, list[str]] = {}
    for im in sorted(common):
        by_method.setdefault(man[im].get("method", "?"), []).append(im)

    selected = []
    for method, imgs in sorted(by_method.items()):
        selected += imgs if args.per_method == 0 else imgs[: args.per_method]

    made = []
    for im in selected:
        fp = find_image(im, roots)
        if fp is None:
            print(f"  (imagem ausente) {im}")
            continue
        img = Image.open(fp).convert("RGB")
        mrow, arow = man[im], aut[im]
        ref_b = parse_boxes(mrow.get("reference_answer", ""))
        man_b = parse_boxes(mrow.get("generated", ""))
        aut_b = parse_boxes(arow.get("generated", ""))

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], hspace=0.05, wspace=0.05)
        for col, (boxes, color, title) in enumerate([
            (ref_b, "#2ca02c", "GOLD (ref)"),
            (man_b, "#1f77b4", "A4-manual"),
            (aut_b, "#d62728", "A4-auto"),
        ]):
            draw(fig.add_subplot(gs[0, col]), img, boxes, color, title)
        for col, txt in enumerate([
            mrow.get("reference_answer", ""),
            mrow.get("generated", ""),
            arow.get("generated", ""),
        ]):
            axt = fig.add_subplot(gs[1, col]); axt.axis("off")
            axt.text(0.02, 1.0, wrap(txt), va="top", ha="left", fontsize=7.5, family="monospace")
        fig.suptitle(f"{im}  [{man[im].get('method','?')}]", fontsize=13, fontweight="bold")
        dest = out / f"{Path(im).stem}.png"
        fig.savefig(dest, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append(dest)
        print(f"  -> {dest}")

    print(f"\n{len(made)} figuras em {out}")
    if made:
        print("Primeiras:")
        for d in made[:6]:
            print(f"  {d}")


if __name__ == "__main__":
    main()
