#!/usr/bin/env python3
"""Localization metrics for DD-VQA grounded predictions.

Reads a ``predictions.jsonl`` from ``evaluation/evaluate.py`` (boxes inline as
``[y1,x1,y2,x2]`` in ``[0,1000]`` inside ``generated`` and ``reference_answer``)
and computes geometric localization metrics against the reference boxes:

  - gen_cov   : fraction of samples where the model emitted >=1 box
  - boxes/img : mean boxes per sample (generated vs reference)
  - mIoU      : mean best-match IoU per reference box (0 if uncovered)
  - hit@t     : recall = frac of reference boxes covered by some gen box (IoU>=t)
  - prec@t    : frac of generated boxes overlapping some ref box (IoU>=t)

Matching is region-agnostic (pure geometry): this is the honest
"did the predicted box land on the annotated artifact?" signal.

Usage::

    python scripts/eval_localization.py \\
        --pred outputs/ablation/g4_gold/evaluation/best_test/predictions.jsonl

    # Compare two models on the same test set (A4-auto vs A4-manual)
    python scripts/eval_localization.py \\
        --pred  outputs/ablation/g4_gold/evaluation/best_test/predictions.jsonl \\
        --compare outputs/ablation/g4_lora_loc/evaluation/gold_test/predictions.jsonl \\
        --labels A4-manual A4-auto
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

BOX_RE = re.compile(r"\[(\d+),(\d+),(\d+),(\d+)\]")


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    """Return [y1,x1,y2,x2] boxes, normalised so y1<y2, x1<x2, clamped 0-1000."""
    out = []
    for m in BOX_RE.finditer(text or ""):
        y1, x1, y2, x2 = (int(m.group(i)) for i in range(1, 5))
        y1, y2 = sorted((y1, y2))
        x1, x2 = sorted((x1, x2))
        y1 = max(0, min(1000, y1)); y2 = max(0, min(1000, y2))
        x1 = max(0, min(1000, x1)); x2 = max(0, min(1000, x2))
        if y2 > y1 and x2 > x1:
            out.append((y1, x1, y2, x2))
    return out


def iou(a, b) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ay2 - ay1) * (ax2 - ax1)
    area_b = (by2 - by1) * (bx2 - bx1)
    return inter / (area_a + area_b - inter)


def best_iou(box, others) -> float:
    return max((iou(box, o) for o in others), default=0.0)


def evaluate(path: str, thresholds=(0.3, 0.5)):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    agg = defaultdict(lambda: {
        "n": 0, "samples_with_gen": 0,
        "n_gen": 0, "n_ref": 0,
        "iou_sum": 0.0,
        "hit": {t: 0 for t in thresholds},   # ref boxes covered
        "prec": {t: 0 for t in thresholds},  # gen boxes matched
    })

    for r in rows:
        method = r.get("method", "?")
        gen = parse_boxes(r.get("generated", ""))
        ref = parse_boxes(r.get("reference_answer", "") or r.get("reference", ""))
        for key in (method, "OVERALL"):
            a = agg[key]
            a["n"] += 1
            a["samples_with_gen"] += 1 if gen else 0
            a["n_gen"] += len(gen)
            a["n_ref"] += len(ref)
            for rb in ref:
                bi = best_iou(rb, gen)
                a["iou_sum"] += bi
                for t in thresholds:
                    if bi >= t:
                        a["hit"][t] += 1
            for gb in gen:
                bi = best_iou(gb, ref)
                for t in thresholds:
                    if bi >= t:
                        a["prec"][t] += 1
    return agg, thresholds


def print_table(agg, thresholds, label):
    print(f"\n================  {label}  ================")
    hdr = (f"{'method':16s} {'n':>4} {'gen_cov':>7} {'gbox/i':>6} {'rbox/i':>6} "
           f"{'mIoU':>6} " + " ".join(f"hit@{t}".rjust(7) for t in thresholds) + " "
           + " ".join(f"prc@{t}".rjust(7) for t in thresholds))
    print(hdr)
    order = [k for k in sorted(agg) if k != "OVERALL"] + ["OVERALL"]
    for k in order:
        a = agg[k]
        n = a["n"] or 1
        nref = a["n_ref"] or 1
        ngen = a["n_gen"] or 1
        miou = a["iou_sum"] / nref
        row = (f"{k:16s} {a['n']:>4} {a['samples_with_gen']/n:>7.2f} "
               f"{a['n_gen']/n:>6.2f} {a['n_ref']/n:>6.2f} {miou:>6.3f} "
               + " ".join(f"{a['hit'][t]/nref:>7.3f}" for t in thresholds) + " "
               + " ".join(f"{a['prec'][t]/ngen:>7.3f}" for t in thresholds))
        print(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--compare", default=None, help="Second predictions.jsonl (same test set)")
    p.add_argument("--labels", nargs=2, default=["model", "compare"])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    args = p.parse_args()

    thr = tuple(args.thresholds)
    agg, _ = evaluate(args.pred, thr)
    print_table(agg, thr, args.labels[0])
    if args.compare:
        agg2, _ = evaluate(args.compare, thr)
        print_table(agg2, thr, args.labels[1])
        o1, o2 = agg["OVERALL"], agg2["OVERALL"]
        print(f"\nΔ (OVERALL {args.labels[0]} - {args.labels[1]}):")
        print(f"  mIoU:   {o1['iou_sum']/max(1,o1['n_ref']) - o2['iou_sum']/max(1,o2['n_ref']):+.3f}")
        for t in thr:
            d = o1['hit'][t]/max(1,o1['n_ref']) - o2['hit'][t]/max(1,o2['n_ref'])
            print(f"  hit@{t}: {d:+.3f}")

    print("\nLegenda: gen_cov=frac imgs com >=1 caixa | gbox/i,rbox/i=caixas por img "
          "(gerada/ref) | mIoU=IoU médio por caixa ref | hit@t=recall de artefatos "
          "| prc@t=precisão das caixas geradas.")


if __name__ == "__main__":
    main()
