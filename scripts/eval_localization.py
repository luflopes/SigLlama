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


def center(b):
    y1, x1, y2, x2 = b
    return ((y1 + y2) / 2.0, (x1 + x2) / 2.0)


def center_in(box, others) -> bool:
    """True if box's center lies inside any of the other boxes."""
    cy, cx = center(box)
    for oy1, ox1, oy2, ox2 in others:
        if oy1 <= cy <= oy2 and ox1 <= cx <= ox2:
            return True
    return False


def evaluate(path: str, thresholds=(0.3, 0.5), exclude=()):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    agg = defaultdict(lambda: {
        "n": 0, "samples_with_gen": 0,
        "n_gen": 0, "n_ref": 0,
        "iou_sum": 0.0,
        "hit": {t: 0 for t in thresholds},   # ref boxes covered (IoU)
        "prec": {t: 0 for t in thresholds},  # gen boxes matched (IoU)
        "chit": 0,   # ref boxes with a gen center inside (center-recall)
        "cprec": 0,  # gen boxes whose center is inside a ref (center-prec)
    })

    real = {"n": 0, "with_box": 0, "n_box": 0}
    for r in rows:
        method = r.get("method", "?")
        gen = parse_boxes(r.get("generated", ""))
        ref = parse_boxes(r.get("reference_answer", "") or r.get("reference", ""))
        is_real = (str(r.get("true_label", "")).lower() == "real") or bool(r.get("is_real"))
        if is_real:
            real["n"] += 1
            real["with_box"] += 1 if gen else 0
            real["n_box"] += len(gen)
        keys = [method]
        if method not in exclude:
            keys.append("OVERALL(fake)" if exclude else "OVERALL")
        for key in keys:
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
                if center_in(rb, gen) or any(center_in(g, [rb]) for g in gen):
                    a["chit"] += 1
            for gb in gen:
                bi = best_iou(gb, ref)
                for t in thresholds:
                    if bi >= t:
                        a["prec"][t] += 1
                if center_in(gb, ref):
                    a["cprec"] += 1
    return agg, real


def print_table(agg, thresholds, label):
    print(f"\n================  {label}  ================")
    hdr = (f"{'method':16s} {'n':>4} {'gen_cov':>7} {'gbox/i':>6} {'rbox/i':>6} "
           f"{'mIoU':>6} " + " ".join(f"hit@{t}".rjust(7) for t in thresholds)
           + f" {'cHit':>6} {'cPrc':>6} "
           + " ".join(f"prc@{t}".rjust(7) for t in thresholds))
    print(hdr)
    specials = ("OVERALL", "OVERALL(fake)")
    order = [k for k in sorted(agg) if k not in specials] + [k for k in specials if k in agg]
    for k in order:
        a = agg[k]
        n = a["n"] or 1
        nref = a["n_ref"] or 1
        ngen = a["n_gen"] or 1
        miou = a["iou_sum"] / nref
        row = (f"{k:16s} {a['n']:>4} {a['samples_with_gen']/n:>7.2f} "
               f"{a['n_gen']/n:>6.2f} {a['n_ref']/n:>6.2f} {miou:>6.3f} "
               + " ".join(f"{a['hit'][t]/nref:>7.3f}" for t in thresholds)
               + f" {a['chit']/nref:>6.3f} {a['cprec']/ngen:>6.3f} "
               + " ".join(f"{a['prec'][t]/ngen:>7.3f}" for t in thresholds))
        print(row)


def per_image_records(path, exclude):
    """Per-image (n_ref, iou_sum, hit50, chit) for fake images -> bootstrap."""
    recs = []
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("method", "?") in exclude:
            continue
        if str(r.get("true_label", "")).lower() == "real" or r.get("is_real"):
            continue
        gen = parse_boxes(r.get("generated", ""))
        ref = parse_boxes(r.get("reference_answer", "") or r.get("reference", ""))
        if not ref:
            continue
        iou_sum = sum(best_iou(rb, gen) for rb in ref)
        hit50 = sum(1 for rb in ref if best_iou(rb, gen) >= 0.5)
        chit = sum(1 for rb in ref
                   if center_in(rb, gen) or any(center_in(g, [rb]) for g in gen))
        recs.append((len(ref), iou_sum, hit50, chit))
    return recs


def bootstrap_ci(recs, B=2000, seed=42):
    import random
    rng = random.Random(seed)
    n = len(recs)

    def agg(sample):
        nref = sum(x[0] for x in sample) or 1
        return (sum(x[1] for x in sample) / nref,   # mIoU
                sum(x[2] for x in sample) / nref,   # hit@0.5
                sum(x[3] for x in sample) / nref)   # cHit
    point = agg(recs)
    dists = [[], [], []]
    for _ in range(B):
        sample = [recs[rng.randrange(n)] for _ in range(n)]
        for i, v in enumerate(agg(sample)):
            dists[i].append(v)
    out = {}
    for name, pt, d in zip(("mIoU", "hit@0.5", "cHit"), point, dists):
        d.sort()
        lo, hi = d[int(0.025 * B)], d[int(0.975 * B)]
        out[name] = (pt, lo, hi)
    return n, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--compare", default=None, help="Second predictions.jsonl (same test set)")
    p.add_argument("--labels", nargs=2, default=["model", "compare"])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    p.add_argument("--exclude-methods", nargs="*", default=[],
                   help="Métodos fora do agregado (ex.: Original). "
                        "Cria linha OVERALL(fake).")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="N reamostragens p/ IC 95%% (fakes). 0 = desliga.")
    args = p.parse_args()

    thr = tuple(args.thresholds)
    excl = tuple(args.exclude_methods)
    okey = "OVERALL(fake)" if excl else "OVERALL"

    def hallu(real, label):
        n = real["n"] or 1
        print(f"  [{label}] alucinação em reais: "
              f"{100*real['with_box']/n:.0f}% das {real['n']} imgs reais com >=1 caixa "
              f"(caixas/real={real['n_box']/n:.2f})")

    agg, real1 = evaluate(args.pred, thr, excl)
    print_table(agg, thr, args.labels[0])
    if args.compare:
        agg2, real2 = evaluate(args.compare, thr, excl)
        print_table(agg2, thr, args.labels[1])
        o1, o2 = agg[okey], agg2[okey]
        print(f"\nΔ ({okey}  {args.labels[0]} - {args.labels[1]}):")
        print(f"  mIoU:   {o1['iou_sum']/max(1,o1['n_ref']) - o2['iou_sum']/max(1,o2['n_ref']):+.3f}")
        for t in thr:
            d = o1['hit'][t]/max(1,o1['n_ref']) - o2['hit'][t]/max(1,o2['n_ref'])
            print(f"  hit@{t}: {d:+.3f}")
        dc = o1['chit']/max(1,o1['n_ref']) - o2['chit']/max(1,o2['n_ref'])
        print(f"  cHit:   {dc:+.3f}")

    print("\nAlucinação em imagens reais (menor = melhor):")
    hallu(real1, args.labels[0])
    if args.compare:
        hallu(real2, args.labels[1])

    if args.bootstrap:
        print(f"\nIC 95% por bootstrap (fakes, B={args.bootstrap}) — ponto [lo, hi]:")
        for lbl, path in ([(args.labels[0], args.pred)]
                          + ([(args.labels[1], args.compare)] if args.compare else [])):
            n, ci = bootstrap_ci(per_image_records(path, excl), B=args.bootstrap)
            print(f"  [{lbl}] (n_imgs={n})")
            for name, (pt, lo, hi) in ci.items():
                print(f"    {name:8s} {pt:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\nLegenda: gen_cov=frac imgs com >=1 caixa | gbox/i,rbox/i=caixas por img "
          "(gerada/ref) | mIoU=IoU médio por caixa ref | hit@t=recall de artefatos "
          "| prc@t=precisão das caixas geradas.")


if __name__ == "__main__":
    main()
