#!/usr/bin/env python3
"""Label-free text-quality metrics for generated answers (formulaic/template).

Runs on a large ``predictions.jsonl`` (e.g. DD-VQA test) so the numbers are
stable regardless of the small gold test. Reports:

  - unique-answer ratio, top repeated answers, coverage by top-K
  - distinct-1 / distinct-2 (n-gram diversity)
  - formulaic-template hit rate (known DD-VQA phrasings)
  - verdict accuracy (real/fake) vs true_label   [needs true_label]
  - hallucination on reals: frac of real images with >=1 box, boxes/real

Usage::

    python scripts/eval_text_quality.py --pred A/predictions.jsonl \\
        [--compare B/predictions.jsonl --labels new old]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BOX = re.compile(r"\[\d+,\d+,\d+,\d+\]")
TEMPLATES = [
    re.compile(r"the person'?s \w+ looks? (a bit |very )?(real|fake)", re.I),
    re.compile(r"the person has naturally", re.I),
    re.compile(r"arched eyebrows, round eyes, straight nose, full mouth", re.I),
    re.compile(r"the person have complete face feature", re.I),
    re.compile(r"it is an image with manipulated face regions", re.I),
]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def strip_boxes(t: str) -> str:
    return BOX.sub("", t or "")


def verdict(text: str) -> str | None:
    m = re.match(r"\s*(real|fake)\b", text or "", re.I)
    if m:
        return m.group(1).lower()
    t = (text or "").lower()
    if "looks fake" in t or "looks real" in t:
        return "fake" if "looks fake" in t else "real"
    return None


def distinct_n(texts, n):
    grams, total = set(), 0
    for t in texts:
        toks = strip_boxes(t).lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / max(1, total)


def analyze(rows: list[dict], topk: int = 5) -> dict:
    gens = [r.get("generated", "") for r in rows]
    n = len(gens)
    c = Counter(g.strip() for g in gens)
    top = c.most_common(topk)
    cover = sum(cnt for _, cnt in top) / max(1, n)
    tmpl = sum(1 for g in gens if any(p.search(g) for p in TEMPLATES)) / max(1, n)

    # verdict accuracy
    vt = [(verdict(r.get("generated", "")), str(r.get("true_label", "")).lower())
          for r in rows]
    vt = [(g, t) for g, t in vt if t in ("real", "fake")]
    vacc = (sum(1 for g, t in vt if g == t) / len(vt)) if vt else float("nan")

    # hallucination on reals
    reals = [r for r in rows if str(r.get("true_label", "")).lower() == "real"
             or r.get("is_real")]
    rb = sum(1 for r in reals if BOX.search(r.get("generated", "")))
    rboxes = sum(len(BOX.findall(r.get("generated", ""))) for r in reals)
    return {
        "n": n,
        "unique_ratio": len(c) / max(1, n),
        "distinct1": distinct_n(gens, 1),
        "distinct2": distinct_n(gens, 2),
        "template_rate": tmpl,
        "top_cover": cover,
        "top": top,
        "verdict_acc": vacc,
        "reals_n": len(reals),
        "reals_box_rate": (rb / max(1, len(reals))),
        "reals_boxes_per": (rboxes / max(1, len(reals))),
    }


def show(s: dict, label: str) -> None:
    print(f"\n===== {label} (n={s['n']}) =====")
    print(f"  unique-answer ratio : {s['unique_ratio']*100:5.1f}%")
    print(f"  distinct-1 / -2     : {s['distinct1']:.3f} / {s['distinct2']:.3f}")
    print(f"  template hit rate   : {s['template_rate']*100:5.1f}%   (menor = melhor)")
    print(f"  top-{len(s['top'])} cobrem       : {s['top_cover']*100:5.1f}% das respostas")
    print(f"  verdict accuracy    : {s['verdict_acc']*100:5.1f}%")
    print(f"  alucinação reais    : {s['reals_box_rate']*100:5.1f}% imgs c/ caixa "
          f"(caixas/real={s['reals_boxes_per']:.2f}, n={s['reals_n']})")
    print("  respostas mais repetidas:")
    for a, cnt in s["top"]:
        print(f"    {cnt:5d}x  «{a[:80]}»")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    s1 = analyze(load(Path(args.pred)), args.topk)
    show(s1, args.labels[0])
    if args.compare:
        s2 = analyze(load(Path(args.compare)), args.topk)
        show(s2, args.labels[1])
        print(f"\nΔ ({args.labels[0]} - {args.labels[1]}):")
        print(f"  unique_ratio : {(s1['unique_ratio']-s2['unique_ratio'])*100:+.1f} pp")
        print(f"  distinct-2   : {s1['distinct2']-s2['distinct2']:+.3f}")
        print(f"  template     : {(s1['template_rate']-s2['template_rate'])*100:+.1f} pp (negativo = melhor)")
        print(f"  verdict_acc  : {(s1['verdict_acc']-s2['verdict_acc'])*100:+.1f} pp")
        print(f"  aluc. reais  : {(s1['reals_box_rate']-s2['reals_box_rate'])*100:+.1f} pp (negativo = melhor)")


if __name__ == "__main__":
    main()
