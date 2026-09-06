#!/usr/bin/env python3
"""Audit label noise in DD-VQA (and the derived automatic grounding).

Produces citable, reproducible evidence that the DD-VQA annotations and the
lexical+landmark grounding pipeline are noisy/formulaic:

  (a) boxes placed on REAL faces by the automatic pipeline (should be ~none);
  (b) formulaic repetition of answers (uniqueness ratio, most repeated);
  (c) suspicious text patterns (no-face descriptions, beard/mustache mentions).

Exports CSVs of flagged cases for the appendix.

Usage::

    python scripts/audit_ddvqa_noise.py \\
        --data-dir label_studio/data \\
        --out-dir outputs/analysis/ddvqa_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

BOX = re.compile(r"\[\d+,\d+,\d+,\d+\]")
NOFACE = re.compile(r"no face|not a face|there is no face|no human|isn't a face|no visible face", re.I)
BEARD = re.compile(r"\b(beard|mustache|moustache|facial hair)\b", re.I)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(repo / "label_studio" / "data"))
    p.add_argument("--out-dir", default=str(repo / "outputs" / "analysis" / "ddvqa_audit"))
    p.add_argument("--loc-splits", nargs="+", default=["train_loc", "val_loc", "test_loc"])
    p.add_argument("--all-file", default="all.jsonl")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- (a) boxes on real faces ----------
    print("=== (a) Caixas automáticas em imagens REAIS ===")
    reals_rows = []
    tot_real = tot_boxed = 0
    for split in args.loc_splits:
        fp = data / f"{split}.jsonl"
        if not fp.is_file():
            print(f"  (pulado, ausente) {fp}")
            continue
        rows = load(fp)
        reals = [r for r in rows if r.get("is_real")]
        boxed = [r for r in reals if BOX.search(r.get("answer", ""))]
        tot_real += len(reals); tot_boxed += len(boxed)
        pct = 100 * len(boxed) / max(1, len(reals))
        print(f"  {split:9s} reais={len(reals):5d}  com_caixa={len(boxed):5d} ({pct:4.0f}%)")
        for r in boxed:
            reals_rows.append({
                "split": split, "image": r.get("image", ""), "method": r.get("method", ""),
                "n_boxes": len(BOX.findall(r.get("answer", ""))),
                "answer": r.get("answer", ""),
            })
    print(f"  TOTAL reais={tot_real}  com_caixa={tot_boxed} "
          f"({100*tot_boxed/max(1,tot_real):.0f}%)")

    # ---------- (b) formulaic repetition ----------
    print("\n=== (b) Repetição formulaica (all.jsonl) ===")
    allrows = load(data / args.all_file)
    ans = Counter(r.get("answer", "").strip() for r in allrows)
    tot, uniq = len(allrows), len(ans)
    print(f"  total={tot}  únicas={uniq}  ({100*uniq/tot:.1f}% únicas)")
    print(f"  top-{args.top} respostas repetidas:")
    for a, c in ans.most_common(args.top):
        print(f"   {c:4d}x  «{a[:80]}»")

    # ---------- (c) suspicious patterns ----------
    print("\n=== (c) Padrões suspeitos (all.jsonl) ===")
    flagged = []
    for r in allrows:
        a = r.get("answer", "")
        tags = []
        if NOFACE.search(a):
            tags.append("no_face")
        if BEARD.search(a):
            tags.append("beard/mustache")
        for t in tags:
            flagged.append({
                "image": r.get("image", ""), "method": r.get("method", ""),
                "is_real": r.get("is_real", ""), "pattern": t, "answer": a,
            })
    cnt = Counter(f["pattern"] for f in flagged)
    print(f"  no_face={cnt.get('no_face',0)}  beard/mustache={cnt.get('beard/mustache',0)}")

    # ---------- export ----------
    def dump(name, rows, cols):
        fp = out / name
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {fp}  ({len(rows)} linhas)")

    print("\n=== Exports ===")
    dump("reals_with_boxes.csv", reals_rows, ["split", "image", "method", "n_boxes", "answer"])
    dump("flagged_text.csv", flagged, ["image", "method", "is_real", "pattern", "answer"])
    dup_rows = [{"count": c, "answer": a} for a, c in ans.most_common() if c > 1]
    dump("duplicated_answers.csv", dup_rows, ["count", "answer"])

    summary = {
        "reals_total": tot_real, "reals_boxed": tot_boxed,
        "reals_boxed_pct": round(100 * tot_boxed / max(1, tot_real), 1),
        "answers_total": tot, "answers_unique": uniq,
        "answers_unique_pct": round(100 * uniq / tot, 1),
        "no_face": cnt.get("no_face", 0), "beard_mustache": cnt.get("beard/mustache", 0),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  -> {out/'summary.json'}")


if __name__ == "__main__":
    main()
