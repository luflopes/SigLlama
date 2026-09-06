#!/usr/bin/env python3
"""Build the cleaned + gold-merged training sets for the revised pipeline.

Given the base DD-VQA text splits, the cleaned localization splits
(``create_loc_annotations.py --no-ground-question --skip-real-boxes``) and the
manual gold splits, produce:

Passo 2 (A3, texto) -- clean text, gold train/val answers substituted by the
  human ``answer_original`` (no boxes), gold oversampled:
    train_text_gold.jsonl, val_text_gold.jsonl, test_text_gold.jsonl

Passo 3 (A4, localização) -- clean loc, gold train/val substituted by the human
  ``answer`` (WITH boxes), gold oversampled:
    train_loc_final.jsonl, val_loc_final.jsonl
  (test = data/ddvqa_gold/test_loc_gold.jsonl for fine metrics;
   test_loc_clean.jsonl already exists for large-n calibration/text.)

Join key is the image filename. Substitution keeps every base field and only
swaps the answer (and, for loc, answer_original/boxes/grounded_regions).

Usage::

    python scripts/build_clean_datasets.py --gold-oversample 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_by_key(rows: list[dict]) -> dict[tuple, dict]:
    """Key by (image, question): gold only covers the GLOBAL question, so we
    must not overwrite the region-specific questions of the same image."""
    return {(r["image"], r.get("question", "")): r for r in rows}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(repo / "label_studio" / "data"))
    ap.add_argument("--clean-dir", default=str(repo / "data" / "ddvqa_clean"))
    ap.add_argument("--gold-dir", default=str(repo / "data" / "ddvqa_gold"))
    ap.add_argument("--out-dir", default=str(repo / "data" / "ddvqa_clean"))
    ap.add_argument("--gold-oversample", type=int, default=3,
                    help="total occurrences of each gold train row (>=1)")
    ap.add_argument("--drop-region-questions", action="store_true",
                    help="keep ONLY the global question rows (drops skin/eyebrows/"
                         "eyes/mouth/nose Q/A). Attacks formulaic templates but "
                         "shrinks the mass.")
    args = ap.parse_args()

    base, clean, gold, out = (Path(args.base_dir), Path(args.clean_dir),
                              Path(args.gold_dir), Path(args.out_dir))
    K = max(1, args.gold_oversample)
    GLOBAL_Q = "Does the image looks real/fake?"

    gold_tr = index_by_key(load(gold / "train_loc_gold.jsonl"))
    gold_va = index_by_key(load(gold / "val_loc_gold.jsonl"))
    gold_te = load(gold / "test_loc_gold.jsonl")

    def substitute(base_rows, gold_idx, fields, oversample):
        """Return rows with gold substitution (matched by image+question) +
        oversampling of substituted (gold) rows."""
        if args.drop_region_questions:
            base_rows = [r for r in base_rows if r.get("question") == GLOBAL_Q]
        out_rows, subbed, extra = [], 0, 0
        for r in base_rows:
            key = (r.get("image"), r.get("question", ""))
            if key in gold_idx:
                g = gold_idx[key]
                r = dict(r)
                for dst, src in fields.items():
                    r[dst] = g.get(src, r.get(dst))
                subbed += 1
                out_rows.append(r)
                if oversample:
                    for _ in range(K - 1):
                        out_rows.append(dict(r)); extra += 1
            else:
                out_rows.append(r)
        miss = len(gold_idx) - subbed
        return out_rows, subbed, extra, miss

    # ---------- Passo 2: TEXTO (A3) ----------
    base_tr = load(base / "train.jsonl")
    base_va = load(base / "val.jsonl")
    # gold text = human answer_original (sem caixas) vai para o campo answer
    tr_txt, s, e, m = substitute(base_tr, gold_tr, {"answer": "answer_original"}, oversample=True)
    print(f"[texto] train: base={len(base_tr)} gold_sub={s} oversample_extra={e} miss={m} -> {len(tr_txt)}")
    va_txt, s, e, m = substitute(base_va, gold_va, {"answer": "answer_original"}, oversample=False)
    print(f"[texto] val:   base={len(base_va)} gold_sub={s} miss={m} -> {len(va_txt)}")
    te_txt = [{**r, "answer": r.get("answer_original", r.get("answer", ""))} for r in gold_te]
    write(out / "train_text_gold.jsonl", tr_txt)
    write(out / "val_text_gold.jsonl", va_txt)
    write(out / "test_text_gold.jsonl", te_txt)
    print(f"[texto] test (gold, answer_original) -> {len(te_txt)}")

    # ---------- Passo 3: LOCALIZAÇÃO (A4) ----------
    clean_tr = load(clean / "train_loc_clean.jsonl")
    clean_va = load(clean / "val_loc_clean.jsonl")
    loc_fields = {"answer": "answer", "answer_original": "answer_original",
                  "boxes": "boxes", "grounded_regions": "grounded_regions"}
    tr_loc, s, e, m = substitute(clean_tr, gold_tr, loc_fields, oversample=True)
    print(f"[loc]   train: base={len(clean_tr)} gold_sub={s} oversample_extra={e} miss={m} -> {len(tr_loc)}")
    va_loc, s, e, m = substitute(clean_va, gold_va, loc_fields, oversample=False)
    print(f"[loc]   val:   base={len(clean_va)} gold_sub={s} miss={m} -> {len(va_loc)}")
    write(out / "train_loc_final.jsonl", tr_loc)
    write(out / "val_loc_final.jsonl", va_loc)
    print(f"[loc]   test fino = {gold/'test_loc_gold.jsonl'} | test n-grande = {clean/'test_loc_clean.jsonl'}")

    print("\nOK. Copie data/ddvqa_clean/*.jsonl para a tarkin em "
          "/datasets/deepfake/ddvqa_prepared/clean/ (mesmo image_root/frames).")


if __name__ == "__main__":
    main()
