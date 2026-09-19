#!/usr/bin/env python3
"""Recomputa métricas textuais com e sem tokens de localização (caixas).

Motivação: os modelos de localização (A4, A4 Gold, A6) emitem coordenadas inline
(``[y1,x1,y2,x2]``) no texto gerado, e a pipeline de avaliação computa BLEU/ROUGE/
CIDEr sobre o texto cru. Como as referências do DD-VQA também contêm coordenadas
e o casamento exato é raro, os tokens de caixa deprimem sistematicamente essas
métricas, tornando a comparação com os modelos sem localização (A1–A3) desigual.

Este script reprocessa os ``predictions.jsonl`` já salvos localmente (sem
re-inferência) e reporta, para cada configuração, as métricas textuais na versão
crua e na versão "só linguagem" (caixas removidas de predição e referência).

Métricas reportadas no quadro do capítulo: BLEU-1, BLEU-4, ROUGE-L, CIDEr.
  - BLEU e CIDEr usam as mesmas funções de ``evaluation.metrics`` (reproduzem os
    números salvos em ``results.json``).
  - ROUGE-L é implementado aqui (LCS + stemming de Porter) porque a lib
    ``rouge_score`` não está disponível neste ambiente; a fidelidade é validada
    contra o ``rouge_l`` salvo em ``results.json`` (versão crua).

Uso::

    python scripts/recompute_text_metrics.py
    python scripts/recompute_text_metrics.py --only-global   # só a pergunta global
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_bleu, compute_cider, strip_loc_tokens  # noqa: E402

ABLATION = ROOT / "outputs" / "ablation"
OUT_JSON = ROOT / "outputs" / "analysis" / "text_metrics_loc_stripped.json"
GLOBAL_Q = "Does the image looks real/fake?"

# (rótulo, diretório, split) — espelha o mapeamento do notebook de análise.
EXPERIMENTS = [
    ("A1 (Baseline SigLIP)", "g1_baseline", "s2_best_test"),
    ("A2 (I-MoF)", "g2_imof", "s2_best_test"),
    ("A3 (DINOv2 LoRA)", "g3_lora", "best_test"),
    ("A3 Gold", "g3_clean_gold", "cleantest"),
    ("A4 (LoRA + Loc)", "g4_auto_clean", "cleantest"),
    ("A4 Gold", "g4_final", "cleantest"),
    ("A5 (Classifier)", "g5_classifier", "best_test"),
    # A6 = gerador da A4 limpa (g4_auto_clean) + veredito do classificador,
    # reavaliado no protocolo limpo (cleantest). Ver scripts/eval_a6_clean.sh.
    ("A6 (Full)", "g6_full", "cleantest"),
]

_STEM_CACHE: dict[str, str] = {}


def _porter():
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer()


_STEMMER = _porter()


def _tokenize_stem(text: str) -> list[str]:
    """Tokeniza como o rouge_score (minúsculas, alfanumérico) e aplica Porter."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = []
    for t in toks:
        s = _STEM_CACHE.get(t)
        if s is None:
            s = _STEMMER.stem(t)
            _STEM_CACHE[t] = s
        out.append(s)
    return out


def _lcs(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F-measure médio (sentence-level, LCS, stemming de Porter)."""
    total = 0.0
    for pred, ref in zip(predictions, references):
        p = _tokenize_stem(pred)
        r = _tokenize_stem(ref)
        if not p or not r:
            continue
        l = _lcs(p, r)
        if l == 0:
            continue
        prec = l / len(p)
        rec = l / len(r)
        total += 2 * prec * rec / (prec + rec)
    return total / max(1, len(predictions))


def text_metrics(preds: list[str], refs: list[str]) -> dict:
    bleu = compute_bleu(preds, refs)
    try:
        cider = compute_cider(preds, refs)
    except Exception:
        cider = float("nan")
    return {
        "bleu_1": bleu.get("bleu_1"),
        "bleu_4": bleu.get("bleu_4"),
        "rouge_l": rouge_l(preds, refs),
        "cider": cider,
    }


def load_rows(pred_path: Path, only_global: bool) -> list[dict]:
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only_global:
        rows = [r for r in rows if r.get("question") == GLOBAL_Q]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only-global", action="store_true",
                    help="Restringe à pergunta global 'Does the image looks real/fake?'.")
    args = ap.parse_args()

    report = {}
    print(f"{'Config':22s} | {'BLEU-1':>15s} | {'BLEU-4':>15s} | {'ROUGE-L':>15s} | {'CIDEr':>15s}")
    print(f"{'':22s} | {'cru→limpo':>15s} | {'cru→limpo':>15s} | {'cru→limpo':>15s} | {'cru→limpo':>15s}")
    print("-" * 100)

    for label, exp_dir, split in EXPERIMENTS:
        pred_path = ABLATION / exp_dir / "evaluation" / split / "predictions.jsonl"
        if not pred_path.exists():
            print(f"{label:22s} | (predictions.jsonl ausente: {pred_path.relative_to(ROOT)})")
            continue

        rows = load_rows(pred_path, args.only_global)
        gen = [r.get("generated", "") for r in rows]
        ref = [r.get("reference_answer", r.get("answer", "")) for r in rows]

        raw = text_metrics(gen, ref)
        clean = text_metrics([strip_loc_tokens(x) for x in gen],
                             [strip_loc_tokens(x) for x in ref])

        # Fração com caixas (para contexto).
        box_re = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]|(?:<loc\d{4}>){4}")
        gen_box = sum(1 for g in gen if box_re.search(g)) / max(1, len(gen))

        # Validação: rouge_l cru vs armazenado em results.json (mesmo split).
        stored = None
        res_path = pred_path.parent / "results.json"
        if res_path.exists() and not args.only_global:
            stored = json.loads(res_path.read_text()).get("rouge_l")

        report[label] = {
            "dir": exp_dir, "split": split, "n": len(rows),
            "frac_pred_com_caixa": round(gen_box, 4),
            "raw": raw, "stripped": clean,
            "rouge_l_stored": stored,
        }

        def cell(a, b):
            return f"{a:.3f}→{b:.3f}"

        val = ""
        if stored is not None:
            val = f"  [rougeL cru vs salvo: {raw['rouge_l']:.3f} vs {stored:.3f}]"
        print(f"{label:22s} | {cell(raw['bleu_1'], clean['bleu_1']):>15s} | "
              f"{cell(raw['bleu_4'], clean['bleu_4']):>15s} | "
              f"{cell(raw['rouge_l'], clean['rouge_l']):>15s} | "
              f"{cell(raw['cider'], clean['cider']):>15s}{val}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON salvo em {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
