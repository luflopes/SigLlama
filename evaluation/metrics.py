"""Evaluation metrics for FaceGroundVLM.

Includes text-generation quality metrics (for explanations) and
classification metrics (for deepfake detection accuracy).

Dependencies::

    pip install nltk rouge-score bert-score sentence-transformers
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


# ------------------------------------------------------------------
# Text generation metrics
# ------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU-1..4 using NLTK."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    refs = [[ref.split()] for ref in references]
    hyps = [pred.split() for pred in predictions]
    smooth = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        scores[f"bleu_{n}"] = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth)

    return scores


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L using rouge-score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "rouge_1": float(np.mean(r1)),
        "rouge_2": float(np.mean(r2)),
        "rouge_l": float(np.mean(rl)),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BERTScore (precision, recall, F1)."""
    from bert_score import score as bert_score

    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_sentence_bert(predictions: list[str], references: list[str]) -> float:
    """Compute mean cosine similarity using Sentence-BERT."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = model.encode(predictions, convert_to_tensor=True)
    ref_emb = model.encode(references, convert_to_tensor=True)

    cos_sim = (pred_emb * ref_emb).sum(dim=1) / (
        pred_emb.norm(dim=1) * ref_emb.norm(dim=1) + 1e-8
    )
    return float(cos_sim.mean())


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """Compute mean METEOR score using NLTK."""
    import nltk
    from nltk.translate.meteor_score import meteor_score

    nltk.download("wordnet", quiet=True)

    scores = []
    for pred, ref in zip(predictions, references):
        s = meteor_score([ref.split()], pred.split())
        scores.append(s)
    return float(np.mean(scores))


def compute_cider(predictions: list[str], references: list[str]) -> float:
    """Compute CIDEr score (simplified TF-IDF based)."""
    from collections import defaultdict

    def _ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    doc_freq = defaultdict(int)
    N = len(references)
    for ref in references:
        seen = set()
        for n in range(1, 5):
            for ng in _ngrams(ref.split(), n):
                if ng not in seen:
                    doc_freq[ng] += 1
                    seen.add(ng)

    def _tfidf(text):
        tokens = text.split()
        vec = Counter()
        for n in range(1, 5):
            for ng in _ngrams(tokens, n):
                tf = 1.0 / max(len(tokens), 1)
                idf = np.log(max(N, 1) / (1.0 + doc_freq.get(ng, 0)))
                vec[ng] += tf * idf
        return vec

    scores = []
    for pred, ref in zip(predictions, references):
        v_pred = _tfidf(pred)
        v_ref = _tfidf(ref)

        if not v_pred or not v_ref:
            scores.append(0.0)
            continue

        all_keys = set(v_pred) | set(v_ref)
        dot = sum(v_pred.get(k, 0) * v_ref.get(k, 0) for k in all_keys)
        norm_p = np.sqrt(sum(v ** 2 for v in v_pred.values()))
        norm_r = np.sqrt(sum(v ** 2 for v in v_ref.values()))
        scores.append(dot / (norm_p * norm_r + 1e-8))

    return float(np.mean(scores)) * 10.0


# ------------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------------

def compute_detection_accuracy(
    predicted_labels: list[str], true_labels: list[str]
) -> float:
    """Binary accuracy for real/fake classification."""
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    return correct / len(true_labels) if true_labels else 0.0


def parse_real_fake(text: str) -> str:
    """Parse model output into a 'real'/'fake' label.

    The model is trained to open the answer with a verdict token
    (``Real.`` / ``Fake.``) followed by a justification that very often
    *mentions* the opposite word (``Real. The eyes look a bit fake ...``).
    Earlier revisions of this parser returned ``fake`` whenever the word
    appeared anywhere, which silently flipped correct ``real`` predictions.
    The current parser uses **first-occurrence** semantics: whichever of
    ``real``/``fake`` appears *first* (as a whole word) wins.
    """
    t = text.lower()
    fm = re.search(r"\bfake\b", t)
    rm = re.search(r"\breal\b", t)
    if fm and rm:
        return "fake" if fm.start() < rm.start() else "real"
    if fm:
        return "fake"
    if rm:
        return "real"
    return "unknown"


def parse_real_fake_legacy(text: str) -> str:
    """Legacy ``fake``-anywhere-wins parser kept for A/B comparison.

    Exposed so that ``evaluate.py`` can report both parsings in
    ``results.json`` without having to re-generate predictions.
    """
    t = text.lower()
    if re.search(r"\bfake\b", t):
        return "fake"
    if re.search(r"\breal\b", t):
        return "real"
    return "unknown"


def compute_detection_f1(
    predicted_labels: list[str], true_labels: list[str]
) -> dict[str, float]:
    """Compute precision, recall, and F1 for fake detection."""
    tp = fp = fn = tn = 0
    for pred, true in zip(predicted_labels, true_labels):
        if true == "fake":
            if pred == "fake":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "fake":
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------------------------------------------
# Localization metrics (Phase 4+)
# ------------------------------------------------------------------

_BBOX_TEXT_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
_BBOX_LOC_RE = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>")


def parse_bbox_text(text: str) -> list[tuple[int, int, int, int]]:
    """Extract bounding boxes from textual ``[y1,x1,y2,x2]`` sequences.

    The integer coordinates are in ``[0, 1000]`` (see
    ``scripts/create_loc_annotations.py``). Returns ``(y1, x1, y2, x2)`` tuples.
    """
    return [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in _BBOX_TEXT_RE.finditer(text)
    ]


def parse_loc_tokens(text: str) -> list[tuple[int, int, int, int]]:
    """Extract bounding boxes from either textual ``[y1,x1,y2,x2]`` (preferred)
    or legacy ``<locYYYY>`` PaliGemma2 sequences.

    Both formats return ``(y1, x1, y2, x2)`` tuples. Coordinate scales
    (1000 vs 1023) are close enough that a shared IoU threshold remains
    meaningful across formats.
    """
    boxes = parse_bbox_text(text)
    if boxes:
        return boxes
    return [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in _BBOX_LOC_RE.finditer(text)
    ]


def box_iou(box_a: tuple[int, ...], box_b: tuple[int, ...]) -> float:
    """Compute IoU between two (y1, x1, y2, x2) boxes."""
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_loc_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute localization metrics: mean IoU and region hit rate.

    Matches predicted bounding boxes to the closest reference box.
    """
    all_ious: list[float] = []
    hits = 0
    total_ref_boxes = 0

    for pred, ref in zip(predictions, references):
        pred_boxes = parse_loc_tokens(pred)
        ref_boxes = parse_loc_tokens(ref)
        if not ref_boxes:
            continue

        total_ref_boxes += len(ref_boxes)
        for rb in ref_boxes:
            if not pred_boxes:
                all_ious.append(0.0)
                continue
            best_iou = max(box_iou(pb, rb) for pb in pred_boxes)
            all_ious.append(best_iou)
            if best_iou > 0.3:
                hits += 1

    return {
        "loc_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "loc_hit_rate_03": hits / total_ref_boxes if total_ref_boxes > 0 else 0.0,
        "loc_total_ref_boxes": total_ref_boxes,
    }


# ------------------------------------------------------------------
# MoE analysis
# ------------------------------------------------------------------

def compute_expert_utilisation(
    expert_counts: dict[str, int], total_tokens: int
) -> dict[str, float]:
    """Per-expert utilisation fraction (for MoE analysis)."""
    return {k: v / total_tokens for k, v in expert_counts.items()}
