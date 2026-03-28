"""Evaluation metrics for SigLlama.

Includes text-generation quality metrics (for explanations) and
classification metrics (for deepfake detection accuracy).
"""
from __future__ import annotations


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score."""
    # TODO: implement with sacrebleu or nltk
    raise NotImplementedError


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    # TODO: implement with rouge_score
    raise NotImplementedError


def compute_detection_accuracy(
    predicted_labels: list[str], true_labels: list[str]
) -> float:
    """Binary accuracy for real/fake classification."""
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    return correct / len(true_labels) if true_labels else 0.0


def compute_expert_utilisation(
    expert_counts: dict[str, int], total_tokens: int
) -> dict[str, float]:
    """Per-expert utilisation fraction (for MoE analysis)."""
    return {k: v / total_tokens for k, v in expert_counts.items()}
