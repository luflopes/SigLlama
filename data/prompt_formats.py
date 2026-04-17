"""Backbone-aware prompt formatting for DD-VQA / LCS-558K samples.

PaliGemma2 uses a terse style: ``{question}\\n{answer}`` with no special
turn markers. The raw tokens are consumed directly by the LLM.

TinyLLaVA uses the standard LLaVA-v1 conversation template inherited from
TinyLlama-Chat: ``USER: {question} ASSISTANT: {answer}``. The prefix up to
and including ``ASSISTANT:`` is masked for loss computation.
"""
from __future__ import annotations

from typing import Any

import torch


def format_caption_prompt(caption: str, backbone: str) -> str:
    """Format an LCS-558K caption sample for either backbone."""
    backbone = backbone.lower()
    if backbone == "paligemma":
        return caption
    if backbone == "tinyllava":
        return (
            "USER: Describe the image briefly. "
            f"ASSISTANT: {caption}"
        )
    raise ValueError(f"Unknown backbone '{backbone}'")


def format_vqa_prompt(question: str, answer: str, backbone: str) -> tuple[str, str]:
    """Return ``(full_text, prefix_text)``.

    - ``full_text`` is tokenized for ``input_ids`` / ``labels``.
    - ``prefix_text`` tokens are masked (``labels = -100``) so the loss
      is only computed on the answer span.
    """
    backbone = backbone.lower()
    if backbone == "paligemma":
        full = f"{question}\n{answer}"
        prefix = f"{question}\n"
        return full, prefix
    if backbone == "tinyllava":
        prefix = f"USER: {question} ASSISTANT: "
        full = f"{prefix}{answer}"
        return full, prefix
    raise ValueError(f"Unknown backbone '{backbone}'")


def format_vqa_query(question: str, backbone: str) -> str:
    """Build only the question prefix (for ``generate``)."""
    backbone = backbone.lower()
    if backbone == "paligemma":
        return f"{question}\n"
    if backbone == "tinyllava":
        return f"USER: {question} ASSISTANT: "
    raise ValueError(f"Unknown backbone '{backbone}'")


def build_generation_inputs(
    questions: list[str],
    tokenizer: Any,
    backbone: str,
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenise a batch of query prefixes with LEFT-padding for generation.

    Causal LMs require left-padding during generation so that the *last*
    non-pad token of every row in the batch is the one the model continues
    from. Training uses right-padding; we swap the side temporarily here.

    Returns a dict with ``input_ids`` and ``attention_mask`` on CPU.
    """
    prefixes = [format_vqa_query(q, backbone) for q in questions]
    original_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = original_side
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
