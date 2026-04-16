"""Backbone-aware prompt formatting for DD-VQA / LCS-558K samples.

PaliGemma2 uses a terse style: ``{question}\\n{answer}`` with no special
turn markers. The raw tokens are consumed directly by the LLM.

TinyLLaVA uses the standard LLaVA-v1 conversation template inherited from
TinyLlama-Chat: ``USER: {question} ASSISTANT: {answer}``. The prefix up to
and including ``ASSISTANT:`` is masked for loss computation.
"""
from __future__ import annotations


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
