"""DD-VQA JSONL dataset for Stage 2+ fine-tuning (from scripts/prepare_ddvqa.py).

Backbone-agnostic: receives tokenizer and image_processor separately plus
an explicit ``backbone`` flag so the prompt formatting matches the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .prompt_formats import format_vqa_prompt

logger = logging.getLogger(__name__)


def _sample_is_real(row: dict) -> bool:
    """Robustly read the real/fake flag from a DD-VQA row.

    The prepare script writes ``is_real`` (boolean). Older/manual metadata
    files may use a string ``label`` field with values like "real"/"fake".
    """
    if "is_real" in row:
        return bool(row["is_real"])
    return str(row.get("label", "")).lower() == "real"


def _sample_label_str(row: dict) -> str:
    return "real" if _sample_is_real(row) else "fake"


# ``Real. `` / ``Fake. `` — uppercase with a period + single space, so that
# SentencePiece (TinyLlama) keeps the verdict as a single token followed by
# ``.`` and the justification after a space boundary.
_VERDICT_PREFIX_RE = re.compile(r"^\s*(real|fake)\s*[.,:;!?-]*\s*", re.IGNORECASE)


def _normalize_answer_with_verdict(answer: str, is_real: bool) -> str:
    """Ensure the answer starts with ``Real. `` or ``Fake. `` exactly.

    If the answer already opens with a real/fake token in any casing /
    punctuation, that token is stripped and replaced by the canonical
    verdict prefix. If no opening verdict is present, the prefix is
    prepended. Either way, downstream tokenisation sees the same verdict
    position on every sample, which is required for the weighted
    first-token loss (``first_token_loss_weight``) to make sense.
    """
    verdict = "Real" if is_real else "Fake"
    body = _VERDICT_PREFIX_RE.sub("", answer or "").lstrip()
    return f"{verdict}. {body}" if body else f"{verdict}."


class DDVQADataset(Dataset):
    """JSONL with ``image``, ``question``, ``answer``, ``is_real``/``label``, ``method``."""

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer: Any,
        image_processor: Any,
        dino_transform: Any,
        backbone: str = "paligemma",
        max_length: int = 256,
        enforce_verdict_prefix: bool = True,
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dino_transform = dino_transform
        self.backbone = backbone
        self.max_length = max_length
        self.enforce_verdict_prefix = enforce_verdict_prefix

        self.samples: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_real = sum(1 for s in self.samples if _sample_is_real(s))
        n_fake = len(self.samples) - n_real
        logger.info(
            "Loaded %d samples from %s (real=%d, fake=%d, backbone=%s)",
            len(self.samples), metadata_path, n_real, n_fake, backbone,
        )

        if enforce_verdict_prefix:
            rewrites = 0
            already_ok = 0
            for s in self.samples:
                orig = s.get("answer", "") or ""
                norm = _normalize_answer_with_verdict(orig, _sample_is_real(s))
                if norm != orig:
                    s["answer"] = norm
                    rewrites += 1
                else:
                    already_ok += 1
            logger.info(
                "Verdict-prefix normalisation: rewritten=%d, already_ok=%d "
                "(total=%d)",
                rewrites, already_ok, len(self.samples),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def is_real(self, idx: int) -> bool:
        return _sample_is_real(self.samples[idx])

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        row = self.samples[idx]
        img_rel = row.get("image", "")
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(self.image_root, img_rel)
        if not os.path.isfile(img_path):
            return None

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        question = row.get("question", "")
        answer = row.get("answer", "")
        image_id = img_rel if img_rel else os.path.basename(img_path)
        prefix_text, answer_text = format_vqa_prompt(
            question, answer, self.backbone,
        )

        pixel_values_siglip = self.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # Two-step tokenisation (prefix WITH specials, answer WITHOUT) so
        # that ``full_ids[:prefix_len] == prefix_ids`` is guaranteed by
        # construction. Single-string tokenisation would otherwise fuse
        # whitespace at the prefix/answer boundary into a different token
        # sequence, causing ``labels[prefix_len-1]`` to silently mask the
        # verdict (Real/Fake) we explicitly want supervised.
        prefix_ids: list[int] = self.tokenizer(
            prefix_text, add_special_tokens=True,
        )["input_ids"]
        answer_ids: list[int] = self.tokenizer(
            answer_text, add_special_tokens=False,
        )["input_ids"]

        full_list = list(prefix_ids) + list(answer_ids)
        if len(full_list) > self.max_length:
            full_list = full_list[: self.max_length]
        prefix_len = min(len(prefix_ids), self.max_length)

        real_len = len(full_list)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        full_list = full_list + [pad_id] * (self.max_length - real_len)

        input_ids = torch.tensor(full_list, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:real_len] = 1

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        pixel_values_dino = self.dino_transform(img)

        label_str = _sample_label_str(row)
        method = row.get("method", "unknown")

        return {
            "pixel_values_siglip": pixel_values_siglip,
            "pixel_values_dino": pixel_values_dino,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_str": label_str,
            "method": method,
            "image": image_id,
            "question": question,
            "answer": answer,
        }


def collate_ddvqa(batch: list) -> dict[str, Any] | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    tensor_keys = [
        "pixel_values_siglip", "pixel_values_dino",
        "input_ids", "attention_mask", "labels",
    ]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    for meta_key in ("label_str", "method", "image", "question", "answer"):
        out[meta_key] = [b[meta_key] for b in batch]
    return out
