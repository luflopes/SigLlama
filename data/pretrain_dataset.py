"""Image-caption dataset for Stage 1 adapter pre-training.

Loads images and captions from LCS-558K (or any dataset with the same
metadata format).  No soft tokens or landmarks at this stage.

Supports two training modes controlled by ``use_prompt``:

**Stage 1a — Alignment** (``use_prompt=False``, default)::

    [visual_patches]  <s> {caption} </s> [pad...]
                      BOS=−100  |--predicted--|

    The model learns to map visual features to text by predicting
    the raw BLIP caption.  Use with ``*_meta.json``.

**Stage 1b — Instruction** (``use_prompt=True``)::

    [visual_patches]  <s> {prompt}\n{caption} </s> [pad...]
                      |--- label=−100 ---|--- predicted ---|

    The model learns to follow instructions.  The loss is computed
    only on the caption (answer) tokens.  Use with the conversations
    JSON that contains human/gpt turn pairs.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe the image."


def _extract_caption(item: dict) -> str:
    """Extract caption from either metadata format (alignment mode)."""
    if "blip_caption" in item:
        return item["blip_caption"]
    if "conversations" in item:
        for turn in item["conversations"]:
            if turn.get("from") == "gpt":
                return turn.get("value", "")
    return item.get("caption", "")


def _extract_prompt_and_caption(item: dict) -> tuple[str, str]:
    """Extract (prompt, caption) from either metadata format (instruction mode)."""
    if "conversations" in item:
        prompt = ""
        caption = ""
        for turn in item["conversations"]:
            if turn.get("from") == "human":
                prompt = turn.get("value", "")
                prompt = prompt.replace("<image>", "").replace("\n", " ").strip()
            elif turn.get("from") == "gpt":
                caption = turn.get("value", "")
        return (prompt or DEFAULT_PROMPT, caption)

    caption = item.get("blip_caption", item.get("caption", ""))
    return (DEFAULT_PROMPT, caption)


class PretrainDataset(Dataset):
    """Image + text dataset for visual-language alignment / instruction.

    Parameters
    ----------
    metadata_path : str
        Path to the JSON metadata file.
    image_root : str
        Root directory that ``image`` paths are relative to.
    processor : SiglipImageProcessor
        HuggingFace image processor for SigLIP.
    tokenizer : PreTrainedTokenizer
        LLM tokenizer (e.g. LlamaTokenizer).
    max_length : int
        Maximum total tokens for the text sequence.
    use_prompt : bool
        If *False* (Stage 1a), only the caption is used and the entire
        sequence (except BOS and padding) is predicted.  If *True*
        (Stage 1b), the human instruction is prepended and only the
        caption tokens are predicted.
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        processor: Any,
        tokenizer: Any,
        max_length: int = 128,
        use_prompt: bool = False,
    ):
        self.image_root = image_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_prompt = use_prompt

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: list[dict[str, str]] = []
        skipped = 0
        for item in raw:
            img_rel = item.get("image", "")
            img_path = os.path.join(image_root, img_rel)
            if not os.path.isfile(img_path):
                skipped += 1
                continue

            if use_prompt:
                prompt, caption = _extract_prompt_and_caption(item)
            else:
                prompt, caption = "", _extract_caption(item)

            if not caption:
                skipped += 1
                continue

            self.samples.append({
                "image_path": img_path,
                "prompt": prompt,
                "caption": caption,
            })

        mode_str = "instruction (use_prompt=True)" if use_prompt else "alignment (use_prompt=False)"
        logger.info(
            "Loaded %d samples [%s] from %s (skipped %d)",
            len(self.samples), mode_str, metadata_path, skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None:
        sample = self.samples[idx]

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        prompt = sample["prompt"]
        caption = sample["caption"]

        if self.use_prompt and prompt:
            full_text = f"{prompt}\n{caption}"

            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]
            # +1 accounts for BOS prepended by the tokenizer
            prompt_len = len(prompt_ids) + 1
        else:
            full_text = caption
            prompt_len = 1  # only BOS is masked

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # padding
        labels[:prompt_len] = -100           # BOS (+ prompt tokens when applicable)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_skip_none(
    batch: list[dict[str, torch.Tensor] | None],
) -> dict[str, torch.Tensor] | None:
    """Stack a batch of samples, silently dropping ``None`` entries."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
