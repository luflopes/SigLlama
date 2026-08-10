#!/usr/bin/env python3
"""Translate Chinese text in LOKI video metadata to English using MarianMT.

Reads metadata_videos.json and produces metadata_videos_en.json with all
Chinese text fields translated to English.

Usage::

    python scripts/translate_loki_metadata.py \
        --input ./outputs/loki_dataset/metadata_videos.json \
        --output ./outputs/loki_dataset/metadata_videos_en.json

    # Also translate image metadata:
    python scripts/translate_loki_metadata.py \
        --input ./outputs/loki_dataset/metadata_images.json \
        --output ./outputs/loki_dataset/metadata_images_en.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time

from tqdm import tqdm

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def load_model():
    """Load MarianMT zh->en model."""
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-zh-en"
    log.info("Loading model: %s", model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    log.info("Model loaded successfully")
    return tokenizer, model


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))


def translate_batch(texts: list[str], tokenizer, model, max_length: int = 512) -> list[str]:
    """Translate a batch of texts from Chinese to English."""
    if not texts:
        return []

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    translated = model.generate(**inputs, max_length=max_length)
    results = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return results


def collect_texts_from_videos(data: dict) -> list[tuple[list, str | int, str]]:
    """Collect all translatable text fields from video metadata.

    Returns list of (path_to_field, key, original_text) for replacement.
    """
    texts_to_translate = []

    for i, sample in enumerate(data.get("samples", [])):
        # overall_description
        od = sample.get("overall_description", "")
        if od and contains_chinese(od):
            texts_to_translate.append((["samples", i], "overall_description", od))

        # segment_annotations[].description
        for j, seg in enumerate(sample.get("segment_annotations", [])):
            desc = seg.get("description", "")
            if desc and contains_chinese(desc):
                texts_to_translate.append(
                    (["samples", i, "segment_annotations", j], "description", desc)
                )

        # frames[].description
        for k, frame in enumerate(sample.get("frames", [])):
            desc = frame.get("description", "")
            if desc and contains_chinese(desc):
                texts_to_translate.append(
                    (["samples", i, "frames", k], "description", desc)
                )

    return texts_to_translate


def collect_texts_from_images(data: dict) -> list[tuple[list, str | int, str]]:
    """Collect all translatable text fields from image metadata."""
    texts_to_translate = []

    for i, sample in enumerate(data.get("samples", [])):
        # global_description
        gd = sample.get("global_description", "")
        if gd and contains_chinese(gd):
            texts_to_translate.append((["samples", i], "global_description", gd))

        # regions[].description
        for j, region in enumerate(sample.get("regions", [])):
            desc = region.get("description", "")
            if desc and contains_chinese(desc):
                texts_to_translate.append(
                    (["samples", i, "regions", j], "description", desc)
                )

    return texts_to_translate


def set_nested(data: dict, path: list, key: str, value: str) -> None:
    """Set a value in a nested dict/list structure."""
    obj = data
    for p in path:
        obj = obj[p]
    obj[key] = value


def translate_metadata(input_path: str, output_path: str, batch_size: int = 32):
    """Main translation pipeline."""
    log.info("Loading metadata: %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect type (videos or images) and collect texts
    if "samples" in data and len(data["samples"]) > 0:
        first = data["samples"][0]
        if "frames" in first or "segment_annotations" in first:
            log.info("Detected video metadata format")
            texts_info = collect_texts_from_videos(data)
        else:
            log.info("Detected image metadata format")
            texts_info = collect_texts_from_images(data)
    else:
        log.warning("No samples found in metadata")
        return

    if not texts_info:
        log.info("No Chinese text found to translate")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    log.info("Found %d text fields to translate", len(texts_info))

    # Load model
    tokenizer, model = load_model()

    # Split multiline texts into individual lines for better translation
    # then rejoin after translation
    all_texts = [info[2] for info in texts_info]

    # Translate in batches
    translated_texts = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Translating"):
        batch = all_texts[i:i + batch_size]

        # For multiline texts, translate line by line for better quality
        batch_results = []
        for text in batch:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) <= 1:
                result = translate_batch([text], tokenizer, model)
                batch_results.append(result[0] if result else text)
            else:
                translated_lines = translate_batch(lines, tokenizer, model)
                batch_results.append("\n".join(translated_lines))

        translated_texts.extend(batch_results)

    # Apply translations back to the data structure
    for (path, key, _original), translation in zip(texts_info, translated_texts):
        set_nested(data, path, key, translation)

    # Also store original Chinese in a separate field for reference
    for (path, key, original), translation in zip(texts_info, translated_texts):
        obj = data
        for p in path:
            obj = obj[p]
        obj[f"{key}_zh"] = original

    # Save translated metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("Translated metadata saved to: %s", output_path)
    log.info("Total texts translated: %d", len(translated_texts))


def main():
    parser = argparse.ArgumentParser(description="Translate LOKI metadata Chinese -> English")
    parser.add_argument("--input", required=True, help="Input metadata JSON")
    parser.add_argument("--output", required=True, help="Output translated JSON")
    parser.add_argument("--batch-size", type=int, default=32, help="Translation batch size")
    args = parser.parse_args()

    translate_metadata(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
