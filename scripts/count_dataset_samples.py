#!/usr/bin/env python3
"""Count samples per dataset, split, and category for the experiments.

Inspects the JSONL metadata files used in training/evaluation and reports
counts broken down by method (deepfake category), real/fake label, and split.

Usage::

    python scripts/count_dataset_samples.py

Adjust the DATASET_PATHS dict below if your paths differ.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATASET_PATHS = {
    "DD-VQA (train)": "/datasets/deepfake/ddvqa_prepared/train.jsonl",
    "DD-VQA (val)": "/datasets/deepfake/ddvqa_prepared/val.jsonl",
    "DD-VQA (test)": "/datasets/deepfake/ddvqa_prepared/test.jsonl",
    "DD-VQA loc (train)": "/datasets/deepfake/ddvqa_prepared/train_loc.jsonl",
    "DD-VQA loc (val)": "/datasets/deepfake/ddvqa_prepared/val_loc.jsonl",
    "DD-VQA loc (test)": "/datasets/deepfake/ddvqa_prepared/test_loc.jsonl",
    "FF++ cls (train)": "/datasets/deepfake/ff_classification/train.jsonl",
    "FF++ cls (val)": "/datasets/deepfake/ff_classification/val.jsonl",
    "FF++ cls (test)": "/datasets/deepfake/ff_classification/test.jsonl",
    "Celeb-DF-v2 (test)": "/datasets/deepfake/celebdf_prepared/test.jsonl",
    "WildDeepfake (test_sampled)": "/datasets/deepfake/wilddeepfake_prepared/test_sampled.jsonl",
    "WildDeepfake (test_full)": "/datasets/deepfake/wilddeepfake_prepared/test.jsonl",
}


def count_jsonl(path: str) -> dict:
    """Count samples in a JSONL file by method and label."""
    if not os.path.isfile(path):
        return None

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    by_method = Counter()
    by_label = Counter()
    by_method_label = Counter()

    for r in rows:
        method = r.get("method", "unknown")
        is_real = r.get("is_real", None)
        if is_real is None:
            label_val = r.get("label", None)
            if label_val is not None:
                is_real = (label_val == 0)
            else:
                answer = r.get("answer", "")
                is_real = "real" in answer.lower()

        label = "real" if is_real else "fake"
        by_method[method] += 1
        by_label[label] += 1
        by_method_label[(method, label)] += 1

    # Count unique videos
    video_ids = set(r.get("video_id", "") for r in rows if r.get("video_id"))

    return {
        "total_samples": total,
        "total_videos": len(video_ids),
        "by_label": dict(by_label),
        "by_method": dict(by_method.most_common()),
        "by_method_label": {
            f"{method} ({label})": count
            for (method, label), count in sorted(by_method_label.items())
        },
    }


def main():
    print("=" * 80)
    print("DATASET SAMPLE COUNTS")
    print("=" * 80)

    all_results = {}

    for name, path in DATASET_PATHS.items():
        result = count_jsonl(path)
        if result is None:
            print(f"\n--- {name} ---")
            print(f"  FILE NOT FOUND: {path}")
            continue

        all_results[name] = result
        print(f"\n--- {name} ---")
        print(f"  File: {path}")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Unique videos: {result['total_videos']}")
        print(f"  By label: real={result['by_label'].get('real', 0)}, fake={result['by_label'].get('fake', 0)}")
        print(f"  By method:")
        for method, count in result['by_method']:
            print(f"    {method:20s}: {count}")
        print(f"  By method+label:")
        for key, count in result['by_method_label'].items():
            print(f"    {key:35s}: {count}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Dataset':<35s} {'Total':>8} {'Real':>8} {'Fake':>8} {'Videos':>8}")
    print("-" * 75)
    for name, result in all_results.items():
        print(
            f"{name:<35s} "
            f"{result['total_samples']:>8} "
            f"{result['by_label'].get('real', 0):>8} "
            f"{result['by_label'].get('fake', 0):>8} "
            f"{result['total_videos']:>8}"
        )

    # Save as JSON
    output_path = "outputs/dataset_counts.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
