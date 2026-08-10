#!/usr/bin/env python3
"""Export Label Studio DD-VQA annotations back to training JSONL.

Reads a Label Studio JSON export (or JSONL of tasks) and writes rows with:
  - ``answer``: reference text with ``[y1,x1,y2,x2]`` tokens injected after
    the linked span (or appended if the span is missing);
  - ``answer_original``: editable answer without bbox tokens;
  - ``grounded_regions`` / ``boxes``: structured localization metadata;
  - original ``question``, ``image``, ``method``, ``is_real``, ``split``.

Coordinate convention matches the training pipeline:
``[y1, x1, y2, x2]`` integers in ``[0, 1000]``.

Usage
-----
::

    python scripts/export_label_studio_ddvqa.py \\
        --ls-export label_studio/export.json \\
        --output /datasets/deepfake/ddvqa_prepared/train_loc_reviewed.jsonl \\
        --split train

    # Or export all splits into one file / split files:
    python scripts/export_label_studio_ddvqa.py \\
        --ls-export label_studio/export.json \\
        --output-dir /datasets/deepfake/ddvqa_prepared/reviewed
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("export_ls_ddvqa")

BBOX_SCALE = 1000
_BBOX_RE = re.compile(r"\s*\[\d+,\d+,\d+,\d+\]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Label Studio DD-VQA localization to JSONL.",
    )
    p.add_argument(
        "--ls-export",
        required=True,
        help="Label Studio export JSON (list of tasks) or JSONL",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Single output JSONL (all splits mixed, or filtered by --split)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Write {split}_loc.jsonl files into this directory",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Keep only this split when writing --output",
    )
    p.add_argument(
        "--prefer-annotations",
        action="store_true",
        default=True,
        help="Prefer completed annotations over predictions (default)",
    )
    p.add_argument(
        "--allow-predictions",
        action="store_true",
        help="Fall back to predictions when no annotation exists",
    )
    return p.parse_args()


def load_export(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list of tasks")
        return data
    # JSONL
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ls_box_to_yxyx(value: dict) -> list[int]:
    """Convert Label Studio percent box to ``[y1,x1,y2,x2]`` in 0–1000."""
    x = float(value["x"])
    y = float(value["y"])
    w = float(value["width"])
    h = float(value["height"])
    x1 = int(round(x / 100.0 * BBOX_SCALE))
    y1 = int(round(y / 100.0 * BBOX_SCALE))
    x2 = int(round((x + w) / 100.0 * BBOX_SCALE))
    y2 = int(round((y + h) / 100.0 * BBOX_SCALE))
    x1 = max(0, min(BBOX_SCALE, x1))
    y1 = max(0, min(BBOX_SCALE, y1))
    x2 = max(0, min(BBOX_SCALE, x2))
    y2 = max(0, min(BBOX_SCALE, y2))
    return [y1, x1, y2, x2]


def pick_result_list(task: dict, allow_predictions: bool) -> Optional[list[dict]]:
    anns = task.get("annotations") or []
    # Prefer the latest non-cancelled annotation.
    usable = [
        a for a in anns
        if not a.get("was_cancelled") and a.get("result")
    ]
    if usable:
        usable.sort(key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        return usable[-1]["result"]

    if allow_predictions:
        preds = task.get("predictions") or []
        if preds:
            return preds[0].get("result") or []
    return None


def extract_fields(result: list[dict]) -> tuple[str, str, list[dict]]:
    """Return (answer, verdict, boxes) from a LS result list."""
    answer = ""
    verdict = ""
    # region_id -> partial box dict
    boxes_by_id: dict[str, dict[str, Any]] = {}

    for item in result:
        from_name = item.get("from_name")
        rtype = item.get("type")
        value = item.get("value") or {}
        rid = item.get("id") or ""

        if from_name == "answer" and rtype == "textarea":
            texts = value.get("text") or []
            if texts:
                answer = texts[0] if isinstance(texts[0], str) else str(texts[0])
        elif from_name == "verdict" and rtype == "choices":
            choices = value.get("choices") or []
            if choices:
                verdict = choices[0]
        elif from_name == "regions" and rtype == "rectanglelabels":
            labels = value.get("rectanglelabels") or []
            region = labels[0] if labels else "custom"
            boxes_by_id.setdefault(rid, {})
            boxes_by_id[rid].update({
                "id": rid,
                "region": region,
                "bbox": ls_box_to_yxyx(value),
                "ls": {
                    "x": value.get("x"),
                    "y": value.get("y"),
                    "width": value.get("width"),
                    "height": value.get("height"),
                },
            })
        elif from_name == "span" and rtype == "textarea":
            texts = value.get("text") or []
            span = texts[0] if texts else ""
            boxes_by_id.setdefault(rid, {})
            boxes_by_id[rid]["span"] = span

    boxes = [b for b in boxes_by_id.values() if "bbox" in b]
    return answer, verdict, boxes


def inject_boxes(answer: str, boxes: list[dict]) -> tuple[str, list[str]]:
    """Inject ``[y1,x1,y2,x2]`` after linked spans; append leftovers."""
    # Start from a clean answer (no previous tokens).
    out = _BBOX_RE.sub("", answer or "")
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    grounded: list[str] = []
    # Inject from right to left to preserve offsets when span is found.
    injections: list[tuple[int, str, str]] = []  # (pos, token, region)

    for box in boxes:
        region = box.get("region") or "custom"
        span = (box.get("span") or "").strip()
        y1, x1, y2, x2 = box["bbox"]
        token = f"[{y1},{x1},{y2},{x2}]"
        if span:
            # First case-insensitive occurrence of the span.
            m = re.search(re.escape(span), out, flags=re.IGNORECASE)
            if m:
                injections.append((m.end(), token, region))
                grounded.append(region)
                continue
        # Fallback: append a grounded clause at the end.
        pretty = region.replace("_", " ")
        clause = f" The {pretty} region {token}."
        out = out.rstrip() + clause
        grounded.append(region)

    injections.sort(key=lambda t: t[0], reverse=True)
    # Avoid double-injecting at the same offset for overlapping spans.
    used_offsets: set[int] = set()
    for pos, token, region in injections:
        if pos in used_offsets:
            # Append instead if collision.
            pretty = region.replace("_", " ")
            out = out.rstrip() + f" The {pretty} region {token}."
            continue
        used_offsets.add(pos)
        out = out[:pos] + f" {token}" + out[pos:]

    return out.strip(), grounded


def verdict_to_is_real(verdict: str, fallback: bool) -> bool:
    v = (verdict or "").strip().lower()
    if v == "real":
        return True
    if v == "fake":
        return False
    return fallback


def task_to_row(task: dict, allow_predictions: bool) -> Optional[dict]:
    data = task.get("data") or {}
    result = pick_result_list(task, allow_predictions=allow_predictions)
    if result is None:
        return None

    answer, verdict, boxes = extract_fields(result)
    if not answer:
        answer = data.get("answer_seed") or ""
    answer_original = _BBOX_RE.sub("", answer)
    answer_original = re.sub(r"[ \t]{2,}", " ", answer_original).strip()

    grounded_answer, grounded_regions = inject_boxes(answer_original, boxes)
    is_real = verdict_to_is_real(verdict, bool(data.get("is_real", False)))

    # Ensure verdict prefix consistency with training.
    prefix = "Real. " if is_real else "Fake. "
    body = re.sub(r"^\s*(Real|Fake)[\.,]?\s*", "", grounded_answer, flags=re.I)
    grounded_answer = prefix + body

    image = data.get("image_name") or data.get("image") or ""
    if image.startswith("/data/local-files/"):
        image = image.split("/")[-1]

    row = {
        "image": image,
        "question": data.get("question", ""),
        "answer": grounded_answer,
        "answer_original": answer_original,
        "is_real": is_real,
        "label": "real" if is_real else "fake",
        "method": data.get("method", ""),
        "split": data.get("split", ""),
        "sample_id": data.get("sample_id", ""),
        "grounded_regions": grounded_regions,
        "boxes": [
            {
                "region": b.get("region"),
                "span": b.get("span", ""),
                "bbox": b.get("bbox"),
            }
            for b in boxes
        ],
    }
    return row


def main() -> None:
    args = parse_args()
    if not args.output and not args.output_dir:
        raise SystemExit("Provide --output and/or --output-dir")

    tasks = load_export(Path(args.ls_export))
    logger.info("Loaded %d tasks from %s", len(tasks), args.ls_export)

    by_split: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for task in tasks:
        row = task_to_row(task, allow_predictions=args.allow_predictions)
        if row is None:
            skipped += 1
            continue
        split = row.get("split") or "unknown"
        if args.split and split != args.split:
            continue
        by_split[split].append(row)

    total = sum(len(v) for v in by_split.values())
    logger.info("Exportable rows=%d skipped=%d splits=%s", total, skipped, dict((k, len(v)) for k, v in by_split.items()))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for split in sorted(by_split):
                for row in by_split[split]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %s", out)

    if args.output_dir:
        odir = Path(args.output_dir)
        odir.mkdir(parents=True, exist_ok=True)
        for split, rows in by_split.items():
            path = odir / f"{split}_loc.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Wrote %d rows -> %s", len(rows), path)


if __name__ == "__main__":
    main()
