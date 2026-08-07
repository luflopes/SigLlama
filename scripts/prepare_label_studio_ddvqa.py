#!/usr/bin/env python3
"""Build Label Studio tasks for DD-VQA localization review (local-first).

By default only the global question ``Does the image looks real/fake?`` is
kept (one task per unique image). Region-specific questions are left for a
later lexical-propagation pass after manual review.

Default local source (no server sync needed):
  - frames under ``label_studio/data/frames`` (or ``ddvqa/frames`` with --from-g4)
  - reference answers / loc JSONL
  - pre-boxes from grounded ``[y1,x1,y2,x2]`` tokens, landmarks as fallback

Usage
-----
::

    bash scripts/setup_label_studio_local.sh prepare
    # or:
    python scripts/prepare_label_studio_ddvqa.py --ddvqa-dir label_studio/data ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from create_loc_annotations import (  # noqa: E402
    detect_regions_in_text,
    landmarks_to_bbox_norm,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("prepare_ls_ddvqa")

_BBOX_RE = re.compile(r"\s*\[\d+,\d+,\d+,\d+\]")
_BBOX_TOKEN_RE = re.compile(r"\[(\d+),(\d+),(\d+),(\d+)\]")
_VERDICT_RE = re.compile(r"^\s*(Real|Fake)\b", re.IGNORECASE)

REGION_LABELS = {
    "eyes", "left_eye", "right_eye",
    "eyebrows", "left_eyebrow", "right_eyebrow",
    "nose", "mouth", "jawline",
    "cheeks", "left_cheek", "right_cheek",
    "forehead", "hair", "face", "skin", "custom",
}

BBOX_SCALE = 1000.0

# One annotation pass per image: only the global real/fake question.
# Region-specific questions (eyes/nose/...) are filled later by lexical match.
GLOBAL_QUESTION = "Does the image looks real/fake?"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    g4_eval = repo / "outputs" / "ablation" / "g4_lora_loc" / "evaluation"

    p = argparse.ArgumentParser(
        description="Prepare Label Studio tasks for DD-VQA localization.",
    )
    p.add_argument(
        "--from-g4",
        action="store_true",
        help=(
            "Local defaults: ddvqa/frames + g4 best_val/best_test "
            "predictions as reference answers. Landmarks optional."
        ),
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Alias for --from-g4 (kept for compatibility).",
    )
    p.add_argument(
        "--ddvqa-dir",
        default=None,
        help="Directory with train/val/test JSONL (server-style metadata).",
    )
    p.add_argument(
        "--ddvqa-jsonl",
        nargs="*",
        default=None,
        help="Explicit JSONL files. Format: [split:]path",
    )
    p.add_argument(
        "--predictions",
        nargs="*",
        default=None,
        help=(
            "Evaluation predictions JSONL with reference_answer "
            "(e.g. g4 best_val / best_test). Format: [split:]path"
        ),
    )
    p.add_argument(
        "--landmarks-jsonl",
        default=None,
        help="Optional landmarks JSONL (fallback when answer has no boxes).",
    )
    p.add_argument(
        "--image-root",
        default=None,
        help="Directory with frame images",
    )
    p.add_argument(
        "--document-root",
        default=None,
        help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
    )
    p.add_argument(
        "--output",
        default=str(repo / "label_studio" / "tasks_ddvqa_global.json"),
        help="Output Label Studio tasks JSON",
    )
    p.add_argument(
        "--image-url-prefix",
        default=None,
        help="Override image URL prefix for Label Studio",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits when using --ddvqa-dir",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--prefer-loc",
        action="store_true",
        help="Prefer {split}_loc.jsonl when using --ddvqa-dir",
    )
    p.add_argument(
        "--as-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Import pre-boxes as predictions (default: on). Tasks stay "
            "unannotated until you Submit in Label Stream — progress tracking "
            "works. Enable Settings→Machine Learning→Show predictions to "
            "annotators so boxes are copied into an editable annotation. "
            "Use --no-as-predictions only if you want prefilled annotations "
            "(marks every task as already done)."
        ),
    )
    p.add_argument(
        "--only-global-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"Keep only '{GLOBAL_QUESTION}' (one task per image). "
            "Region-specific questions are skipped for manual labeling "
            "(default: on). Use --no-only-global-question for all questions."
        ),
    )
    p.add_argument(
        "--question-equals",
        default=None,
        help="Keep only tasks whose question equals this string exactly.",
    )
    args = p.parse_args()

    if args.local:
        args.from_g4 = True

    if args.from_g4:
        args.image_root = args.image_root or str(repo / "ddvqa" / "frames")
        args.document_root = args.document_root or str(repo / "ddvqa")
        if not args.predictions and not args.ddvqa_dir and not args.ddvqa_jsonl:
            args.predictions = [
                f"val:{g4_eval / 'best_val' / 'predictions.jsonl'}",
                f"test:{g4_eval / 'best_test' / 'predictions.jsonl'}",
            ]
        # landmarks optional; only set default if file exists
        default_lm = repo / "label_studio" / "data" / "landmarks.jsonl"
        alt_lm = repo / "ddvqa" / "landmarks.jsonl"
        if args.landmarks_jsonl is None:
            if default_lm.is_file():
                args.landmarks_jsonl = str(default_lm)
            elif alt_lm.is_file():
                args.landmarks_jsonl = str(alt_lm)

    if not args.image_root:
        p.error("Provide --image-root, or pass --from-g4")

    if not (args.predictions or args.ddvqa_dir or args.ddvqa_jsonl):
        p.error("Provide --from-g4 / --predictions / --ddvqa-dir / --ddvqa-jsonl")

    if args.image_url_prefix is None:
        if args.document_root:
            try:
                rel = Path(args.image_root).resolve().relative_to(
                    Path(args.document_root).resolve()
                )
                rel_s = str(rel).replace("\\", "/")
                args.image_url_prefix = f"/data/local-files/?d={rel_s}"
            except ValueError:
                args.image_url_prefix = "/data/local-files/?d=frames"
                logger.warning(
                    "image-root not under document-root; using %s",
                    args.image_url_prefix,
                )
        else:
            args.image_url_prefix = "/data/local-files/?d=frames"

    return args


def strip_bbox_tokens(text: str) -> str:
    cleaned = _BBOX_RE.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def parse_spec(spec: str) -> tuple[str, Path]:
    if ":" in spec and not Path(spec).exists():
        split, path = spec.split(":", 1)
        return split, Path(path)
    path = Path(spec)
    stem = path.stem.replace("_loc", "").replace("predictions", "")
    # Infer split from parent dir name: best_val, best_test, ...
    parent = path.parent.name
    for cand in ("train", "val", "test"):
        if cand in parent or cand in stem:
            return cand, path
    return "unknown", path


def resolve_inputs(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    """Return list of (split, path, source_kind) where kind is 'pred'|'jsonl'."""
    items: list[tuple[str, Path, str]] = []

    if args.predictions:
        for spec in args.predictions:
            split, path = parse_spec(spec)
            items.append((split, path, "pred"))
        return items

    if args.ddvqa_jsonl:
        for spec in args.ddvqa_jsonl:
            split, path = parse_spec(spec)
            items.append((split, path, "jsonl"))
        return items

    root = Path(args.ddvqa_dir)
    for split in args.splits:
        loc = root / f"{split}_loc.jsonl"
        plain = root / f"{split}.jsonl"
        if args.prefer_loc and loc.is_file():
            items.append((split, loc, "jsonl"))
        elif plain.is_file():
            items.append((split, plain, "jsonl"))
        elif loc.is_file():
            items.append((split, loc, "jsonl"))
        else:
            logger.warning("Split missing, skipping: %s", split)
    if not items:
        raise SystemExit(f"No JSONL splits found under {root}")
    return items


def load_landmarks(path: Optional[str]) -> dict[str, list[list[float]]]:
    if not path:
        return {}
    db: dict[str, list[list[float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec["image"]
            db[key] = rec["landmarks"]
            db[os.path.basename(key)] = rec["landmarks"]
    return db


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def normalize_row(row: dict, source_kind: str) -> dict:
    """Unify prediction rows and DD-VQA JSONL rows."""
    if source_kind == "pred" or "reference_answer" in row:
        answer = row.get("reference_answer") or row.get("answer") or ""
        true_label = str(row.get("true_label", "")).lower()
        is_real = true_label == "real"
        if "is_real" in row:
            is_real = bool(row["is_real"])
        return {
            "image": row.get("image", ""),
            "question": row.get("question", "") or "",
            "answer": answer,
            "answer_original": strip_bbox_tokens(answer),
            "is_real": is_real,
            "method": row.get("method", "") or "",
        }

    answer = row.get("answer", "") or ""
    return {
        "image": row.get("image", ""),
        "question": row.get("question", "") or "",
        "answer": answer,
        "answer_original": row.get("answer_original") or strip_bbox_tokens(answer),
        "is_real": _row_is_real(row),
        "method": row.get("method", "") or "",
    }


def _row_is_real(row: dict) -> bool:
    if "is_real" in row:
        return bool(row["is_real"])
    label = str(row.get("label", "")).lower()
    if label in {"real", "0", "0.0"}:
        return True
    if label in {"fake", "1", "1.0"}:
        return False
    ans = row.get("answer_original") or row.get("answer") or ""
    m = _VERDICT_RE.match(ans)
    if m:
        return m.group(1).lower() == "real"
    return False


def yxyx_to_ls_percent(
    y1: int, x1: int, y2: int, x2: int,
) -> tuple[float, float, float, float]:
    """Convert [y1,x1,y2,x2] in 0–1000 → LS (x, y, w, h) percent."""
    x = (x1 / BBOX_SCALE) * 100.0
    y = (y1 / BBOX_SCALE) * 100.0
    w = ((x2 - x1) / BBOX_SCALE) * 100.0
    h = ((y2 - y1) / BBOX_SCALE) * 100.0
    return x, y, w, h


def parse_boxes_from_answer(answer: str) -> list[dict[str, Any]]:
    """Extract region boxes already grounded in the reference text.

    For each ``[y1,x1,y2,x2]`` token, take the nearest preceding lexical
    region mention as the label/span.
    """
    boxes: list[dict[str, Any]] = []
    seen: set[str] = set()
    region_matches = detect_regions_in_text(answer)

    for m in _BBOX_TOKEN_RE.finditer(answer):
        y1, x1, y2, x2 = (int(m.group(i)) for i in range(1, 5))
        # Nearest region mention ending at or before the bbox.
        best = None
        for region, start, end in region_matches:
            if end <= m.start() and (best is None or end > best[2]):
                best = (region, start, end)
        if best is None:
            region, start, end = "custom", m.start(), m.start()
            span = "custom"
        else:
            region, start, end = best
            span = answer[start:end]
        if region in seen:
            # Still keep extra boxes under custom to avoid dropping them.
            region_key = "custom"
            span = span or "custom"
        else:
            region_key = region if region in REGION_LABELS else "custom"
            if region_key != "custom":
                seen.add(region)

        x, y, w, h = yxyx_to_ls_percent(y1, x1, y2, x2)
        if w <= 0 or h <= 0:
            continue
        boxes.append({
            "region": region_key,
            "span": span,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        })
    return boxes


def boxes_from_landmarks(
    answer_clean: str,
    landmarks: list[list[float]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for region, start, end in detect_regions_in_text(answer_clean):
        if region in seen or region not in REGION_LABELS:
            continue
        norm = landmarks_to_bbox_norm(landmarks, region)
        if norm is None:
            continue
        x_min, y_min, x_max, y_max = norm
        x, y = x_min * 100.0, y_min * 100.0
        w, h = (x_max - x_min) * 100.0, (y_max - y_min) * 100.0
        if w <= 0 or h <= 0:
            continue
        seen.add(region)
        results.append({
            "region": region,
            "span": answer_clean[start:end],
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        })
    return results


def boxes_to_prediction_results(
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        rid = f"r{idx}_{box['region']}"
        common = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "id": rid,
            "to_name": "image",
        }
        results.append({
            **common,
            "from_name": "regions",
            "type": "rectanglelabels",
            "value": {
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "rotation": 0,
                "rectanglelabels": [box["region"]],
            },
        })
        results.append({
            **common,
            "from_name": "span",
            "type": "textarea",
            "value": {
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "rotation": 0,
                "text": [box["span"]],
            },
        })
    return results


def build_task(
    row: dict,
    split: str,
    sample_id: str,
    image_root: Path,
    image_url_prefix: str,
    landmarks_db: dict[str, list[list[float]]],
    as_predictions: bool = False,
) -> Optional[dict[str, Any]]:
    image_name = row.get("image", "")
    if not image_name:
        return None
    img_path = Path(image_name) if os.path.isabs(image_name) else image_root / image_name
    if not img_path.is_file():
        img_path = image_root / os.path.basename(image_name)
    if not img_path.is_file():
        logger.warning("Missing image: %s", image_name)
        return None

    width, height = image_size(img_path)
    raw_answer = row.get("answer", "") or ""
    answer_clean = row.get("answer_original") or strip_bbox_tokens(raw_answer)
    question = row.get("question", "") or ""
    is_real = bool(row.get("is_real", False))
    verdict = "Real" if is_real else "Fake"
    method = row.get("method", "") or ""
    basename = os.path.basename(image_name)

    # Prefer boxes already in the grounded reference; else landmarks.
    boxes = parse_boxes_from_answer(raw_answer)
    source = "reference_bboxes"
    if not boxes:
        landmarks = landmarks_db.get(image_name) or landmarks_db.get(basename)
        if landmarks is not None:
            boxes = boxes_from_landmarks(answer_clean, landmarks)
            source = "lexical_landmarks"
        else:
            source = "none"

    # Prefill as predictions by default so Data Manager progress stays
    # meaningful (0 annotations until human Submit). Predictions are
    # read-only until copied into an annotation (Label Stream does this
    # when "Show predictions to annotators" is enabled).
    prefill_results: list[dict[str, Any]] = [
        {
            "id": "answer0",
            "from_name": "answer",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [answer_clean]},
        },
        {
            "id": "verdict0",
            "from_name": "verdict",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [verdict]},
        },
    ]
    prefill_results.extend(boxes_to_prediction_results(boxes, width, height))
    n_boxes = len(boxes)

    url = f"{image_url_prefix.rstrip('/')}/{basename}"
    # Attention overlays: label_studio/data/attention/<basename>
    # DOCUMENT_ROOT must be label_studio/data (parent of frames/ and attention/).
    if "/frames" in image_url_prefix:
        attn_prefix = image_url_prefix.replace("/frames", "/attention").rstrip("/")
    else:
        attn_prefix = "/data/local-files/?d=attention"
    attn_url = f"{attn_prefix}/{basename}"
    task: dict[str, Any] = {
        "data": {
            "image": url,
            "image_attn": attn_url,
            "image_name": basename,
            "question": question,
            "method": method,
            "split": split,
            "sample_id": sample_id,
            "is_real": is_real,
            "answer_seed": answer_clean,
            "n_pre_boxes": n_boxes,
            "box_source": source,
            # Label Studio Text only supports a single $field per value=
            "meta_line": f"{split} · {method} · {sample_id} · {basename}",
        },
        "meta": {
            "split": split,
            "method": method,
            "sample_id": sample_id,
            "image": basename,
        },
    }
    if as_predictions:
        # Single model_version so Settings→Annotation can select one set
        # covering all tasks (Label Studio prelabel dropdown is one version).
        task["predictions"] = [
            {
                "model_version": "g4_ref",
                "score": 1.0 if n_boxes else 0.0,
                "result": prefill_results,
            }
        ]
    else:
        task["annotations"] = [
            {
                "ground_truth": False,
                "was_cancelled": False,
                "result": prefill_results,
            }
        ]
    return task


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    inputs = resolve_inputs(args)

    landmarks_db = load_landmarks(args.landmarks_jsonl)
    if args.landmarks_jsonl:
        logger.info("Landmarks loaded from %s (%d keys)", args.landmarks_jsonl, len(landmarks_db))
    else:
        logger.info("No landmarks file; will use boxes from reference_answer when present")

    logger.info("Image root: %s", image_root)
    logger.info("Document root: %s", args.document_root)
    logger.info("Image URL prefix: %s", args.image_url_prefix)

    question_filter: Optional[str] = args.question_equals
    if question_filter is None and args.only_global_question:
        question_filter = GLOBAL_QUESTION
    if question_filter:
        logger.info("Question filter (exact): %r", question_filter)

    tasks: list[dict[str, Any]] = []
    skipped = 0
    skipped_question = 0
    with_boxes = 0
    by_source: dict[str, int] = {}
    seen_images: set[str] = set()

    for split, path, kind in inputs:
        if not path.is_file():
            logger.error("Missing input: %s", path)
            continue
        logger.info("Reading %s (%s, kind=%s)", path, split, kind)
        with open(path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                if args.max_samples is not None and len(tasks) >= args.max_samples:
                    break
                raw = json.loads(line)
                row = normalize_row(raw, kind)
                q = (row.get("question") or "").strip()
                if question_filter is not None and q != question_filter:
                    skipped_question += 1
                    continue
                sample_id = f"{split}-{line_idx:06d}"
                task = build_task(
                    row=row,
                    split=split,
                    sample_id=sample_id,
                    image_root=image_root,
                    image_url_prefix=args.image_url_prefix,
                    landmarks_db=landmarks_db,
                    as_predictions=args.as_predictions,
                )
                if task is None:
                    skipped += 1
                    continue
                img_name = task["data"]["image_name"]
                if img_name in seen_images:
                    # Same frame can appear in multiple metadata rows; keep one.
                    skipped_question += 1
                    continue
                seen_images.add(img_name)
                if task["data"]["n_pre_boxes"] > 0:
                    with_boxes += 1
                src = task["data"]["box_source"]
                by_source[src] = by_source.get(src, 0) + 1
                tasks.append(task)
        if args.max_samples is not None and len(tasks) >= args.max_samples:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote %d tasks -> %s (unique_images=%d, with_pre_boxes=%d, "
        "skipped_missing=%d, skipped_question_or_dup=%d, by_source=%s)",
        len(tasks), out, len(seen_images), with_boxes, skipped,
        skipped_question, by_source,
    )
    if args.from_g4:
        logger.info(
            "Note: g4 predictions cover val+test only (~797 unique images). "
            "Train references are not in the ablation outputs."
        )


if __name__ == "__main__":
    main()
