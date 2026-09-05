#!/usr/bin/env python3
"""Inject A4 model predictions into Label Studio as editable suggestions.

Reads a ``predictions.jsonl`` produced by ``evaluation/evaluate.py`` (boxes are
inline ``[y1,x1,y2,x2]`` tokens in the ``generated`` field), converts them to
Label Studio result objects, and writes them into the ``prediction`` table of
the live SQLite DB under a chosen ``model_version`` (default ``a4_gold``).

Annotations are NEVER touched. By default, only tasks *without* a human
annotation are updated (so reviewed gold stays clean). Enable
"Settings -> Machine Learning -> Show predictions to annotators" and select the
``a4_gold`` model version so the boxes are copied into an editable annotation on
open. Submit = human-reviewed.

Stop Label Studio before running (avoids DB locks).

Usage::

    python scripts/inject_model_predictions.py \\
        --predictions outputs/ablation/g4_gold/evaluation/pool/predictions.jsonl \\
        --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
        --project-title DD-VQA-Loc \\
        --model-version a4_gold
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_label_studio_ddvqa import (  # noqa: E402
    boxes_to_prediction_results,
    image_size,
    parse_boxes_from_answer,
    strip_bbox_tokens,
)

DEFAULT_SIZE = (384, 384)


def resolve_project_id(cur, title: Optional[str], pid: Optional[int]) -> int:
    if pid is not None:
        return pid
    rows = cur.execute("SELECT id, title FROM project").fetchall()
    if title:
        for r in rows:
            if r["title"] == title:
                return int(r["id"])
        raise SystemExit(f"Project {title!r} not found. Have: {[r['title'] for r in rows]}")
    if len(rows) == 1:
        return int(rows[0]["id"])
    raise SystemExit(f"Pass --project-title/--project-id. Projects: {[r['title'] for r in rows]}")


def load_predictions(path: str) -> dict[str, dict]:
    """image basename -> prediction record (last wins)."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = os.path.basename(r.get("image", ""))
        if img:
            out[img] = r
    return out


def build_result(rec: dict, width: int, height: int) -> tuple[list[dict], int]:
    generated = rec.get("generated") or rec.get("prediction") or ""
    clean = strip_bbox_tokens(generated).strip()
    label = str(rec.get("pred_label", "")).lower()
    verdict = "Real" if label == "real" else "Fake"
    boxes = parse_boxes_from_answer(generated)
    result = [
        {"id": "answer0", "from_name": "answer", "to_name": "image",
         "type": "textarea", "value": {"text": [clean]}},
        {"id": "verdict0", "from_name": "verdict", "to_name": "image",
         "type": "choices", "value": {"choices": [verdict]}},
    ]
    result.extend(boxes_to_prediction_results(boxes, width, height))
    return result, len(boxes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="predictions.jsonl from evaluate.py")
    p.add_argument("--sqlite", default=str(Path.home() / ".local/share/label-studio/label_studio.sqlite3"))
    p.add_argument("--project-title", default="DD-VQA-Loc")
    p.add_argument("--project-id", type=int, default=None)
    p.add_argument("--model-version", default="a4_gold")
    p.add_argument("--image-root", default="label_studio/data/frames",
                   help="Frames dir to read image size (fallback 384x384)")
    p.add_argument("--only-unannotated", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip tasks that already have a human annotation (default: on)")
    p.add_argument("--dry-run", action="store_true", help="Report only, no DB writes")
    args = p.parse_args()

    preds = load_predictions(args.predictions)
    print(f"Loaded {len(preds)} prediction records from {args.predictions}")

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    project_id = resolve_project_id(cur, args.project_title, args.project_id)

    annotated = {
        r["task_id"]
        for r in cur.execute(
            "SELECT DISTINCT task_id FROM task_completion "
            "WHERE project_id = ? AND (was_cancelled = 0 OR was_cancelled IS NULL)",
            (project_id,),
        ).fetchall()
    }

    image_root = Path(args.image_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

    matched = updated = skipped_ann = skipped_nomatch = 0
    size_cache: dict[str, tuple[int, int]] = {}
    rows_to_write: list[tuple] = []

    for r in cur.execute("SELECT id, data FROM task WHERE project_id = ?", (project_id,)).fetchall():
        data = json.loads(r["data"])
        img = os.path.basename(data.get("image_name") or data.get("image") or "")
        rec = preds.get(img)
        if rec is None:
            continue
        matched += 1
        if args.only_unannotated and r["id"] in annotated:
            skipped_ann += 1
            continue
        if img not in size_cache:
            fp = image_root / img
            try:
                size_cache[img] = image_size(fp) if fp.is_file() else DEFAULT_SIZE
            except Exception:
                size_cache[img] = DEFAULT_SIZE
        w, h = size_cache[img]
        result, n_boxes = build_result(rec, w, h)
        rows_to_write.append((
            json.dumps(result, ensure_ascii=False),
            1.0 if n_boxes else 0.5,
            args.model_version, now, now, r["id"], project_id, 0.0,
        ))
        updated += 1

    skipped_nomatch = len(preds) - matched
    print(f"project={project_id}  matched={matched}  to_update={updated}  "
          f"skipped_annotated={skipped_ann}  preds_without_task={skipped_nomatch}")

    if args.dry_run:
        print("(dry-run) nenhuma escrita. Rode sem --dry-run para aplicar.")
        conn.close()
        return

    task_ids = [row[5] for row in rows_to_write]
    if task_ids:
        qmarks = ",".join("?" * len(task_ids))
        cur.execute(
            f"DELETE FROM prediction WHERE project_id = ? AND model_version = ? "
            f"AND task_id IN ({qmarks})",
            (project_id, args.model_version, *task_ids),
        )
    cur.executemany(
        "INSERT INTO prediction "
        "(result, score, model_version, created_at, updated_at, task_id, project_id, mislabeling) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows_to_write,
    )
    conn.commit()
    conn.close()
    print(f"\nInseridas {len(rows_to_write)} predições (model_version={args.model_version}).")
    print("No Label Studio: Settings -> Machine Learning -> Show predictions to "
          f"annotators, e selecione a versão '{args.model_version}'.")


if __name__ == "__main__":
    main()
