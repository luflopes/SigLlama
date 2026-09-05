#!/usr/bin/env python3
"""Build the A4 gold dataset from the finished Label Studio annotations.

Reads the reviewed annotations (directly from the live Label Studio SQLite DB,
or from a UI export JSON) and writes training JSONL split by ``split``:

    <out-dir>/train_loc_gold.jsonl
    <out-dir>/val_loc_gold.jsonl
    <out-dir>/test_loc_gold.jsonl

It also writes the *pool* of not-yet-gold images (``plan_bucket == none``) as
``<out-dir>/pool.jsonl`` so the trained A4 can pre-annotate them for human
review (see ``inject_model_predictions.py``).

Only tasks with a real (non-cancelled) human annotation are exported. Boxes are
converted to the training convention ``[y1,x1,y2,x2]`` in ``[0,1000]`` and
injected inline in the answer text (reusing ``export_label_studio_ddvqa.py``).

Usage (from the machine that runs Label Studio)::

    # From the live DB (no manual export needed)
    python scripts/prepare_gold_dataset.py \\
        --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
        --project-title DD-VQA-Loc \\
        --out-dir data/ddvqa_gold

    # From a Label Studio UI export instead
    python scripts/prepare_gold_dataset.py \\
        --ls-export label_studio/export.json \\
        --out-dir data/ddvqa_gold

Then copy ``data/ddvqa_gold/*_loc_gold.jsonl`` to the training machine
(e.g. ``/datasets/deepfake/ddvqa_prepared/gold/``) referenced by
``configs/ablation/g4_gold.yaml``.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_label_studio_ddvqa import task_to_row  # noqa: E402

GOLD_BUCKETS = {"train_seed", "val_gold", "test_gold"}


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


def tasks_from_sqlite(sqlite_path: str, project_id: Optional[int], title: Optional[str]):
    """Yield LS-export-shaped task dicts (data + annotations) from the DB."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    pid = resolve_project_id(cur, title, project_id)

    comps: dict[int, list[dict]] = defaultdict(list)
    for r in cur.execute(
        "SELECT task_id, result, was_cancelled, updated_at, created_at "
        "FROM task_completion WHERE project_id = ?",
        (pid,),
    ).fetchall():
        comps[r["task_id"]].append({
            "result": json.loads(r["result"]) if r["result"] else [],
            "was_cancelled": bool(r["was_cancelled"]),
            "updated_at": r["updated_at"],
            "created_at": r["created_at"],
        })

    tasks = []
    for r in cur.execute("SELECT id, data FROM task WHERE project_id = ?", (pid,)).fetchall():
        tasks.append({
            "id": r["id"],
            "data": json.loads(r["data"]),
            "annotations": comps.get(r["id"], []),
        })
    conn.close()
    return tasks


def load_ls_export(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def has_human_annotation(task: dict) -> bool:
    return any(
        (not a.get("was_cancelled")) and a.get("result")
        for a in (task.get("annotations") or [])
    )


def pool_row(data: dict) -> dict:
    """A {split}.jsonl-style row for inference (no boxes needed)."""
    img = data.get("image_name") or data.get("image") or ""
    if img.startswith("/data/local-files/"):
        img = img.split("/")[-1]
    return {
        "image": os.path.basename(img),
        "question": data.get("question", ""),
        "answer": data.get("answer_seed", "") or "",
        "method": data.get("method", ""),
        "is_real": bool(data.get("is_real", False)),
        "split": data.get("split", ""),
        "sample_id": data.get("sample_id", ""),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sqlite", help="Live Label Studio SQLite DB")
    src.add_argument("--ls-export", help="Label Studio UI export (JSON or JSONL)")
    p.add_argument("--project-title", default="DD-VQA-Loc")
    p.add_argument("--project-id", type=int, default=None)
    p.add_argument("--out-dir", default=str(repo / "data" / "ddvqa_gold"))
    p.add_argument("--suffix", default="_loc_gold", help="Output split file suffix")
    p.add_argument(
        "--require-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only export tasks flagged in the sampling plan (plan_bucket in "
             "train_seed/val_gold/test_gold). Off = export every annotated task.",
    )
    p.add_argument("--no-pool", action="store_true", help="Do not write pool.jsonl")
    args = p.parse_args()

    if args.sqlite:
        tasks = tasks_from_sqlite(args.sqlite, args.project_id, args.project_title)
    else:
        tasks = load_ls_export(args.ls_export)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict]] = defaultdict(list)
    pool: list[dict] = []
    cover = defaultdict(Counter)   # split -> method -> count (gold)
    skipped_no_ann = skipped_not_plan = 0

    for task in tasks:
        data = task.get("data") or {}
        bucket = data.get("plan_bucket", "none")
        annotated = has_human_annotation(task)

        # Pool = not part of the gold plan (candidates for A4 pre-annotation).
        if bucket == "none" and not annotated:
            pool.append(pool_row(data))

        if not annotated:
            skipped_no_ann += 1
            continue
        if args.require_plan and bucket not in GOLD_BUCKETS:
            skipped_not_plan += 1
            continue

        row = task_to_row(task, allow_predictions=False)
        if row is None:
            skipped_no_ann += 1
            continue
        split = row.get("split") or "unknown"
        row["plan_bucket"] = bucket
        by_split[split].append(row)
        cover[split][row.get("method", "?")] += 1

    # ---- Write gold split files ----
    print("Gold dataset por split/método:")
    total = 0
    for split in ("train", "val", "test", "unknown"):
        rows = by_split.get(split)
        if not rows:
            continue
        path = out_dir / f"{split}{args.suffix}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_box = sum(1 for r in rows if r.get("boxes"))
        print(f"  [{split}] {len(rows):4d} rows  ({n_box} com caixa)  -> {path}")
        for m, c in sorted(cover[split].items()):
            print(f"       {m:16s} {c}")
        total += len(rows)

    print(f"\nTOTAL gold: {total}  (sem anotação: {skipped_no_ann}, "
          f"fora do plano: {skipped_not_plan})")

    # ---- Write pool for pre-annotation ----
    if not args.no_pool:
        pool_path = out_dir / "pool.jsonl"
        with open(pool_path, "w", encoding="utf-8") as f:
            for r in pool:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pc = Counter(r["split"] for r in pool)
        print(f"\nPool p/ pré-anotação (não-gold): {len(pool)}  {dict(pc)}")
        print(f"  -> {pool_path}")

    print("\nPróximo passo: copie os *_loc_gold.jsonl para a máquina de treino")
    print("(ex.: /datasets/deepfake/ddvqa_prepared/gold/) e rode scripts/train_a4_gold.sh")


if __name__ == "__main__":
    main()
