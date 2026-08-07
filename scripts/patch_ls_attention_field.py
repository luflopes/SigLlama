#!/usr/bin/env python3
"""Add ``image_attn`` to Label Studio tasks without touching annotations.

Updates:
  1) JSON task files (tasks_ddvqa_*.json)
  2) Optionally the local Label Studio SQLite DB (preserves all annotations)

Usage::

    # JSON only
    python scripts/patch_ls_attention_field.py \\
        --tasks-json label_studio/tasks_ddvqa_reimport.json

    # Live project DB (no Delete Tasks needed)
    python scripts/patch_ls_attention_field.py \\
        --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
        --project-title DD-VQA-Loc
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


def attn_url(image_name: str, prefix: str = "/data/local-files/?d=attention") -> str:
    return f"{prefix.rstrip('/')}/{image_name}"


def patch_data(data: dict[str, Any], prefix: str) -> bool:
    name = data.get("image_name") or Path(str(data.get("image", ""))).name
    if not name:
        return False
    url = attn_url(name, prefix)
    if data.get("image_attn") == url:
        return False
    data["image_attn"] = url
    return True


def patch_tasks_json(path: Path, prefix: str) -> int:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for t in tasks:
        data = t.setdefault("data", {})
        if patch_data(data, prefix):
            n += 1
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def patch_sqlite(
    db_path: Path,
    prefix: str,
    project_title: Optional[str] = None,
    project_id: Optional[int] = None,
) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if project_id is None:
        if project_title:
            row = cur.execute(
                "SELECT id, title FROM project WHERE title = ?",
                (project_title,),
            ).fetchone()
            if row is None:
                rows = cur.execute("SELECT id, title FROM project").fetchall()
                titles = ", ".join(f"{r['id']}:{r['title']}" for r in rows)
                raise SystemExit(
                    f"Project title {project_title!r} not found. Available: {titles}"
                )
            project_id = int(row["id"])
        else:
            rows = cur.execute("SELECT id, title FROM project").fetchall()
            if len(rows) != 1:
                titles = ", ".join(f"{r['id']}:{r['title']}" for r in rows)
                raise SystemExit(f"Pass --project-title or --project-id. Projects: {titles}")
            project_id = int(rows[0]["id"])

    rows = cur.execute(
        "SELECT id, data FROM task WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    updated = 0
    for row in rows:
        data = json.loads(row["data"])
        if patch_data(data, prefix):
            cur.execute(
                "UPDATE task SET data = ? WHERE id = ?",
                (json.dumps(data, ensure_ascii=False), row["id"]),
            )
            updated += 1
    conn.commit()
    # Label Studio OSS uses task_completion for annotations.
    try:
        n_ann = cur.execute(
            "SELECT COUNT(*) FROM task_completion tc "
            "JOIN task t ON t.id = tc.task_id WHERE t.project_id = ?",
            (project_id,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        n_ann = cur.execute(
            "SELECT COUNT(*) FROM annotation a "
            "JOIN task t ON t.id = a.task_id WHERE t.project_id = ?",
            (project_id,),
        ).fetchone()[0]
    conn.close()
    return updated, int(n_ann)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks-json",
        nargs="*",
        default=None,
        help="Task JSON files to patch (default: global + reimport if present)",
    )
    p.add_argument(
        "--sqlite",
        default=None,
        help="Path to label_studio.sqlite3 to patch in place",
    )
    p.add_argument("--project-title", default="DD-VQA-Loc")
    p.add_argument("--project-id", type=int, default=None)
    p.add_argument(
        "--url-prefix",
        default="/data/local-files/?d=attention",
        help="URL prefix for attention overlays",
    )
    args = p.parse_args()

    json_paths = args.tasks_json
    if json_paths is None and args.sqlite is None:
        json_paths = []
        for name in ("tasks_ddvqa_reimport.json", "tasks_ddvqa_global.json"):
            path = repo / "label_studio" / name
            if path.is_file():
                json_paths.append(str(path))

    if json_paths:
        for jp in json_paths:
            path = Path(jp)
            n = patch_tasks_json(path, args.url_prefix)
            print(f"JSON {path}: patched {n} tasks")

    if args.sqlite:
        updated, n_ann = patch_sqlite(
            Path(args.sqlite).expanduser(),
            args.url_prefix,
            project_title=args.project_title,
            project_id=args.project_id,
        )
        print(
            f"SQLite {args.sqlite}: updated data on {updated} tasks; "
            f"annotations untouched (count probe={n_ann})"
        )


if __name__ == "__main__":
    main()
