#!/usr/bin/env python3
"""Build a stratified gold sampling plan for DD-VQA localization.

Reads the live Label Studio project (SQLite), keeps every image already
annotated, and fills per-method quotas for three buckets:

    test_gold  : fully manual (no model assist)   -> evaluation
    val_gold   : fully manual (no model assist)   -> model selection
    train_seed : manual seed (already-labeled count here) -> training

It never touches annotations. With ``--apply`` it writes plan flags into
``task.data`` (``in_plan``, ``plan_bucket``, ``plan_status``) so you can
filter the Data Manager by ``plan_bucket`` and label only the sample.

Usage::

    # Dry run (prints the plan, writes manifest files)
    python scripts/build_sampling_plan.py \\
        --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
        --project-title DD-VQA-Loc

    # Apply flags to the DB (stop Label Studio first)
    python scripts/build_sampling_plan.py \\
        --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
        --project-title DD-VQA-Loc --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

DEFAULT_QUOTAS = {"test": 30, "val": 30, "train": 50}
BUCKET = {"test": "test_gold", "val": "val_gold", "train": "train_seed"}


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


def load_tasks(cur, project_id: int) -> list[dict]:
    annotated_task_ids = {
        r["task_id"]
        for r in cur.execute(
            "SELECT DISTINCT task_id FROM task_completion "
            "WHERE project_id = ? AND (was_cancelled = 0 OR was_cancelled IS NULL)",
            (project_id,),
        ).fetchall()
    }
    tasks = []
    for r in cur.execute(
        "SELECT id, data FROM task WHERE project_id = ?", (project_id,)
    ).fetchall():
        d = json.loads(r["data"])
        tasks.append({
            "task_id": r["id"],
            "image_name": d.get("image_name") or "",
            "split": d.get("split") or "?",
            "method": d.get("method") or "?",
            "annotated": r["id"] in annotated_task_ids,
        })
    return tasks


def build_plan(tasks: list[dict], quotas: dict[str, int], seed: int):
    pool: dict[tuple, list] = defaultdict(list)
    for t in tasks:
        pool[(t["split"], t["method"])].append(t)

    rng = random.Random(seed)
    plan: dict[str, dict] = {}
    for (split, method), items in pool.items():
        quota = quotas.get(split, 0)
        if quota == 0:
            continue
        bucket = BUCKET[split]
        done = [x for x in items if x["annotated"]]
        free = [x for x in items if not x["annotated"]]
        free.sort(key=lambda x: x["image_name"])
        rng.shuffle(free)
        need = max(0, quota - len(done))
        chosen = done + free[:need]
        for x in chosen:
            plan[x["image_name"]] = {
                "task_id": x["task_id"],
                "bucket": bucket,
                "status": "done" if x["annotated"] else "todo",
                "split": split,
                "method": method,
            }
    return plan


def summarize(plan: dict[str, dict], quotas: dict[str, int]):
    agg: dict = defaultdict(lambda: defaultdict(lambda: {"done": 0, "todo": 0}))
    for info in plan.values():
        cell = agg[info["split"]][info["method"]]
        cell[info["status"]] += 1
    return agg


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqlite", default=str(Path.home() / ".local/share/label-studio/label_studio.sqlite3"))
    p.add_argument("--project-title", default="DD-VQA-Loc")
    p.add_argument("--project-id", type=int, default=None)
    p.add_argument("--test-per-method", type=int, default=DEFAULT_QUOTAS["test"])
    p.add_argument("--val-per-method", type=int, default=DEFAULT_QUOTAS["val"])
    p.add_argument("--train-per-method", type=int, default=DEFAULT_QUOTAS["train"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest-json", default=str(repo / "label_studio" / "sampling_plan.json"))
    p.add_argument("--manifest-csv", default=str(repo / "label_studio" / "sampling_plan.csv"))
    p.add_argument("--apply", action="store_true", help="Write plan flags into the DB task.data")
    args = p.parse_args()

    quotas = {
        "test": args.test_per_method,
        "val": args.val_per_method,
        "train": args.train_per_method,
    }

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    project_id = resolve_project_id(cur, args.project_title, args.project_id)
    tasks = load_tasks(cur, project_id)
    plan = build_plan(tasks, quotas, args.seed)
    agg = summarize(plan, quotas)

    # ---- Print summary ----
    methods = sorted({t["method"] for t in tasks})
    print(f"Project id={project_id}  tasks={len(tasks)}  in_plan={len(plan)}")
    print(f"Quotas per method: {quotas}\n")
    for split in ("train", "val", "test"):
        if split not in agg:
            continue
        print(f"[{split}]  bucket={BUCKET[split]}")
        tot_done = tot_todo = 0
        for m in methods:
            c = agg[split].get(m)
            if not c:
                continue
            tot_done += c["done"]
            tot_todo += c["todo"]
            print(f"   {m:16s} done={c['done']:3d}  todo={c['todo']:3d}  -> {c['done']+c['todo']:3d}")
        print(f"   {'TOTAL':16s} done={tot_done:3d}  todo={tot_todo:3d}  -> {tot_done+tot_todo:3d}\n")

    total_todo = sum(1 for i in plan.values() if i["status"] == "todo")
    total_done = sum(1 for i in plan.values() if i["status"] == "done")
    print(f"GOLD TOTAL: {len(plan)}  (done={total_done}, todo={total_todo})")
    print(f"Estimativa manual pleno (~50/dia): {total_todo/50:.1f} dias\n")

    # ---- Manifest files ----
    Path(args.manifest_json).write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(args.manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "split", "method", "bucket", "status", "task_id"])
        for img, info in sorted(plan.items(), key=lambda kv: (kv[1]["split"], kv[1]["method"], kv[0])):
            w.writerow([img, info["split"], info["method"], info["bucket"], info["status"], info["task_id"]])
    print(f"Manifest: {args.manifest_json}\n          {args.manifest_csv}")

    # ---- Apply flags to DB ----
    if args.apply:
        n = 0
        for r in cur.execute(
            "SELECT id, data FROM task WHERE project_id = ?", (project_id,)
        ).fetchall():
            d = json.loads(r["data"])
            img = d.get("image_name") or ""
            info = plan.get(img)
            new = {
                "in_plan": bool(info),
                "plan_bucket": info["bucket"] if info else "none",
                "plan_status": info["status"] if info else "none",
            }
            if all(d.get(k) == v for k, v in new.items()):
                continue
            d.update(new)
            cur.execute("UPDATE task SET data = ? WHERE id = ?", (json.dumps(d, ensure_ascii=False), r["id"]))
            n += 1
        conn.commit()
        print(f"\nApplied plan flags to {n} tasks (annotations untouched).")
        print("In Label Studio: Data Manager -> Filter -> plan_bucket = test_gold/val_gold/train_seed")
    else:
        print("\n(dry-run) Re-run with --apply to write plan_bucket into the DB.")

    conn.close()


if __name__ == "__main__":
    main()
