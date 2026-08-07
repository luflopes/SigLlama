#!/usr/bin/env python3
"""Merge a Label Studio export into a reimport JSON with progress preserved.

Human-reviewed tasks (lead_time / updated_at) keep ``annotations``.
Everything else becomes ``predictions`` so Data Manager progress works again.

Usage::

    python scripts/merge_ls_export_progress.py \\
        --export label_studio/export_parcial.json \\
        --base label_studio/tasks_ddvqa_global.json \\
        --output label_studio/tasks_ddvqa_reimport.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_reviewed(ann: dict[str, Any], min_seconds: float = 3.0) -> bool:
    lt = float(ann.get("lead_time") or 0.0)
    c, u = parse_ts(ann.get("created_at")), parse_ts(ann.get("updated_at"))
    delta = abs((u - c).total_seconds()) if c and u else 0.0
    return lt >= min_seconds or delta >= min_seconds


def clean_result(result: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in result or []:
        item: dict[str, Any] = {
            "id": r.get("id"),
            "from_name": r["from_name"],
            "to_name": r["to_name"],
            "type": r["type"],
            "value": r["value"],
        }
        for k in ("original_width", "original_height", "image_rotation", "parentID"):
            if k in r:
                item[k] = r[k]
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, help="Label Studio JSON export")
    p.add_argument(
        "--base",
        default=None,
        help="Optional tasks JSON with predictions fallback (e.g. tasks_ddvqa_global.json)",
    )
    p.add_argument("--output", required=True, help="Output reimport JSON")
    p.add_argument("--min-seconds", type=float, default=3.0)
    args = p.parse_args()

    exp = json.loads(Path(args.export).read_text(encoding="utf-8"))
    base_by_img: dict[str, dict] = {}
    if args.base:
        base = json.loads(Path(args.base).read_text(encoding="utf-8"))
        base_by_img = {t["data"]["image_name"]: t for t in base}

    reviewed: list[str] = []
    tasks_out: list[dict[str, Any]] = []
    for t in exp:
        data = dict(t.get("data") or {})
        img = data.get("image_name") or ""
        if "meta_line" not in data:
            data["meta_line"] = (
                f"{data.get('split', '')} · {data.get('method', '')} · "
                f"{data.get('sample_id', '')} · {data.get('image_name', '')}"
            )
        if img and "image_attn" not in data:
            data["image_attn"] = f"/data/local-files/?d=attention/{img}"
        anns = t.get("annotations") or []
        base_t = base_by_img.get(img)
        pred_fallback = None
        if base_t and base_t.get("predictions"):
            pred_fallback = base_t["predictions"][0]

        task: dict[str, Any] = {
            "data": data,
            "meta": t.get("meta") or (base_t or {}).get("meta") or {},
        }

        if anns and is_reviewed(anns[0], args.min_seconds):
            reviewed.append(img)
            task["annotations"] = [{
                "ground_truth": False,
                "was_cancelled": False,
                "result": clean_result(anns[0].get("result")),
            }]
            if pred_fallback:
                task["predictions"] = [{
                    "model_version": "g4_ref",
                    "score": pred_fallback.get("score", 0.0),
                    "result": clean_result(pred_fallback.get("result")),
                }]
        else:
            if pred_fallback:
                task["predictions"] = [{
                    "model_version": "g4_ref",
                    "score": pred_fallback.get("score", 0.0),
                    "result": clean_result(pred_fallback.get("result")),
                }]
            elif anns:
                task["predictions"] = [{
                    "model_version": "g4_ref",
                    "score": 1.0 if data.get("n_pre_boxes") else 0.0,
                    "result": clean_result(anns[0].get("result")),
                }]
            else:
                task["predictions"] = []
        tasks_out.append(task)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks_out)} tasks -> {out}")
    print(f"Preserved as annotations (reviewed): {len(reviewed)}")
    for img in reviewed:
        print(f"  - {img}")


if __name__ == "__main__":
    main()
