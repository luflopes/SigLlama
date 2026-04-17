"""Quick sanity check on a prepared DD-VQA JSONL split.

Reports:
  - total samples
  - real / fake counts (using the same logic as DDVQADataset)
  - per-method counts
  - first few sample rows

Usage::

    python scripts/inspect_ddvqa_split.py /datasets/deepfake/ddvqa_prepared/train.jsonl
    python scripts/inspect_ddvqa_split.py /datasets/deepfake/ddvqa_prepared/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def sample_is_real(row: dict) -> bool:
    if "is_real" in row:
        return bool(row["is_real"])
    return str(row.get("label", "")).lower() == "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL path(s) to inspect")
    ap.add_argument("--head", type=int, default=3, help="Print N first rows")
    args = ap.parse_args()

    for p in args.paths:
        path = Path(p)
        if not path.is_file():
            print(f"[MISS] {p}", file=sys.stderr)
            continue

        rows: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        n_real = sum(1 for r in rows if sample_is_real(r))
        n_fake = len(rows) - n_real
        by_method: Counter = Counter(r.get("method", "unknown") for r in rows)
        by_real_method: Counter = Counter(
            (sample_is_real(r), r.get("method", "unknown")) for r in rows
        )

        print(f"\n=== {p} ===")
        print(f"total : {len(rows)}")
        print(f"real  : {n_real}  ({100 * n_real / max(len(rows), 1):.1f}%)")
        print(f"fake  : {n_fake}  ({100 * n_fake / max(len(rows), 1):.1f}%)")
        print("by method:")
        for m, c in by_method.most_common():
            print(f"  {m:16s}: {c}")
        print("by (is_real, method):")
        for (is_real, m), c in sorted(by_real_method.items()):
            tag = "real" if is_real else "fake"
            print(f"  ({tag:4s}, {m:16s}): {c}")

        keys_seen: set = set()
        for r in rows[:20]:
            keys_seen.update(r.keys())
        print(f"schema keys observed in first 20 rows: {sorted(keys_seen)}")

        for row in rows[: args.head]:
            print("  row:", json.dumps(row, ensure_ascii=False)[:220])


if __name__ == "__main__":
    main()
