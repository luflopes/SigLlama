from __future__ import annotations

import json
import os
from typing import IO


class NDJSONWriter:
    """Append-mode NDJSON writer with context-manager support and optional
    resume capability (skips already-processed IDs)."""

    def __init__(self, output_path: str, flush: bool = True):
        self._path = output_path
        self._flush = flush
        self._f: IO[str] | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "NDJSONWriter":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._f = open(self._path, "a", encoding="utf-8")

    def write(self, entry: dict) -> None:
        if self._f is None:
            raise RuntimeError("Writer not opened. Use open() or context manager.")
        line = json.dumps(entry, ensure_ascii=False)
        self._f.write(line + "\n")
        if self._flush:
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_processed_ids(path: str, id_key: str = "image_id") -> set[str]:
        """Read an existing NDJSON file and return a set of already-processed
        IDs so the extraction pipeline can skip them."""
        ids: set[str] = set()
        if not os.path.isfile(path):
            return ids
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ids.add(str(record[id_key]))
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids
