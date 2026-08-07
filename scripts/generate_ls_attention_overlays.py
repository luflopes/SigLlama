#!/usr/bin/env python3
"""Thin wrapper around ``extract_attention_maps.py`` for Label Studio.

Prefer calling extract_attention_maps.py directly on tarkin::

    python scripts/extract_attention_maps.py \\
        --checkpoint outputs/dino_lora_classifier/best.pt \\
        --images-dir /datasets/deepfake/ddvqa_prepared/frames \\
        --all-frames \\
        --overlay-dir /tmp/ddvqa_attention \\
        --batch-size 16
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "extract_attention_maps.py"
    # Default args if user runs this wrapper with no CLI flags.
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--checkpoint", str(repo / "outputs" / "dino_lora_classifier" / "best.pt"),
            "--images-dir", str(repo / "label_studio" / "data" / "frames"),
            "--all-frames",
            "--overlay-dir", str(repo / "label_studio" / "data" / "attention"),
            "--batch-size", "8",
            "--no-store-images",
        ])
    # Re-dispatch: extract_attention_maps owns the real CLI.
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    # If wrapper-style flags were used historically, map a couple of aliases.
    mapped: list[str] = []
    it = iter(cmd[2:])
    for arg in it:
        if arg == "--frames-dir":
            mapped.extend(["--images-dir", next(it)])
        elif arg == "--output-dir":
            mapped.extend(["--overlay-dir", next(it)])
        else:
            mapped.append(arg)
    # Ensure all-frames mode when using overlay defaults
    if "--all-frames" not in mapped and "--predictions" not in mapped:
        mapped.append("--all-frames")
    raise SystemExit(subprocess.call([cmd[0], cmd[1], *mapped]))


if __name__ == "__main__":
    main()
