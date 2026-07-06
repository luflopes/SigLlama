#!/usr/bin/env python3
"""Sample N frames per video from WildDeepfake test.jsonl.

Reads the full test.jsonl and produces test_sampled.jsonl with at most
--frames-per-video frames uniformly sampled from each video_id group.

Usage::

    python scripts/sample_wilddeepfake.py \
        --input /datasets/deepfake/wilddeepfake_prepared/test.jsonl \
        --output /datasets/deepfake/wilddeepfake_prepared/test_sampled.jsonl \
        --frames-per-video 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sample N frames per video from WildDeepfake")
    parser.add_argument("--input", required=True, help="Input test.jsonl path")
    parser.add_argument("--output", required=True, help="Output sampled jsonl path")
    parser.add_argument("--frames-per-video", type=int, default=32,
                        help="Max frames to keep per video_id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    log.info("Loading %s", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    log.info("Total samples: %d", len(samples))

    # Group by video_id
    by_video = defaultdict(list)
    for s in samples:
        by_video[s["video_id"]].append(s)

    log.info("Unique videos: %d", len(by_video))

    # Sample uniformly from each video
    sampled = []
    for vid_id, frames in sorted(by_video.items()):
        if len(frames) <= args.frames_per_video:
            sampled.extend(frames)
        else:
            indices = np.linspace(0, len(frames) - 1, args.frames_per_video, dtype=int)
            sampled.extend([frames[i] for i in indices])

    log.info("Sampled: %d (from %d videos, max %d/video)",
             len(sampled), len(by_video), args.frames_per_video)

    n_real = sum(1 for s in sampled if s["is_real"])
    n_fake = len(sampled) - n_real
    log.info("Distribution: real=%d, fake=%d", n_real, n_fake)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in sampled:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    log.info("Saved to: %s", args.output)


if __name__ == "__main__":
    main()
