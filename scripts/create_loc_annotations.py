"""Enrich DD-VQA annotations with **content-aware** localization tokens.

Reads the original DD-VQA JSONL plus a landmarks JSONL (from
``scripts/extract_landmarks.py``) and produces an enriched version where
the answer text is *grounded* to facial regions actually mentioned in
the question/answer.

Strategy
--------
For each sample:
  1. Detect facial regions named in the question and the answer text
     (eyes, eyebrows, nose, mouth, jawline, cheeks, forehead, etc.) by
     keyword matching with synonyms and word boundaries.
  2. For every region detected in the answer, inject the corresponding
     ``[y1,x1,y2,x2]`` bounding box (integer coords in [0, 1000])
     immediately after the first mention. Subsequent mentions of the
     same region are left untouched (a single grounding per region per
     answer keeps the supervision signal focused).
  3. If the question asks specifically about a region that the answer
     does not explicitly name, prepend a grounded clause referencing
     that region so the model still learns to ground question-driven
     attention.

Differences from the previous template-based version
----------------------------------------------------
The previous implementation discarded the original answer and built a
canned explanation from a (method -> regions) lookup. That made the
model regurgitate templates instead of grounding free-form reasoning.
This version *preserves* the original answer style and only annotates
mentioned regions, which is closer to the TruthLens recipe.

Coordinate format
-----------------
``[y1,x1,y2,x2]`` integers in ``[0, 1000]``. The bounding box is computed
from the convex hull of the relevant landmark indices (clipped to the
image). Order ``y1, x1, y2, x2`` is consistent with PaliGemma's location
tokens and survives both PaliGemma2 and TinyLlama tokenisation as plain
ASCII without vocabulary extension.

Usage
-----
::

    python scripts/create_loc_annotations.py \\
        --ddvqa-jsonl /datasets/deepfake/ddvqa_prepared/train.jsonl \\
        --landmarks-jsonl /datasets/deepfake/ddvqa_prepared/landmarks.jsonl \\
        --output /datasets/deepfake/ddvqa_prepared/train_loc.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from typing import Optional

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── MediaPipe 478 landmark → semantic facial region mapping ──
# Indices follow the canonical 468-point face mesh (+10 iris). They are
# stable across MediaPipe versions and are documented at
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker
FACIAL_REGIONS: dict[str, list[int]] = {
    # ── Eyes ──
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246,
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        466, 388, 387, 386, 385, 384, 398,
    ],
    # ── Eyebrows ──
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    # ── Nose ──
    "nose": [
        1, 2, 3, 4, 5, 6, 19, 94, 195, 197,
        45, 275, 220, 440, 48, 64, 278, 294,
    ],
    # ── Mouth (inner + outer lip) ──
    "mouth": [
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 95,
        146, 178, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317,
        318, 324, 375, 402, 405,
    ],
    # ── Jawline / chin ──
    "jawline": [
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
        454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    ],
    # ── Cheeks ──
    "left_cheek": [116, 117, 118, 119, 120, 121, 128, 36, 205, 206, 207],
    "right_cheek": [345, 346, 347, 348, 349, 350, 357, 266, 425, 426, 427],
    # ── Forehead ──
    "forehead": [
        10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109,
        68, 104, 69, 108, 151,
    ],
}

# Composite regions: union of subregions, single bbox computed from all
# the indices below (so "eyes" with no left/right qualifier yields one
# bbox spanning both eyes).
COMPOSITE_REGIONS: dict[str, list[str]] = {
    "eyes": ["left_eye", "right_eye"],
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
    "cheeks": ["left_cheek", "right_cheek"],
}

# ── Keyword vocabulary ──
# Each tuple is (canonical_region_key, surface_pattern). Patterns are
# regex (case-insensitive) anchored with word boundaries so that "eye"
# does not match inside "eyebrow" or "eyebrows". Order matters: longer
# / more specific keywords are tried first to avoid "left eye" being
# preempted by "eye".
#
# Surface forms gathered from inspecting DD-VQA answers (e.g. ``eye
# brows`` with a space, ``eyebrow`` singular, ``lip`` vs ``lips``,
# ``chin`` ~= jawline). Add new entries here as new patterns are
# encountered in the corpus.
_KEYWORD_TABLE: list[tuple[str, str]] = [
    # ── Specific eye/eyebrow side first ──
    ("left_eyebrow", r"left\s+eyebrow"),
    ("right_eyebrow", r"right\s+eyebrow"),
    ("left_eye", r"left\s+eye(?!\w)"),
    ("right_eye", r"right\s+eye(?!\w)"),

    # ── Composite regions ──
    ("eyebrows", r"eye\s?brows?"),  # "eyebrows" or rare "eye brows"
    ("eyebrows", r"\bbrows?\b"),
    ("eyes", r"\beyes\b"),
    ("eyes", r"\beye\b"),
    ("cheeks", r"\bcheeks?\b"),

    # ── Single regions ──
    ("nose", r"\bnose\b"),
    ("nose", r"\bnostrils?\b"),
    ("nose", r"\bnasal\b"),
    ("mouth", r"\bmouth\b"),
    ("mouth", r"\blips?\b"),
    ("mouth", r"\bsmile\b"),
    ("jawline", r"\bjaw\s?line\b"),
    ("jawline", r"\bjaw\b"),
    ("jawline", r"\bchin\b"),
    ("forehead", r"\bforehead\b"),
    ("forehead", r"\bhairline\b"),
]

# Compile once. The tuple stores (region_key, compiled_pattern).
KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    (region, re.compile(pat, flags=re.IGNORECASE))
    for region, pat in _KEYWORD_TABLE
]

BBOX_COORD_SCALE = 1000


# ────────────────────────────────────────────────────────────────────
# Bounding-box construction
# ────────────────────────────────────────────────────────────────────

def _resolve_indices(region_key: str) -> Optional[list[int]]:
    """Map a region key (single or composite) to its landmark indices."""
    if region_key in FACIAL_REGIONS:
        return FACIAL_REGIONS[region_key]
    sub = COMPOSITE_REGIONS.get(region_key)
    if sub is None:
        return None
    indices: list[int] = []
    for s in sub:
        indices.extend(FACIAL_REGIONS.get(s, []))
    return indices or None


def landmarks_to_bbox_text(
    landmarks: list[list[float]],
    region_key: str,
) -> Optional[str]:
    """Convert landmark indices for a region to ``[y1,x1,y2,x2]`` text.

    Returns ``None`` if the region has no valid landmarks (e.g. partial
    occlusion truncated the mesh) — caller should skip enrichment for
    that region.
    """
    indices = _resolve_indices(region_key)
    if not indices:
        return None
    valid = [i for i in indices if 0 <= i < len(landmarks)]
    if not valid:
        return None

    pts = [landmarks[i] for i in valid]
    x_min = min(p[0] for p in pts)
    y_min = min(p[1] for p in pts)
    x_max = max(p[0] for p in pts)
    y_max = max(p[1] for p in pts)

    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    if x_max <= x_min or y_max <= y_min:
        return None

    ly1 = int(round(y_min * BBOX_COORD_SCALE))
    lx1 = int(round(x_min * BBOX_COORD_SCALE))
    ly2 = int(round(y_max * BBOX_COORD_SCALE))
    lx2 = int(round(x_max * BBOX_COORD_SCALE))
    return f"[{ly1},{lx1},{ly2},{lx2}]"


# ────────────────────────────────────────────────────────────────────
# Region detection in free text
# ────────────────────────────────────────────────────────────────────

def detect_regions_in_text(text: str) -> list[tuple[str, int, int]]:
    """Return all (region_key, start, end) matches in ``text``.

    Matches across keyword variants are merged so the same character
    span is never returned twice. Order is reading order (sorted by
    start offset). When two patterns overlap, the longest match wins,
    which preserves "left eye" over the shorter "eye".
    """
    candidates: list[tuple[int, int, str]] = []
    for region, pat in KEYWORD_PATTERNS:
        for m in pat.finditer(text):
            candidates.append((m.start(), m.end(), region))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: list[tuple[str, int, int]] = []
    last_end = -1
    for start, end, region in candidates:
        if start < last_end:
            continue  # already covered by a longer earlier match
        selected.append((region, start, end))
        last_end = end
    return selected


# ────────────────────────────────────────────────────────────────────
# Annotation injection
# ────────────────────────────────────────────────────────────────────

def annotate_answer(
    answer: str,
    landmarks: list[list[float]],
    grounded_in_question: Optional[set[str]] = None,
) -> tuple[str, list[str]]:
    """Inject ``[y1,x1,y2,x2]`` after the first mention of each region.

    Parameters
    ----------
    answer : str
        Original answer text.
    landmarks : list[list[float]]
        478 normalized landmarks for the corresponding image.
    grounded_in_question : set[str] | None
        Region keys named by the question. If a region is queried but
        not present in the answer, a grounded clause is prepended after
        the verdict so the model still learns the question-region link.

    Returns
    -------
    (enriched_answer, regions_grounded)
        ``regions_grounded`` is the list of region keys that received a
        bbox in the final text (useful for stats).
    """
    regions_grounded: list[str] = []
    matches = detect_regions_in_text(answer)

    # Inject bboxes from right to left so earlier offsets remain valid.
    seen: set[str] = set()
    edits: list[tuple[int, str]] = []  # (insert_position, text_to_insert)
    for region, _start, end in matches:
        if region in seen:
            continue
        bbox = landmarks_to_bbox_text(landmarks, region)
        if bbox is None:
            continue
        edits.append((end, f" {bbox}"))
        seen.add(region)
        regions_grounded.append(region)

    edits.sort(key=lambda x: x[0], reverse=True)
    out = answer
    for pos, ins in edits:
        out = out[:pos] + ins + out[pos:]

    # If the question asked about a region the answer never mentions,
    # prepend a single grounded clause right after the verdict so the
    # model still gets supervision linking question -> region bbox.
    if grounded_in_question:
        unmentioned = grounded_in_question - seen
        if unmentioned:
            extra_clauses: list[str] = []
            for region in sorted(unmentioned):
                bbox = landmarks_to_bbox_text(landmarks, region)
                if bbox is None:
                    continue
                pretty = region.replace("_", " ")
                extra_clauses.append(f"The {pretty} region {bbox} is the focus.")
                regions_grounded.append(region)
            if extra_clauses:
                # Insert extras after the verdict prefix (e.g. "Fake. ").
                m = re.match(
                    r"^\s*(Real|Fake)[\.,]\s*", out, flags=re.IGNORECASE,
                )
                if m:
                    insert_at = m.end()
                    out = (
                        out[:insert_at]
                        + " ".join(extra_clauses) + " "
                        + out[insert_at:]
                    )
                else:
                    out = " ".join(extra_clauses) + " " + out

    return out, regions_grounded


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Enrich DD-VQA answers with [y1,x1,y2,x2] grounding.",
    )
    p.add_argument("--ddvqa-jsonl", required=True, help="Original DD-VQA JSONL")
    p.add_argument(
        "--landmarks-jsonl", required=True,
        help="Landmarks JSONL produced by scripts/extract_landmarks.py",
    )
    p.add_argument("--output", required=True, help="Output enriched JSONL")
    p.add_argument(
        "--ground-question", action="store_true", default=True,
        help=(
            "If set (default), regions named in the question but missing "
            "from the answer get a prepended grounded clause."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading landmarks from %s", args.landmarks_jsonl)
    landmarks_db: dict[str, list[list[float]]] = {}
    with open(args.landmarks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            landmarks_db[rec["image"]] = rec["landmarks"]
    logger.info("Loaded landmarks for %d images", len(landmarks_db))

    logger.info("Processing %s", args.ddvqa_jsonl)
    total = 0
    enriched = 0
    no_landmarks = 0
    no_regions = 0
    region_counter: Counter = Counter()

    with open(args.ddvqa_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line.strip())
            total += 1

            img_rel = row.get("image", "")
            landmarks = landmarks_db.get(img_rel)
            if landmarks is None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                no_landmarks += 1
                continue

            question = row.get("question", "") or ""
            answer = row.get("answer", "") or ""

            grounded_q: set[str] = set()
            if args.ground_question and question:
                grounded_q = {r for r, _, _ in detect_regions_in_text(question)}

            new_answer, regions = annotate_answer(answer, landmarks, grounded_q)

            if not regions:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                no_regions += 1
                continue

            row["answer_original"] = answer
            row["answer"] = new_answer
            row["grounded_regions"] = regions
            for r in regions:
                region_counter[r] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            enriched += 1

    logger.info(
        "Done. total=%d, enriched=%d, no_landmarks=%d (kept original), "
        "no_regions=%d (kept original). Output: %s",
        total, enriched, no_landmarks, no_regions, args.output,
    )
    if region_counter:
        top = region_counter.most_common()
        logger.info("Region grounding frequency:")
        for region, count in top:
            logger.info("  %-15s %d", region, count)


if __name__ == "__main__":
    main()
