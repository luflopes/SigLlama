from __future__ import annotations

import os
from typing import Any

from PIL import Image, ImageDraw

# MediaPipe face-mesh landmark index groups
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
NOSE_TIP = [1, 2, 98, 327]


def plot_and_save(
    image_id: str,
    image_path: str,
    detections: list[dict[str, Any]],
    landmarks: list[dict[str, Any]],
    class_names: dict[int, str] | None = None,
    out_dir: str = "annotated",
) -> str:
    """Draw YOLO boxes and face-mesh landmarks on an image and save it.

    Returns the path of the saved annotated image.
    """
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    _draw_detections(draw, detections, w, h, class_names)
    _draw_landmarks(draw, landmarks, w, h)

    out_path = os.path.join(out_dir, f"{image_id}.jpg")
    img.save(out_path)
    return out_path


def _draw_detections(
    draw: ImageDraw.ImageDraw,
    detections: list[dict],
    w: int,
    h: int,
    class_names: dict[int, str] | None,
) -> None:
    for det in detections:
        cx, cy, bw, bh = det["bbox"]
        cls = det["class_id"]
        conf = det["confidence"]

        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        label = class_names[cls] if class_names else str(cls)
        draw.text((x1, max(0, y1 - 12)), f"{label} ({conf:.2f})", fill="red")


def _draw_landmarks(
    draw: ImageDraw.ImageDraw,
    landmarks: list[dict],
    w: int,
    h: int,
) -> None:
    if not landmarks:
        return

    lm_dict = landmarks[0] if isinstance(landmarks[0], dict) else None
    if lm_dict is None:
        return

    abs_pts = lm_dict.get("landmarks_absolute")
    lm_conf = lm_dict.get("confidence")

    if abs_pts is None and "landmarks_normalized" in lm_dict:
        abs_pts = [[x * w, y * h] for x, y in lm_dict["landmarks_normalized"]]

    if not abs_pts:
        return

    for x, y in abs_pts:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="cyan", outline="cyan")

    def _poly(idx_list: list[int], closed: bool = False, color: str = "cyan", width: int = 2) -> None:
        pts = [tuple(abs_pts[i]) for i in idx_list if i < len(abs_pts)]
        if len(pts) < 2:
            return
        if closed:
            pts.append(pts[0])
        draw.line(pts, fill=color, width=width)

    _poly(FACE_OVAL, closed=True, color="lime", width=2)
    _poly(LEFT_EYE, closed=True, color="yellow", width=2)
    _poly(RIGHT_EYE, closed=True, color="yellow", width=2)
    _poly(LIPS_OUTER, closed=True, color="orange", width=2)
    _poly(LIPS_INNER, closed=True, color="orange", width=1)

    for idx in NOSE_TIP:
        if idx < len(abs_pts):
            x, y = abs_pts[idx]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="blue", width=2)

    if lm_conf is not None:
        draw.text((10, 10), f"LM conf: {lm_conf:.2f}", fill="cyan")
