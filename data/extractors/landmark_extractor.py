"""Face landmark extraction using MediaPipe Tasks (>= 0.10.31).

Downloads the ``face_landmarker.task`` model automatically on first use
and caches it under ``~/.cache/mediapipe/``.
"""
from __future__ import annotations

import os
import urllib.request
import warnings
from typing import Any

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", message=".*mediapipe.*")

import mediapipe as mp  # noqa: E402

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
_MODEL_PATH = os.path.join(_CACHE_DIR, "face_landmarker.task")


def _ensure_model() -> str:
    """Download the face-landmarker model if it is not cached yet."""
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"[LandmarkExtractor] Downloading face_landmarker.task → {_MODEL_PATH}")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class LandmarkExtractor:
    """Extracts 478 face-mesh landmarks via MediaPipe FaceLandmarker (Tasks API).

    Returns normalised ([0,1]) and absolute (pixel) coordinates plus a
    confidence score (fraction of landmarks inside the image bounds).
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        model_path: str | None = None,
    ):
        model_path = model_path or _ensure_model()

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(self, image_path: str) -> dict[str, Any] | None:
        """Return landmark dict or ``None`` when no face is found."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(img),
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        normalized: list[list[float]] = []
        absolute: list[list[float]] = []
        visible_points = 0

        for p in landmarks:
            x, y = float(p.x), float(p.y)
            normalized.append([x, y])
            absolute.append([x * w, y * h])
            if 0 <= x <= 1 and 0 <= y <= 1:
                visible_points += 1

        return {
            "normalized": normalized,
            "absolute": absolute,
            "confidence": visible_points / len(normalized),
        }

    def close(self) -> None:
        self._landmarker.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
