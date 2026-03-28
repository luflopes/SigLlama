from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
warnings.filterwarnings("ignore", category=FutureWarning, module="mediapipe")
warnings.filterwarnings("ignore", message=".*mediapipe.*")

import mediapipe as mp  # noqa: E402


class LandmarkExtractor:
    """Extracts 468 face-mesh landmarks via MediaPipe.

    Returns normalised ([0,1]) and absolute (pixel) coordinates plus a
    confidence score (fraction of landmarks inside the image bounds).
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image_path: str) -> dict[str, Any] | None:
        """Return landmark dict or ``None`` when no face is found."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        arr = np.array(img)

        results = self.face_mesh.process(arr)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        normalized: list[list[float]] = []
        absolute: list[list[float]] = []
        visible_points = 0

        for p in lm:
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
