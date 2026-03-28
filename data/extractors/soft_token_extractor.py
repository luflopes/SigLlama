from __future__ import annotations

from typing import Any

from .yolo_detector import YoloDetector
from .landmark_extractor import LandmarkExtractor


class SoftTokenExtractor:
    """Orchestrates YOLO detection + MediaPipe face landmarks into a single
    soft-token record per image."""

    def __init__(self, yolo_model: str = "yolov8l.pt", yolo_conf: float = 0.25):
        self.detector = YoloDetector(yolo_model, conf=yolo_conf)
        self.landmarks = LandmarkExtractor()

    def process_image(
        self, image_id: str, img_path: str, caption: str
    ) -> dict[str, Any]:
        det = self.detector.detect(img_path)
        lms = self.landmarks.extract(img_path)

        output: dict[str, Any] = {
            "image_id": image_id,
            "image_path": img_path,
            "caption": caption,
            "detection_tokens": det,
            "landmark_tokens": [],
        }

        if lms is not None:
            output["landmark_tokens"].append({
                "landmarks_normalized": lms["normalized"],
                "landmarks_absolute": lms["absolute"],
                "confidence": lms["confidence"],
            })

        return output
