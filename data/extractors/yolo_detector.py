from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class YoloDetector:
    """Wraps YOLOv8 for object detection with normalised bbox output.

    Each detection is a dict with keys:
        bbox       – [cx, cy, w, h] normalised to [0, 1]
        class_id   – COCO class index
        class_name – human-readable class label
        confidence – detection score
    """

    def __init__(self, model_path: str = "yolov8l.pt", conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names: dict[int, str] = self.model.names

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        results = self.model.predict(image_path, conf=self.conf, verbose=False)
        if len(results) == 0:
            return []

        h, w = results[0].orig_shape
        detections: list[dict[str, Any]] = []

        for box in results[0].boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])

            x1, y1, x2, y2 = xyxy.tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            detections.append({
                "bbox": [float(cx), float(cy), float(bw), float(bh)],
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, str(cls_id)),
                "confidence": conf,
            })

        return detections
