from pathlib import Path

import cv2
import numpy as np

from object_counter.counter import Detection
from object_counter.labels import COCO_LABELS


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path = "models/yolov8n.onnx",
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado: {self.model_path}. "
                "Rode `python scripts/download_model.py`."
            )

        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        image_height, image_width = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward()
        predictions = self._normalize_outputs(outputs)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        x_factor = image_width / self.input_size
        y_factor = image_height / self.input_size

        for prediction in predictions:
            class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            center_x, center_y, width, height = prediction[:4]
            left = int((center_x - width / 2) * x_factor)
            top = int((center_y - height / 2) * y_factor)
            box_width = int(width * x_factor)
            box_height = int(height * y_factor)

            boxes.append([left, top, box_width, box_height])
            confidences.append(confidence)
            class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections: list[Detection] = []
        for index in np.array(indexes).flatten():
            class_id = class_ids[int(index)]
            if class_id >= len(COCO_LABELS):
                continue

            x, y, width, height = boxes[int(index)]
            detections.append(
                Detection(
                    label=COCO_LABELS[class_id],
                    confidence=confidences[int(index)],
                    box=(x, y, width, height),
                )
            )

        return detections

    def _normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        predictions = np.squeeze(outputs)

        if predictions.ndim != 2:
            raise ValueError(f"Formato inesperado do modelo: {outputs.shape}")

        # YOLOv8 ONNX geralmente retorna (84, 8400); OpenCV pode entregar transposto.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        return predictions


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame_bgr.copy()

    for detection in detections:
        x, y, width, height = detection.box
        cv2.rectangle(output, (x, y), (x + width, y + height), (20, 180, 90), 2)
        label = f"{detection.label} {detection.confidence:.0%}"
        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 180, 90),
            2,
            cv2.LINE_AA,
        )

    return output

