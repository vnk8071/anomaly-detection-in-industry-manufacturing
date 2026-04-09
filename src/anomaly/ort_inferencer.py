from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class OrtPrediction:
    pred_score: float
    pred_label: str
    heat_map: np.ndarray
    pred_mask: np.ndarray
    segmentation: np.ndarray


class OnnxRuntimeInferencer:
    """ONNXRuntime inferencer.

    This is intentionally simple and expects the ONNX model to return an anomaly
    map (H x W) or (1 x 1 x H x W) that we can post-process similarly to the
    TorchInferencer path.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        input_size: int = 224,
        providers: list[str] | None = None,
        score_threshold: float = 0.5,
    ):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers or ort.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: str) -> dict:
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = _preprocess_bgr(original, self.input_size)
        outputs = self.session.run(None, {self.input_name: image})

        score = _extract_scalar_score(outputs)

        anomaly_map = None
        for out in outputs:
            try:
                anomaly_map = _coerce_anomaly_map(out)
                break
            except ValueError:
                continue

        if anomaly_map is None:
            # Some exports only return a scalar score. Fall back to a constant
            # map so downstream visualization still works.
            anomaly_map = np.full(
                (self.input_size, self.input_size), score, dtype=np.float32
            )

        # If no explicit scalar score output existed, derive it from anomaly map.
        if score is None:
            score = float(np.max(anomaly_map))
        label = "Anomalous" if score >= self.score_threshold else "Normal"

        heat_map = _colorize_anomaly_map(anomaly_map)
        mask = (anomaly_map >= self.score_threshold).astype(np.uint8)
        segmentation = _build_segmentation(original, mask)

        return {
            "pred_score": score,
            "pred_label": label,
            "heat_map": heat_map,
            "pred_mask": mask,
            "segmentation": segmentation,
        }


def _preprocess_bgr(image_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    # NCHW
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _coerce_anomaly_map(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    # Ignore scalars / singletons (typically score/label).
    if arr.size <= 1:
        raise ValueError(f"Not an anomaly map (scalar): shape={arr.shape}")
    # Common shapes: (1, 1, H, W), (1, H, W), (H, W)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported output shape for anomaly map: {arr.shape}")
    return arr.astype(np.float32)


def _extract_scalar_score(outputs: list[np.ndarray]) -> float | None:
    for out in outputs:
        arr = np.asarray(out)
        if arr.size == 1:
            try:
                return float(arr.reshape(()))
            except Exception:
                continue
    return None


def _colorize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def _build_segmentation(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = original.copy()
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
    return vis
