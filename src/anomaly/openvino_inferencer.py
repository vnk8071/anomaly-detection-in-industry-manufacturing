from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OpenVINOInferencer:
    """OpenVINO inferencer.

    Loads an ONNX model via OpenVINO Runtime and runs inference on a single image.
    This mirrors the ORT inferencer behavior: it tries to find a 2D anomaly map
    among outputs, otherwise falls back to a constant map from a scalar score.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 224,
        device: str = "AUTO",
        score_threshold: float = 0.5,
    ):
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception:
            # Some OpenVINO installs expose Core at top-level.
            from openvino import Core  # type: ignore

        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.device = device
        self.score_threshold = float(score_threshold)

        core = Core()
        model = core.read_model(model=str(self.model_path))
        self.compiled = core.compile_model(model=model, device_name=self.device)
        self.input_layer = self.compiled.inputs[0]
        self.output_layers = list(self.compiled.outputs)

    def predict(self, image_path: str) -> dict:
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = _preprocess_bgr(original, self.input_size)
        out_map = self.compiled({self.input_layer: image})
        outputs = [out_map[o] for o in self.output_layers]

        score = _extract_scalar_score(outputs)

        anomaly_map = None
        for out in outputs:
            try:
                anomaly_map = _coerce_anomaly_map(out)
                break
            except ValueError:
                continue

        if anomaly_map is None:
            if score is None:
                raise ValueError(
                    "OpenVINO model outputs did not include a scalar score or anomaly map"
                )
            anomaly_map = np.full(
                (self.input_size, self.input_size), score, dtype=np.float32
            )

        if score is None:
            score = float(np.max(anomaly_map))

        label = "Anomalous" if score >= self.score_threshold else "Normal"

        heat_map = _colorize_anomaly_map(anomaly_map)
        mask = (anomaly_map >= self.score_threshold).astype(np.uint8)
        segmentation = _build_segmentation(original, mask)

        return {
            "pred_score": float(score),
            "pred_label": label,
            "heat_map": heat_map,
            "pred_mask": mask,
            "segmentation": segmentation,
        }


def _preprocess_bgr(image_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _coerce_anomaly_map(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.size <= 1:
        raise ValueError(f"Not an anomaly map (scalar): shape={arr.shape}")
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
