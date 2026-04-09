from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from anomalib.deploy import TorchInferencer

from src.anomaly.ort_inferencer import OnnxRuntimeInferencer
from src.anomaly.openvino_inferencer import OpenVINOInferencer


class AnomalyInferencer:
    """Thin wrapper around inferencer backends.

    Backends:
    - torch: anomalib.deploy.TorchInferencer on `.pt`/`.ckpt`
    - onnx: ONNXRuntime on `.onnx`
    - openvino: OpenVINO Runtime on `.onnx`
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "torch",
        input_size: int = 224,
        onnx_providers: list[str] | None = None,
        openvino_device: str = "AUTO",
        onnx_enable_profiling: bool = False,
        onnx_profile_prefix: str | None = None,
    ):
        self.backend = backend
        if backend == "torch":
            self._inferencer = TorchInferencer(path=Path(model_path))
        elif backend == "onnx":
            self._inferencer = OnnxRuntimeInferencer(
                model_path,
                input_size=input_size,
                providers=onnx_providers,
                enable_profiling=onnx_enable_profiling,
                profile_prefix=onnx_profile_prefix,
            )
        elif backend == "openvino":
            self._inferencer = OpenVINOInferencer(
                model_path,
                input_size=input_size,
                device=openvino_device,
            )
        else:
            raise ValueError("backend must be one of: 'torch', 'onnx', 'openvino'")

    def predict(self, image_path: str) -> dict:
        """Run inference on a single image.

        Returns a dict with keys:
            pred_score   – float anomaly score
            pred_label   – "Normal" or "Anomalous"
            heat_map     – BGR ndarray (H x W x 3) colourised heatmap
            pred_mask    – uint8 ndarray (H x W) binary mask
            segmentation – BGR ndarray (H x W x 3) contour overlay
        """
        if self.backend in {"onnx", "openvino"}:
            # ONNXRuntimeInferencer returns the final dict already.
            return self._inferencer.predict(image_path)

        raw = self._inferencer.predict(image=Path(image_path))
        pred = raw.to_numpy()

        # Batch dim: (1, H, W) → (H, W)
        anomaly_map = pred.anomaly_map[0]
        mask = pred.pred_mask[0].astype(np.uint8)
        score = float(pred.pred_score[0])
        label = "Anomalous" if bool(pred.pred_label[0]) else "Normal"

        heat_map = _colorize_anomaly_map(anomaly_map)
        original = cv2.imread(image_path)
        segmentation = _build_segmentation(original, mask)

        return {
            "pred_score": score,
            "pred_label": label,
            "heat_map": heat_map,
            "pred_mask": mask,
            "segmentation": segmentation,
        }


def _colorize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def _build_segmentation(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if original is None or mask is None:
        return original
    vis = original.copy()
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
    return vis
