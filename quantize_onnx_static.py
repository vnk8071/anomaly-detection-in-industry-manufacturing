"""CLI: static INT8 quantization for ONNX models using ONNXRuntime.

Static quantization requires a calibration set (representative images).

Example:
  python quantize_onnx_static.py \
    --onnx models/patchcore_resnet18_aqa.onnx \
    --out models/patchcore_resnet18_aqa.int8.onnx \
    --calib-dir data/aqa/train/good \
    --num-samples 200 \
    --input-size 224
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _iter_images(calib_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in calib_dir.rglob("*") if p.suffix.lower() in exts])


def _preprocess_bgr(image_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    return x


class _ImageCalibrationDataReader:
    def __init__(
        self,
        *,
        input_name: str,
        image_paths: list[Path],
        input_size: int,
    ):
        self.input_name = input_name
        self.image_paths = image_paths
        self.input_size = int(input_size)
        self._idx = 0

    def get_next(self):  # onnxruntime expects this exact name
        if self._idx >= len(self.image_paths):
            return None
        p = self.image_paths[self._idx]
        self._idx += 1
        img = cv2.imread(str(p))
        if img is None:
            # Skip unreadable images.
            return self.get_next()
        x = _preprocess_bgr(img, self.input_size)
        return {self.input_name: x}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for ONNX models"
    )
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--out", required=True, help="Output quantized ONNX model")
    parser.add_argument(
        "--calib-dir", required=True, help="Directory of calibration images"
    )
    parser.add_argument("--input-size", type=int, default=224, help="Model input size")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Max number of calibration images to use",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization",
    )
    args = parser.parse_args()

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    import onnxruntime as ort

    onnx_in = Path(args.onnx)
    onnx_out = Path(args.out)
    calib_dir = Path(args.calib_dir)

    if not onnx_in.is_file():
        raise SystemExit(f"ONNX model not found: {onnx_in}")
    if not calib_dir.is_dir():
        raise SystemExit(f"Calibration dir not found: {calib_dir}")

    sess = ort.InferenceSession(str(onnx_in), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    images = _iter_images(calib_dir)
    if not images:
        raise SystemExit(f"No images found under: {calib_dir}")
    images = images[: max(1, int(args.num_samples))]

    reader = _ImageCalibrationDataReader(
        input_name=input_name,
        image_paths=images,
        input_size=args.input_size,
    )

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=bool(args.per_channel),
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(f"Wrote quantized model: {onnx_out}")


if __name__ == "__main__":
    main()
