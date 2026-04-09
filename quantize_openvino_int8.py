"""CLI: OpenVINO INT8 post-training quantization (PTQ) using NNCF.

This is the recommended route for OpenVINO INT8 because importing an already
INT8 QDQ ONNX graph is often unsupported or unstable across OpenVINO versions.

Flow:
  FP32 ONNX -> OpenVINO Model -> NNCF PTQ (INT8) -> OpenVINO IR (xml/bin)

Example:
  python quantize_openvino_int8.py \
    --onnx models/patchcore_resnet18_aqa.onnx \
    --out-dir models/openvino_int8/patchcore_resnet18_aqa \
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenVINO INT8 PTQ (NNCF) from FP32 ONNX + calibration images"
    )
    parser.add_argument("--onnx", required=True, help="Input FP32 ONNX model")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for OpenVINO IR (xml/bin)",
    )
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
    args = parser.parse_args()

    onnx_in = Path(args.onnx)
    out_dir = Path(args.out_dir)
    calib_dir = Path(args.calib_dir)

    if not onnx_in.is_file():
        raise SystemExit(f"ONNX model not found: {onnx_in}")
    if not calib_dir.is_dir():
        raise SystemExit(f"Calibration dir not found: {calib_dir}")

    # OpenVINO + NNCF imports
    try:
        from openvino.runtime import Core, serialize  # type: ignore
    except Exception:
        from openvino import Core, serialize  # type: ignore

    import nncf

    images = _iter_images(calib_dir)
    if not images:
        raise SystemExit(f"No images found under: {calib_dir}")
    images = images[: max(1, int(args.num_samples))]

    core = Core()
    ov_model = core.read_model(model=str(onnx_in))

    # Use the first input as image tensor.
    input_layer = ov_model.inputs[0]
    input_name = input_layer.get_any_name()

    def transform_fn(image_path: Path) -> dict[str, np.ndarray]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return {input_name: _preprocess_bgr(img, args.input_size)}

    class _Dataset:
        def __len__(self):
            return len(images)

        def __iter__(self):
            for p in images:
                try:
                    yield transform_fn(p)
                except Exception:
                    continue

    dataset = nncf.Dataset(_Dataset())

    quantized_model = nncf.quantize(
        ov_model,
        dataset,
        preset=nncf.QuantizationPreset.MIXED,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "model.xml"
    bin_path = out_dir / "model.bin"
    serialize(quantized_model, str(xml_path), str(bin_path))
    print(f"Wrote OpenVINO INT8 IR: {xml_path} + {bin_path}")


if __name__ == "__main__":
    main()
