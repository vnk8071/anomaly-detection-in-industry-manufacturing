"""CLI inference script for Anomalib v2.x models."""

import argparse
import os

import cv2

from src.anomaly.inferencer import AnomalyInferencer


def main():
    parser = argparse.ArgumentParser(description="Run anomaly inference on an image")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model file (.pt/.ckpt for torch backend, .onnx for onnx backend)",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "onnx", "openvino"],
        help="Inference backend",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Model input size (used for ONNX preprocessing)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow anomalib to unpickle torch checkpoints (sets TRUST_REMOTE_CODE=1)",
    )
    parser.add_argument(
        "--onnx-coreml",
        action="store_true",
        help="Prefer CoreMLExecutionProvider for ONNXRuntime",
    )
    parser.add_argument(
        "--onnx-no-coreml",
        action="store_true",
        help="Force CPUExecutionProvider only for ONNXRuntime",
    )
    parser.add_argument(
        "--openvino-device",
        default="AUTO",
        help="OpenVINO device name (default: AUTO)",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--output-dir",
        default="./results_inference",
        help="Directory for output images",
    )
    args = parser.parse_args()

    if args.trust_remote_code:
        os.environ["TRUST_REMOTE_CODE"] = "1"

    onnx_providers = None
    if args.backend == "onnx":
        if args.onnx_coreml and args.onnx_no_coreml:
            raise SystemExit("Choose only one of --onnx-coreml or --onnx-no-coreml")
        if args.onnx_no_coreml:
            onnx_providers = ["CPUExecutionProvider"]
        elif args.onnx_coreml:
            onnx_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    os.makedirs(args.output_dir, exist_ok=True)
    inferencer = AnomalyInferencer(
        args.model,
        backend=args.backend,
        input_size=args.input_size,
        onnx_providers=onnx_providers,
        openvino_device=args.openvino_device,
    )
    result = inferencer.predict(args.image)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(
        os.path.join(args.output_dir, f"{stem}_heatmap.jpg"), result["heat_map"]
    )
    cv2.imwrite(
        os.path.join(args.output_dir, f"{stem}_mask.jpg"),
        (result["pred_mask"] * 255).astype("uint8"),
    )
    cv2.imwrite(
        os.path.join(args.output_dir, f"{stem}_segment.jpg"), result["segmentation"]
    )

    print(f"Score : {result['pred_score']:.4f}")
    print(f"Label : {result['pred_label']}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
