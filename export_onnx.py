"""CLI: export saved Torch model (.pt) to ONNX.

Example:
  python export_onnx.py --pt models/patchcore_resnet18_aqa.pt --onnx models/patchcore_resnet18_aqa.onnx
"""

import argparse

from src.anomaly.onnx_export import export_pt_to_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pt model to ONNX")
    parser.add_argument("--pt", required=True, help="Path to .pt saved by this repo")
    parser.add_argument("--onnx", required=True, help="Output .onnx path")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use PyTorch's dynamo-based ONNX exporter (may fail for some models)",
    )
    args = parser.parse_args()

    out = export_pt_to_onnx(
        pt_path=args.pt,
        onnx_path=args.onnx,
        input_size=args.input_size,
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"Exported ONNX model to: {out}")


if __name__ == "__main__":
    main()
