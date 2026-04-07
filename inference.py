"""CLI inference script for Anomalib v2.x models."""

import argparse
import os

import cv2

from src.anomaly.inferencer import AnomalyInferencer


def main():
    parser = argparse.ArgumentParser(description="Run anomaly inference on an image")
    parser.add_argument("--model", required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="./results_inference", help="Directory for output images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inferencer = AnomalyInferencer(args.model)
    result = inferencer.predict(args.image)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.output_dir, f"{stem}_heatmap.jpg"), result["heat_map"])
    cv2.imwrite(
        os.path.join(args.output_dir, f"{stem}_mask.jpg"),
        (result["pred_mask"] * 255).astype("uint8"),
    )
    cv2.imwrite(os.path.join(args.output_dir, f"{stem}_segment.jpg"), result["segmentation"])

    print(f"Score : {result['pred_score']:.4f}")
    print(f"Label : {result['pred_label']}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
