"""CLI training script for Anomalib v2.x models."""

import argparse

from src.anomaly.trainer import MODEL_REGISTRY, train_model


def main():
    parser = argparse.ArgumentParser(description="Train an anomaly detection model")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root folder")
    parser.add_argument("--dataset-name", required=True, help="Dataset name (used for checkpoint naming)")
    parser.add_argument(
        "--model",
        default="patchcore",
        choices=list(MODEL_REGISTRY.keys()),
        help="Anomaly detection model",
    )
    parser.add_argument("--backbone", default="resnet18", help="Backbone architecture")
    parser.add_argument(
        "--task",
        default="segmentation",
        choices=["segmentation", "classification"],
        help="Task type",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--output-dir", default="./results", help="Engine output directory")
    parser.add_argument("--models-dir", default="./models", help="Directory to save checkpoints")
    args = parser.parse_args()

    ckpt_path = train_model(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        model_name=args.model,
        backbone=args.backbone,
        task=args.task,
        image_size=args.image_size,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
    print(f"Training complete. Model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
