import argparse
import numpy as np
import time

import cv2
from skimage.segmentation import mark_boundaries
from custom_inference import TorchInferencer
from anomalib.post_processing.post_process import (
    add_anomalous_label,
    add_normal_label,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file", default='configs/patchcore_grid.yaml')
    parser.add_argument("--weight", type=str, help="Path to model weights", default='models/patchcore_grid.ckpt')
    parser.add_argument("--image", type=str, default='samples/006_grid.png')
    return parser.parse_args()

def get_inferencer(args):
    """Parse args and open inferencer.
    Args:
        config_path (Path): Path to model configuration file or the name of the model.
        weight_path (Path): Path to model weights.
        meta_data_path (Optional[Path], optional): Metadata is required for OpenVINO models. Defaults to None.
    Raises:
        ValueError: If unsupported model weight is passed.
    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """
    if args.weight.endswith(".ckpt"):
        inferencer = TorchInferencer(config=args.config, model_source=args.weight)
    else:
        raise ValueError("Model checkpoint should be .ckpt")
    return inferencer

def inference(args, inferencer):
    filename = args.image.rsplit('/')[-1]
    image = cv2.imread(args.image)
    predictions = inferencer.predict(image=image)
    output = mark_boundaries(
                predictions.heat_map, predictions.pred_mask, color=(1, 0, 0), mode="thick"
            )
    if predictions.pred_label:
        output = add_anomalous_label(output, predictions.pred_score)
    else:
        output = add_normal_label(output, 1 - predictions.pred_score)
    output = (output * 255).astype(np.uint8)
    cv2.imwrite("results_inference/"+ filename, output)
    return (predictions.pred_score, predictions.pred_label)

if __name__ == "__main__":
    start = time.time()
    args = get_args()
    inferencer = get_inferencer(args)
    pred_score, pred_label = inference(args, inferencer)
    end = time.time()
    print(f"Anomaly score: {pred_score:.4f} - Target label: {pred_label}")
    print(f"Time elapsed: {end - start} second")

