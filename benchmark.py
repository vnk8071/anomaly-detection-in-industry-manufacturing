"""Benchmark Torch vs ONNXRuntime inference.

Measures end-to-end latency of this repo's inference pipeline (pre/post included)
for a single image repeated N times.

Example:
  python benchmark.py \
    --image static/aqa.png \
    --torch-model models/patchcore_resnet18_aqa.pt \
    --onnx-model models/patchcore_resnet18_aqa.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from src.anomaly.inferencer import AnomalyInferencer


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _run_benchmark(
    *, inferencer: AnomalyInferencer, image: str, warmup: int, runs: int
) -> dict:
    for _ in range(max(0, int(warmup))):
        inferencer.predict(image)

    times_ms: list[float] = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        inferencer.predict(image)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    times_ms_sorted = sorted(times_ms)
    mean_ms = statistics.fmean(times_ms)
    p50 = _percentile(times_ms_sorted, 50)
    p95 = _percentile(times_ms_sorted, 95)
    p99 = _percentile(times_ms_sorted, 99)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return {
        "runs": len(times_ms),
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": times_ms_sorted[0],
        "max_ms": times_ms_sorted[-1],
        "fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark torch vs onnx backends")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--torch-model", required=True, help="Path to .pt/.ckpt model")
    parser.add_argument("--onnx-model", required=True, help="Path to .onnx model")
    parser.add_argument(
        "--openvino-model",
        default=None,
        help="Optional path to .onnx model for OpenVINO (defaults to --onnx-model)",
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help="ONNX preprocessing input size"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per backend"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Timed iterations per backend"
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write results as JSON",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow anomalib to unpickle torch checkpoints (sets TRUST_REMOTE_CODE=1)",
    )
    parser.add_argument(
        "--onnx-compare-coreml",
        action="store_true",
        help="Benchmark ONNXRuntime with CoreML EP (if available) and CPU-only",
    )
    parser.add_argument(
        "--openvino",
        action="store_true",
        help="Also benchmark OpenVINO Runtime (device AUTO)",
    )
    args = parser.parse_args()

    if args.trust_remote_code:
        os.environ["TRUST_REMOTE_CODE"] = "1"

    image = str(Path(args.image))

    torch_inf = AnomalyInferencer(
        args.torch_model, backend="torch", input_size=args.input_size
    )

    # ONNXRuntime provider selection
    onnx_variants: dict[str, AnomalyInferencer] = {}
    if args.onnx_compare_coreml:
        # Prefer CoreML if present, otherwise this will fall back to CPU.
        onnx_variants["onnx_coreml"] = AnomalyInferencer(
            args.onnx_model,
            backend="onnx",
            input_size=args.input_size,
            onnx_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        onnx_variants["onnx_cpu"] = AnomalyInferencer(
            args.onnx_model,
            backend="onnx",
            input_size=args.input_size,
            onnx_providers=["CPUExecutionProvider"],
        )
    else:
        onnx_variants["onnx"] = AnomalyInferencer(
            args.onnx_model,
            backend="onnx",
            input_size=args.input_size,
            onnx_providers=None,
        )

    results: dict = {
        "image": image,
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "torch": _run_benchmark(
            inferencer=torch_inf,
            image=image,
            warmup=args.warmup,
            runs=args.runs,
        ),
    }

    for name, inf in onnx_variants.items():
        results[name] = _run_benchmark(
            inferencer=inf,
            image=image,
            warmup=args.warmup,
            runs=args.runs,
        )

    if args.openvino:
        ov_model = args.openvino_model or args.onnx_model
        ov_inf = AnomalyInferencer(
            ov_model,
            backend="openvino",
            input_size=args.input_size,
            openvino_device="AUTO",
        )
        results["openvino_auto"] = _run_benchmark(
            inferencer=ov_inf,
            image=image,
            warmup=args.warmup,
            runs=args.runs,
        )

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
