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


def _try_get_ort_providers(inferencer: AnomalyInferencer) -> list[str] | None:
    sess = getattr(getattr(inferencer, "_inferencer", None), "session", None)
    if sess is None:
        return None
    try:
        return list(sess.get_providers())
    except Exception:
        return None


def _try_end_ort_profiling(inferencer: AnomalyInferencer) -> str | None:
    inner = getattr(inferencer, "_inferencer", None)
    fn = getattr(inner, "end_profiling", None)
    if callable(fn):
        return fn()
    return None


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
        "--openvino-int8-model",
        default=None,
        help="Optional path to OpenVINO INT8 IR (model.xml) to benchmark",
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
    parser.add_argument(
        "--openvino-compare-int8",
        action="store_true",
        help="Benchmark OpenVINO FP32 (onnx) and INT8 IR (model.xml)",
    )
    parser.add_argument(
        "--ort-report-providers",
        action="store_true",
        help="Include the actual ORT providers used in output JSON",
    )
    parser.add_argument(
        "--ort-profile",
        action="store_true",
        help="Enable ORT profiling (writes a JSON profile file)",
    )
    parser.add_argument(
        "--ort-profile-dir",
        default="./results/ort_profiles",
        help="Where to write ORT profile JSON files",
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
            onnx_enable_profiling=bool(args.ort_profile),
            onnx_profile_prefix=str(Path(args.ort_profile_dir) / "onnx_coreml")
            if args.ort_profile
            else None,
        )
        onnx_variants["onnx_cpu"] = AnomalyInferencer(
            args.onnx_model,
            backend="onnx",
            input_size=args.input_size,
            onnx_providers=["CPUExecutionProvider"],
            onnx_enable_profiling=bool(args.ort_profile),
            onnx_profile_prefix=str(Path(args.ort_profile_dir) / "onnx_cpu")
            if args.ort_profile
            else None,
        )
    else:
        onnx_variants["onnx"] = AnomalyInferencer(
            args.onnx_model,
            backend="onnx",
            input_size=args.input_size,
            onnx_providers=None,
            onnx_enable_profiling=bool(args.ort_profile),
            onnx_profile_prefix=str(Path(args.ort_profile_dir) / "onnx")
            if args.ort_profile
            else None,
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

        if args.ort_report_providers:
            providers = _try_get_ort_providers(inf)
            if providers is not None:
                results[name]["ort_providers"] = providers

        if args.ort_profile:
            Path(args.ort_profile_dir).mkdir(parents=True, exist_ok=True)
            profile_path = _try_end_ort_profiling(inf)
            if profile_path:
                results[name]["ort_profile"] = profile_path

    if args.openvino or args.openvino_compare_int8:
        ov_model = args.openvino_model or args.onnx_model

        # OpenVINO may not support all INT8 QDQ graphs depending on version/plugins.
        # If the provided ONNX is an int8 artifact, try the matching fp32 model.
        ov_model_path = Path(ov_model)
        if args.openvino_model is None and ov_model_path.name.endswith(".int8.onnx"):
            fp32_candidate = ov_model_path.with_name(
                ov_model_path.name.replace(".int8.onnx", ".onnx")
            )
            if fp32_candidate.is_file():
                ov_model = str(fp32_candidate)

        def _bench_openvino(key: str, model_path: str) -> None:
            try:
                ov_inf = AnomalyInferencer(
                    model_path,
                    backend="openvino",
                    input_size=args.input_size,
                    openvino_device="AUTO",
                )
                results[key] = _run_benchmark(
                    inferencer=ov_inf,
                    image=image,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                results[key]["model"] = model_path
            except Exception as e:
                results[key] = {
                    "error": f"{type(e).__name__}: {e}",
                    "model": model_path,
                }

        if args.openvino_compare_int8:
            _bench_openvino("openvino_auto_fp32", ov_model)
            if args.openvino_int8_model:
                _bench_openvino("openvino_auto_int8", args.openvino_int8_model)
            else:
                results["openvino_auto_int8"] = {
                    "error": "Missing --openvino-int8-model (expected path to model.xml)",
                }
        else:
            _bench_openvino("openvino_auto", ov_model)

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
