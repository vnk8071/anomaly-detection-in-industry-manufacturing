"""ONNX export helpers.

This repo currently saves trained models as a Torch pickle:

    torch.save({"model": model}, "models/<name>.pt")

To run inference with ONNXRuntime we need an `.onnx` graph. The exact export
mechanism depends on the upstream Anomalib version and model type. Rather than
guessing internals here, we provide a minimal, explicit export entrypoint that
relies on anomalib + torch being installed.
"""

from __future__ import annotations

from pathlib import Path


def export_pt_to_onnx(
    *,
    pt_path: str | Path,
    onnx_path: str | Path,
    input_size: int = 224,
    opset: int = 17,
    dynamo: bool = False,
) -> Path:
    """Export a saved `{ "model": <AnomalibModule> }` .pt file to ONNX.

    Notes:
    - This assumes the stored object exposes a torch `nn.Module` compatible
      forward for tracing/export.
    - Some anomaly models (e.g. PatchCore/PaDiM) include non-neural parts and
      may not be exportable as a single ONNX graph without Anomalib's dedicated
      exporters. In that case, use Anomalib's official export utilities.
    """

    import torch

    pt_path = Path(pt_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # PyTorch 2.6 changed `torch.load` default `weights_only` to True, which
    # blocks loading pickled module objects by default. Our checkpoints store a
    # full Anomalib model object, so we need `weights_only=False`.
    # Only do this for checkpoints you trust.
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without `weights_only`.
        payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"Unexpected checkpoint format at {pt_path}. Expected a dict with key 'model'."
        )

    model = payload["model"]
    model.eval()

    # Common image shape for anomalib models: NCHW float tensor.
    dummy = torch.randn(1, 3, int(input_size), int(input_size), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        # PyTorch 2.6 defaults to the new dynamo exporter which may fail on
        # data-dependent control flow in Anomalib modules.
        dynamo=bool(dynamo),
        input_names=["image"],
        output_names=["output"],
        dynamic_axes={"image": {0: "batch"}},
    )

    return onnx_path
