import logging
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Draem, Fastflow, Padim, Patchcore, Stfpm

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "fastflow": Fastflow,
    "draem": Draem,
    "stfpm": Stfpm,
}

# Models that accept a backbone parameter
BACKBONE_MODELS = {"patchcore", "padim", "fastflow", "stfpm"}

_NORMAL_NAMES = {"good", "normal", "ok"}


def _find_normal_dir(dataset_root: Path) -> str:
    """Find the normal (defect-free) training subdirectory."""
    for candidate in ["train/good", "train/normal", "train/ok", "train"]:
        if (dataset_root / candidate).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find normal training dir under {dataset_root}. "
        "Expected one of: train/good, train/normal, train/ok"
    )


def _find_abnormal_dir(dataset_root: Path) -> str | None:
    """Find the first non-normal subdirectory under test/.

    Returns None if no abnormal directory is found.
    """
    test_dir = dataset_root / "test"
    if not test_dir.is_dir():
        return None
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name.lower() not in _NORMAL_NAMES:
            logger.info("Auto-detected abnormal dir: test/%s", d.name)
            return f"test/{d.name}"
    return None


def train_model(
    dataset_root: str,
    dataset_name: str,
    model_name: str = "patchcore",
    backbone: str = "resnet18",
    output_dir: str = "./results",
    models_dir: str = "./models",
) -> str:
    """Train and export an anomaly detection model using Anomalib v2.x.

    The dataset at ``dataset_root`` must contain:
        train/<normal>/   – defect-free training images (good / normal / ok)
        test/<abnormal>/  – defective test images (any non-normal folder name)

    Returns:
        Path to the exported ``<name>.pt`` file saved in ``models_dir``.
    """
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}"
        )

    model_kwargs = {}
    if model_name in BACKBONE_MODELS and backbone:
        model_kwargs["backbone"] = backbone
    model = model_cls(**model_kwargs)

    root = Path(dataset_root)
    normal_dir = _find_normal_dir(root)
    abnormal_dir = _find_abnormal_dir(root)
    logger.info(
        "Dataset: root=%s normal_dir=%s abnormal_dir=%s",
        dataset_root, normal_dir, abnormal_dir,
    )

    datamodule = Folder(
        name=dataset_name,
        root=str(root),
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        train_batch_size=32,
        eval_batch_size=1,
    )

    engine = Engine(
        default_root_dir=output_dir,
        accelerator="auto",
        devices=1,
        max_epochs=1,
    )

    logger.info("Training: model=%s backbone=%s dataset=%s", model_name, backbone, dataset_name)
    engine.fit(model=model, datamodule=datamodule)

    logger.info("Running test evaluation.")
    engine.test(model=model, datamodule=datamodule)

    # Save in the format TorchInferencer expects: {"model": <AnomalibModule>}
    model_stem = f"{model_name}_{backbone}_{dataset_name}"
    dest = Path(models_dir) / f"{model_stem}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save({"model": model}, dest)  # nosec B614
    logger.info("Model saved to %s", dest)
    return str(dest)
