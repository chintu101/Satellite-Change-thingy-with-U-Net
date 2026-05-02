"""Utility helpers for reproducibility, visualization, metrics, and timing."""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class MetricBundle:
    """Container for segmentation metrics.

    Args:
        f1: Binary F1 score.
        precision: Binary precision.
        recall: Binary recall.
        iou: Binary intersection-over-union.
    """

    f1: float
    precision: float
    recall: float
    iou: float


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Global random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path for convenience.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def select_device(device: Optional[str] = None) -> torch.device:
    """Resolve a torch device from a config value.

    Args:
        device: Explicit device string or ``None``.

    Returns:
        A resolved torch device.
    """

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization.

    Args:
        image: Tensor shaped ``(3, H, W)``.

    Returns:
        De-normalized image clipped to ``[0, 1]``.
    """

    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def segmentation_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute TP/FP/FN counts on the current device.

    Args:
        probabilities: Probability map shaped ``(B, 1, H, W)``.
        targets: Binary target map shaped ``(B, 1, H, W)``.
        threshold: Decision threshold.

    Returns:
        Tuple of tensors ``(tp, fp, fn)``.
    """

    preds = (probabilities >= threshold).to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)
    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    return tp, fp, fn


def metrics_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    smooth: float = 1e-6,
) -> MetricBundle:
    """Convert TP/FP/FN counts into segmentation metrics.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        smooth: Numerical stability constant.

    Returns:
        Metric bundle on CPU as Python floats.
    """

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2.0 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)
    return MetricBundle(
        f1=float(f1.detach().cpu()),
        precision=float(precision.detach().cpu()),
        recall=float(recall.detach().cpu()),
        iou=float(iou.detach().cpu()),
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@contextmanager
def timer() -> Iterator[MutableMapping[str, float]]:
    """Simple timing context manager.

    Returns:
        Mutable mapping containing ``elapsed`` after exit.
    """

    result: MutableMapping[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def save_history_plots(
    history: Mapping[str, Sequence[float]],
    output_dir: Path,
    prefix: str,
) -> None:
    """Save training curves for loss and metrics.

    Args:
        history: Training history lists.
        output_dir: Directory for plots.
        prefix: Output filename prefix.
    """

    ensure_dir(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss.png", dpi=200)
    plt.close()

    if "val_f1" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["val_f1"], label="Val F1")
        if "val_precision" in history:
            plt.plot(epochs, history["val_precision"], label="Val Precision")
        if "val_recall" in history:
            plt.plot(epochs, history["val_recall"], label="Val Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_metrics.png", dpi=200)
        plt.close()


def gaussian_weight_map(tile_size: int, sigma_scale: float = 0.125) -> torch.Tensor:
    """Create a 2D Gaussian weight map for sliding-window blending.

    Args:
        tile_size: Square tile size in pixels.
        sigma_scale: Gaussian sigma as a fraction of tile size.

    Returns:
        Weight map shaped ``(tile_size, tile_size)``.
    """

    coords = torch.arange(tile_size, dtype=torch.float32)
    center = (tile_size - 1) / 2.0
    sigma = tile_size * sigma_scale
    gauss_1d = torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))
    weight = torch.outer(gauss_1d, gauss_1d)
    return weight / weight.max().clamp_min(1e-6)


def plot_comparison_bar_chart(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Plot F1 comparison bars for multiple models.

    Args:
        rows: Metric rows containing ``model`` and ``f1``.
        output_path: Save path.
    """

    labels = [str(row["model"]) for row in rows]
    values = [float(row["f1"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#2f7ed8")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def visualize_predictions_grid(
    samples: Sequence[Mapping[str, object]],
    output_path: Path,
    title: str,
) -> None:
    """Save a qualitative grid with five columns.

    Args:
        samples: Sequence containing tensors or arrays for visualization.
        output_path: Destination image path.
        title: Figure title.
    """

    num_rows = len(samples)
    fig, axes = plt.subplots(num_rows, 5, figsize=(18, max(4, num_rows * 3)))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ["Time1", "Time2", "GT Mask", "Probability Map", "Prediction"]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header)

    for row, sample in enumerate(samples):
        image_a = sample["image_a"]
        image_b = sample["image_b"]
        gt_mask = sample["mask"]
        prob_map = sample["probability"]
        pred_mask = sample["prediction"]
        row_f1 = sample.get("f1")

        for axis in axes[row]:
            axis.axis("off")

        axes[row, 0].imshow(image_a)
        axes[row, 1].imshow(image_b)
        axes[row, 2].imshow(gt_mask, cmap="gray")
        axes[row, 3].imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[row, 4].imshow(pred_mask, cmap="gray")

        label = f"F1={row_f1:.3f}" if row_f1 is not None else f"Sample {row + 1}"
        axes[row, 0].set_ylabel(label, rotation=90, size=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

