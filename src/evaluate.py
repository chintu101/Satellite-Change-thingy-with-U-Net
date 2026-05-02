"""Evaluation, threshold sweep, metrics, and reporting utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from .dataset import compute_positive_class_weight, create_dataloaders
from .factory import build_criterion, build_model, load_config
from .utils import MetricBundle, ensure_dir, metrics_from_counts, plot_comparison_bar_chart, segmentation_counts, select_device

try:
    from cuml.metrics import precision_recall_fscore_support as cuml_prfs  # type: ignore

    HAS_CUML = True
except Exception:
    cuml_prfs = None
    HAS_CUML = False


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Dict,
    split_name: str,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate a model on a dataset split."""

    model.eval()
    total_loss = 0.0
    tp = torch.tensor(0.0, device=device)
    fp = torch.tensor(0.0, device=device)
    fn = torch.tensor(0.0, device=device)
    threshold = float(threshold if threshold is not None else config["evaluation"]["threshold"])

    for batch in tqdm(loader, desc=f"Eval-{split_name}", leave=False):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        mask = batch["mask"]
        if config["loss"]["name"] == "contrastive":
            emb_a, emb_b, distance = model(batch["image_a"], batch["image_b"])
            loss = criterion(emb_a, emb_b, mask)
            probability = 1.0 - (distance / float(config["loss"]["margin"])).clamp(0.0, 1.0)
        else:
            output = model(batch["image_a"], batch["image_b"])
            probability = output["probability"] if isinstance(output, dict) else output[2]
            loss = criterion(probability, mask)
        total_loss += float(loss.detach().cpu())
        batch_tp, batch_fp, batch_fn = segmentation_counts(probability, mask, threshold=threshold)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    metrics = metrics_from_counts(tp, fp, fn)
    return {
        "loss": total_loss / max(1, len(loader)),
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "iou": metrics.iou,
    }


@torch.no_grad()
def threshold_sweep(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: torch.Tensor,
    output_dir: Path,
    margin: float = 2.0,
) -> Dict[str, object]:
    """Sweep thresholds over the validation set mostly on GPU tensors."""

    ensure_dir(output_dir)
    model.eval()
    rows: List[Dict[str, float]] = []

    for threshold in thresholds.to(device):
        tp = torch.tensor(0.0, device=device)
        fp = torch.tensor(0.0, device=device)
        fn = torch.tensor(0.0, device=device)
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(batch["image_a"], batch["image_b"])
            if isinstance(output, dict):
                probability = output["probability"]
            else:
                _, _, distance = output
                probability = 1.0 - (distance / margin).clamp(0.0, 1.0)
            batch_tp, batch_fp, batch_fn = segmentation_counts(probability, batch["mask"], threshold=float(threshold.item()))
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
        metrics = metrics_from_counts(tp, fp, fn)
        rows.append(
            {
                "threshold": float(threshold.detach().cpu()),
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
            }
        )

    best_row = max(rows, key=lambda row: row["f1"])
    best_threshold = best_row["threshold"]

    plt.figure(figsize=(8, 5))
    plt.plot([row["threshold"] for row in rows], [row["f1"] for row in rows], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold Sweep")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot([row["recall"] for row in rows], [row["precision"] for row in rows], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    header = f"{'Threshold':>10} | {'F1':>8} | {'Precision':>10} | {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['threshold']:10.4f} | {row['f1']:8.4f} | {row['precision']:10.4f} | {row['recall']:8.4f}")
    print(
        f"Best threshold={best_threshold:.4f} | F1={best_row['f1']:.4f} | "
        f"Precision={best_row['precision']:.4f} | Recall={best_row['recall']:.4f}"
    )

    return {
        "best_threshold": best_threshold,
        "rows": rows,
        "best_row": best_row,
    }


def compare_models(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    """Create a bar chart from model comparison rows."""

    plot_comparison_bar_chart(rows, output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a trained change detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"), help="YAML config file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to load.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override threshold.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = load_config(args.config)
    device = select_device(config.get("device"))
    train_ds, _, _, _, val_loader, test_loader = create_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader
    pos_weight = 1.0 if config["loss"]["name"] == "contrastive" else compute_positive_class_weight(train_ds)
    criterion = build_criterion(config, pos_weight=pos_weight).to(device)
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    metrics = evaluate_model(model, loader, criterion, device, config, split_name=args.split, threshold=args.threshold)
    print(metrics)


if __name__ == "__main__":
    main()
