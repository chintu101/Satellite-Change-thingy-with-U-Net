"""Training entrypoint for baseline and upgraded satellite change detection models."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm.auto import tqdm

from .dataset import compute_positive_class_weight, create_dataloaders
from .evaluate import evaluate_model
from .factory import build_criterion, build_model, build_optimizer, build_scheduler, load_config
from .models import SiameseResNetUNet
from .utils import count_parameters, ensure_dir, save_history_plots, select_device, set_seed


def forward_for_loss(model: torch.nn.Module, batch: Dict[str, torch.Tensor], config: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run a model forward pass and compute the primary training output."""

    image_a = batch["image_a"]
    image_b = batch["image_b"]
    mask = batch["mask"]
    if config["loss"]["name"] == "contrastive":
        emb_a, emb_b, distance = model(image_a, image_b)
        return distance, {"embedding_a": emb_a, "embedding_b": emb_b, "mask": mask}
    output = model(image_a, image_b)
    if isinstance(output, dict):
        return output["probability"], {"probability": output["probability"], "mask": mask}
    _, _, probability = output
    return probability, {"probability": probability, "mask": mask}


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Dict,
) -> float:
    """Train the model for one epoch."""

    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        primary_output, outputs = forward_for_loss(model, batch, config)
        if config["loss"]["name"] == "contrastive":
            loss = criterion(outputs["embedding_a"], outputs["embedding_b"], outputs["mask"])
        else:
            loss = criterion(outputs["probability"], outputs["mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["training"]["grad_clip"]))
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(1, len(loader))


def maybe_update_backbone_freeze(model: torch.nn.Module, epoch_index: int, config: Dict) -> None:
    """Freeze or unfreeze the backbone according to the config."""

    if not isinstance(model, SiameseResNetUNet):
        return
    freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    if epoch_index < freeze_epochs:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()


def train_from_config(config: Dict, config_name: str = "<in-memory-config>") -> Dict:
    """Full training pipeline from an in-memory config dictionary.

    Args:
        config: Parsed configuration dictionary.
        config_name: User-facing config identifier for logs.

    Returns:
        Training summary dictionary.
    """

    set_seed(int(config["seed"]))
    device = select_device(config.get("device"))
    results_dir = ensure_dir(Path(config["paths"]["results_dir"]))
    checkpoints_dir = ensure_dir(results_dir / "checkpoints")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(config)
    pos_weight = compute_positive_class_weight(train_ds) if config["loss"]["name"] == "bce_dice" else None
    model = build_model(config).to(device)
    criterion = build_criterion(config, pos_weight=pos_weight).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }
    best_state = None
    best_val_loss = float("inf")
    best_path = checkpoints_dir / config["paths"]["best_checkpoint_name"]

    for epoch in range(int(config["training"]["epochs"])):
        maybe_update_backbone_freeze(model, epoch, config)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_metrics = evaluate_model(model, val_loader, criterion, device, config, split_name="val")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        print(
            f"Epoch {epoch + 1:02d}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint. Check the training loop for silent failures.")

    model.load_state_dict(best_state)
    save_history_plots(history, results_dir, prefix=config["paths"]["history_prefix"])
    test_metrics = evaluate_model(model, test_loader, criterion, device, config, split_name="test")

    if config["loss"]["name"] == "bce_dice" and test_metrics["f1"] < 0.30:
        print(
            "Diagnostic: test F1 is below 0.30. Check the positive pixel ratio in the dataset and recompute "
            f"pos_weight. Current pos_weight={pos_weight:.4f}."
        )

    summary = {
        "config": str(config_name),
        "device": str(device),
        "best_checkpoint": str(best_path),
        "parameter_count": count_parameters(model),
        "pos_weight": pos_weight,
        "history": history,
        "test_metrics": test_metrics,
    }
    print(
        f"Test metrics | F1={test_metrics['f1']:.4f} | Precision={test_metrics['precision']:.4f} | "
        f"Recall={test_metrics['recall']:.4f} | IoU={test_metrics['iou']:.4f}"
    )
    return summary


def train_model(config_path: Path) -> Dict:
    """Full training pipeline from a YAML config path."""

    config = load_config(config_path)
    return train_from_config(config, config_name=str(config_path))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a satellite image change detection model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
