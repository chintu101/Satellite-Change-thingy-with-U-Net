"""Shared factory helpers for configs, models, losses, and optimizers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import yaml

from .losses import BCEDiceLoss, ContrastiveLoss
from .models import SiameseResNetUNet, SiameseUNet


def load_config(config_path: Path) -> Dict:
    """Load a YAML config file.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed configuration dictionary.
    """

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: Dict) -> torch.nn.Module:
    """Instantiate a model from config."""

    model_cfg = config["model"]
    model_name = model_cfg["name"]
    if model_name == "siamese_unet":
        return SiameseUNet(
            embed_dim=int(model_cfg["embed_dim"]),
            use_change_head=bool(model_cfg.get("use_change_head", False)),
        )
    if model_name == "siamese_resnet":
        return SiameseResNetUNet(
            embed_dim=int(model_cfg["embed_dim"]),
            pretrained=bool(model_cfg.get("pretrained", True)),
            use_cbam=bool(model_cfg.get("use_cbam", False)),
        )
    raise ValueError(f"Unsupported model.name '{model_name}'. Expected 'siamese_unet' or 'siamese_resnet'.")


def build_criterion(config: Dict, pos_weight: float | None = None) -> torch.nn.Module:
    """Instantiate a loss function from config."""

    loss_cfg = config["loss"]
    loss_name = loss_cfg["name"]
    if loss_name == "contrastive":
        return ContrastiveLoss(margin=float(loss_cfg["margin"]))
    if loss_name == "bce_dice":
        if pos_weight is None:
            raise ValueError("BCEDiceLoss requires pos_weight computed from the training set.")
        return BCEDiceLoss(
            pos_weight=pos_weight,
            bce_weight=float(loss_cfg["bce_weight"]),
            dice_weight=float(loss_cfg["dice_weight"]),
            smooth=float(loss_cfg["smooth"]),
        )
    raise ValueError(f"Unsupported loss.name '{loss_name}'. Expected 'contrastive' or 'bce_dice'.")


def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Build Adam optimizer, including backbone param groups when needed."""

    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    if model_cfg["name"] == "siamese_resnet":
        assert isinstance(model, SiameseResNetUNet)
        return torch.optim.Adam(
            [
                {"params": list(model.backbone_parameters()), "lr": float(opt_cfg["backbone_lr"])},
                {"params": list(model.decoder_head_parameters()), "lr": float(opt_cfg["decoder_lr"])},
            ],
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    """Build the configured scheduler."""

    sched_cfg = config["scheduler"]
    if sched_cfg["name"] != "cosine":
        raise ValueError(f"Unsupported scheduler.name '{sched_cfg['name']}'. Only 'cosine' is implemented.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(sched_cfg["t_max"]),
        eta_min=float(sched_cfg["eta_min"]),
    )
