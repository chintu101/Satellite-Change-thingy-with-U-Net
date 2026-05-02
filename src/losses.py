"""Loss functions used by the satellite change detection models."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Pixel-wise contrastive loss for embedding-distance learning.

    Args:
        margin: Margin applied to unchanged pixels.
    """

    def __init__(self, margin: float = 2.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the pixel-wise contrastive objective.

        Args:
            f1: Embeddings from image A shaped ``(B, C, H, W)``.
            f2: Embeddings from image B shaped ``(B, C, H, W)``.
            mask: Binary change mask shaped ``(B, 1, H, W)``.

        Returns:
            Scalar loss tensor.
        """

        dist = torch.norm(f1 - f2, p=2, dim=1, keepdim=True)
        loss_pos = mask * dist.pow(2)
        loss_neg = (1.0 - mask) * F.relu(self.margin - dist).pow(2)
        return 0.5 * (loss_pos + loss_neg).mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCE plus Dice loss for binary segmentation.

    Args:
        pos_weight: Positive class weighting factor for BCE.
        bce_weight: Weight for BCE term.
        dice_weight: Weight for Dice term.
        smooth: Smoothing constant for Dice.
    """

    def __init__(
        self,
        pos_weight: float,
        bce_weight: float = 0.4,
        dice_weight: float = 0.6,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be > 0. Recompute it from the training set changed-pixel ratio.")
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE-Dice objective on probabilities.

        Args:
            probabilities: Predicted probabilities in ``[0, 1]``.
            targets: Binary masks in ``{0, 1}``.

        Returns:
            Scalar loss tensor.
        """

        probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
        targets = targets.float()
        weights = torch.where(targets > 0.5, self.pos_weight, torch.ones_like(targets))
        bce = F.binary_cross_entropy(probabilities, targets, weight=weights)

        probs_flat = probabilities.view(probabilities.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss

