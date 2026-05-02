"""CBAM attention blocks for decoder skip refinement."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention branch used inside CBAM.

    Args:
        channels: Input channel count.
        reduction: Bottleneck reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Feature map shaped ``(B, C, H, W)``.

        Returns:
            Attention-refined feature map.
        """

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weights = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """Spatial attention branch used inside CBAM."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Feature map shaped ``(B, C, H, W)``.

        Returns:
            Attention-refined feature map.
        """

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weights


class CBAMBlock(nn.Module):
    """Full CBAM block: channel attention followed by spatial attention.

    Args:
        channels: Input channel count.
        reduction: Channel reduction ratio for the MLP.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full CBAM sequence.

        Args:
            x: Feature map shaped ``(B, C, H, W)``.

        Returns:
            Refined feature map.
        """

        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

