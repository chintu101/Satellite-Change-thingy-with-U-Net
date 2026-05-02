"""Custom Siamese U-Net baseline and lightweight change head."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a two-layer Conv-BN-ReLU block.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.

    Returns:
        Sequential block.
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Upsample by 2x then reduce channels.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.

    Returns:
        Sequential upsample block.
    """

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ChangeHead(nn.Module):
    """Convert a distance map into per-pixel change probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, distance_map: torch.Tensor) -> torch.Tensor:
        """Predict a change probability map.

        Args:
            distance_map: Tensor shaped ``(B, 1, H, W)``.

        Returns:
            Probability map shaped ``(B, 1, H, W)``.
        """

        return self.layers(distance_map)


class SiameseUNetEncoder(nn.Module):
    """Four-stage baseline encoder with shared weights across both branches."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = conv_block(3, 32)
        self.stage2 = conv_block(32, 64)
        self.stage3 = conv_block(64, 128)
        self.stage4 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode an input image into bottleneck and skip tensors."""

        s1 = self.stage1(x)
        s2 = self.stage2(self.pool(s1))
        s3 = self.stage3(self.pool(s2))
        s4 = self.stage4(self.pool(s3))
        bn = self.bottleneck(self.pool(s4))
        return bn, [s1, s2, s3, s4]


class SiameseUNetDecoder(nn.Module):
    """Baseline decoder that reconstructs full-resolution embeddings."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.up4 = upsample_block(512, 256)
        self.dec4 = conv_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up2 = upsample_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up1 = upsample_block(64, 32)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, embed_dim, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Decode embeddings back to the input resolution."""

        s1, s2, s3, s4 = skips
        x = self.dec4(torch.cat([self.up4(bottleneck), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return self.head(x)


class SiameseUNet(nn.Module):
    """Baseline Siamese U-Net with optional change head output."""

    def __init__(self, embed_dim: int = 64, use_change_head: bool = False) -> None:
        super().__init__()
        self.encoder = SiameseUNetEncoder()
        self.decoder = SiameseUNetDecoder(embed_dim=embed_dim)
        self.change_head = ChangeHead() if use_change_head else None

    def forward_embeddings(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Siamese embeddings for a pair of images."""

        emb_a = self.decoder(*self.encoder(image_a))
        emb_b = self.decoder(*self.encoder(image_b))
        return emb_a, emb_b

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full baseline model.

        Returns:
            Tuple ``(embedding_a, embedding_b, distance_or_probability)``.
        """

        emb_a, emb_b = self.forward_embeddings(image_a, image_b)
        distance = torch.norm(emb_a - emb_b, p=2, dim=1, keepdim=True)
        if self.change_head is None:
            return emb_a, emb_b, distance
        probability = self.change_head(distance)
        return emb_a, emb_b, probability

