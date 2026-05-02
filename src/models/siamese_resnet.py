"""ResNet-18 Siamese U-Net variants with optional CBAM skip attention."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .cbam import CBAMBlock
from .siamese_unet import ChangeHead


def decoder_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create a two-layer decoder block."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsample(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create an upsample-reduce block."""

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SharedResNetEncoder(nn.Module):
    """Shared ResNet-18 backbone exposing the requested feature stages."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode an image into bottleneck and skip tensors."""

        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l4, [l0, l1, l2, l3, l4]


class ResNetDecoder(nn.Module):
    """ResNet-compatible U-Net decoder with optional CBAM on skip tensors."""

    def __init__(self, embed_dim: int = 64, use_cbam: bool = False) -> None:
        super().__init__()
        self.use_cbam = use_cbam
        self.skip_attn = nn.ModuleList(
            [
                CBAMBlock(512),
                CBAMBlock(256),
                CBAMBlock(128),
                CBAMBlock(64),
                CBAMBlock(64),
            ]
        ) if use_cbam else None

        self.up5 = upsample(512, 256)
        self.d5 = decoder_conv(768, 256)
        self.up4 = upsample(256, 128)
        self.d4 = decoder_conv(384, 128)
        self.up3 = upsample(128, 64)
        self.d3 = decoder_conv(192, 64)
        self.up2 = upsample(64, 64)
        self.d2 = decoder_conv(128, 64)
        self.up1 = upsample(64, 32)
        self.d1 = decoder_conv(96, 32)
        self.head = nn.Conv2d(32, embed_dim, kernel_size=1)

    def _attend(self, skip: torch.Tensor, index: int) -> torch.Tensor:
        if self.skip_attn is None:
            return skip
        return self.skip_attn[index](skip)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Decode ResNet features into dense embeddings."""

        l0, l1, l2, l3, l4 = skips
        x = self.d5(torch.cat([self.up5(bottleneck), self._attend(l4, 0)], dim=1))
        x = self.d4(torch.cat([self.up4(x), self._attend(l3, 1)], dim=1))
        x = self.d3(torch.cat([self.up3(x), self._attend(l2, 2)], dim=1))
        x = self.d2(torch.cat([self.up2(x), self._attend(l1, 3)], dim=1))
        x = self.d1(torch.cat([self.up1(x), self._attend(l0, 4)], dim=1))
        return self.head(x)


class SiameseResNetUNet(nn.Module):
    """Final Siamese model with shared ResNet-18 encoder and change head.

    Args:
        embed_dim: Embedding dimension for the dense decoder output.
        pretrained: Whether to load ImageNet pretrained weights.
        use_cbam: Whether to apply CBAM to decoder skip connections.
    """

    def __init__(self, embed_dim: int = 64, pretrained: bool = True, use_cbam: bool = False) -> None:
        super().__init__()
        self.encoder = SharedResNetEncoder(pretrained=pretrained)
        self.decoder = ResNetDecoder(embed_dim=embed_dim, use_cbam=use_cbam)
        self.change_head = ChangeHead()

    def backbone_parameters(self):
        """Yield backbone parameters for fine-tuning param groups."""

        return self.encoder.parameters()

    def decoder_head_parameters(self):
        """Yield decoder and segmentation-head parameters."""

        for module in [self.decoder, self.change_head]:
            yield from module.parameters()

    def freeze_backbone(self) -> None:
        """Freeze the shared ResNet backbone."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the shared ResNet backbone."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward_embeddings(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate paired dense embeddings."""

        emb_a = self.decoder(*self.encoder(image_a))
        emb_b = self.decoder(*self.encoder(image_b))
        return emb_a, emb_b

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward inference and expose embeddings, distance, and probabilities."""

        emb_a, emb_b = self.forward_embeddings(image_a, image_b)
        distance = torch.norm(emb_a - emb_b, p=2, dim=1, keepdim=True)
        probability = self.change_head(distance)
        return {
            "embedding_a": emb_a,
            "embedding_b": emb_b,
            "distance": distance,
            "probability": probability,
        }

