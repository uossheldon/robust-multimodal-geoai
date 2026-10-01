from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TerraMindUpsamplingDecoder(nn.Module):
    """Lightweight 14x14-to-224x224 segmentation decoder."""

    def __init__(self, in_channels: int = 192, num_classes: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        if logits.shape[-2:] != (224, 224):
            logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)
        return logits


class TerraMindFrozenSegmenter(nn.Module):
    """Frozen TerraMind v1 tiny multimodal backbone with a trainable decoder."""

    def __init__(self, backbone: nn.Module, *, feature_dim: int = 192, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.decoder = TerraMindUpsamplingDecoder(in_channels=feature_dim, num_classes=num_classes)

    @staticmethod
    def _last_feature(backbone_output: object) -> torch.Tensor:
        if isinstance(backbone_output, (list, tuple)):
            return backbone_output[-1]
        if isinstance(backbone_output, dict):
            for key in ("features", "out", "last_hidden_state"):
                value = backbone_output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            tensor_values = [value for value in backbone_output.values() if isinstance(value, torch.Tensor)]
            if tensor_values:
                return tensor_values[-1]
        if isinstance(backbone_output, torch.Tensor):
            return backbone_output
        raise TypeError(f"Unsupported TerraMind output type: {type(backbone_output)!r}")

    @staticmethod
    def _tokens_to_map(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features
        if features.ndim != 3:
            raise ValueError(f"Expected TerraMind tokens [B,N,C] or feature map [B,C,H,W], got {tuple(features.shape)}")
        batch, tokens, channels = features.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise ValueError(f"Cannot reshape {tokens} tokens to a square feature map")
        return features.transpose(1, 2).reshape(batch, channels, side, side).contiguous()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self._last_feature(self.backbone(inputs))
        feature_map = self._tokens_to_map(features)
        return {"out": self.decoder(feature_map)}


def create_terramind_frozen_segmenter(num_classes: int = 4) -> TerraMindFrozenSegmenter:
    from terratorch.registry import BACKBONE_REGISTRY

    backbone = BACKBONE_REGISTRY.build(
        "terramind_v1_tiny",
        pretrained=True,
        modalities=["RGB", "S1RTC"],
        merge_method="mean",
    )
    return TerraMindFrozenSegmenter(backbone, feature_dim=192, num_classes=num_classes)
