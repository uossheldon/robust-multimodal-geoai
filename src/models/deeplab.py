from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large


def create_s2_deeplab(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model



def replace_deeplab_input_stem(model: nn.Module, input_channels: int) -> None:
    stem = model.backbone["0"][0]
    if not isinstance(stem, nn.Conv2d):
        raise TypeError("Unexpected torchvision DeepLab MobileNetV3 stem structure.")
    if stem.in_channels == input_channels:
        return
    replacement = nn.Conv2d(
        input_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        groups=stem.groups,
        bias=stem.bias is not None,
        padding_mode=stem.padding_mode,
    )
    import torch
    with torch.no_grad():
        if input_channels == 2:
            replacement.weight.copy_(stem.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) * 1.5)
        elif input_channels > 3:
            replacement.weight[:, :3].copy_(stem.weight)
            extension = stem.weight.mean(dim=1, keepdim=True)
            replacement.weight[:, 3:].copy_(extension.expand(-1, input_channels - 3, -1, -1))
        else:
            replacement.weight[:, :input_channels].copy_(stem.weight[:, :input_channels])
        if stem.bias is not None:
            replacement.bias.copy_(stem.bias)
    model.backbone["0"][0] = replacement


def create_sar_deeplab(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights, weights_backbone=None)
    replace_deeplab_input_stem(model, 2)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model



def create_fusion_deeplab(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights, weights_backbone=None)
    replace_deeplab_input_stem(model, 5)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model
