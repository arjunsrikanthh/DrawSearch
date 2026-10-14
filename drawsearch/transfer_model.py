"""A small, isolated transfer-learning model for the DrawSearch project.

The paper experimented with MobileNet.  This module keeps that experiment
separate from the compact CNN used by the demo: a MobileNetV3-Small image
backbone is frozen and only a new ten-class classifier head is trainable.
Quick Draw images are one-channel 28x28 bitmaps, so the forward pass repeats
the channel three times and resizes the image to MobileNet's expected size.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pretrained_weights_path() -> Path:
    """Return the usual local cache path for the default MobileNet weights."""
    filename = Path(urlparse(MobileNet_V3_Small_Weights.DEFAULT.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def pretrained_weights_are_cached() -> bool:
    """Check for cached weights without downloading anything."""
    return pretrained_weights_path().is_file()


def _make_backbone(pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    if pretrained and not pretrained_weights_are_cached():
        path = pretrained_weights_path()
        raise FileNotFoundError(
            "MobileNet pretrained weights are not cached. "
            f"Expected: {path}. See the 'Transfer-learning weights' section "
            "in PROJECT_DETAILS.md, or use pretrained=False."
        )

    backbone = mobilenet_v3_small(weights=None)
    if pretrained:
        # Loading the cached file directly guarantees this experiment cannot
        # silently start a network download.
        state = torch.load(pretrained_weights_path(), map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    return backbone


class FrozenMobileNetV3Small(nn.Module):
    """MobileNetV3-Small with a frozen feature extractor and new class head."""

    def __init__(self, number_of_classes: int, *, pretrained: bool = False):
        super().__init__()
        self.backbone = _make_backbone(pretrained)
        # Freeze every parameter from ImageNet, including MobileNet's original
        # classifier. The original classifier is discarded below; only the new
        # lightweight head is trainable.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # The pooled MobileNet feature vector enters the original classifier
        # at classifier[0].in_features (576 for MobileNetV3-Small). Replace
        # both original classifier layers, rather than only the final layer.
        input_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_features, number_of_classes),
        )
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected grayscale images with shape [batch, 1, height, width].")
        images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        return self.backbone(images)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count only parameters that an optimizer would update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    """Count all parameters, including the frozen backbone."""
    return sum(parameter.numel() for parameter in model.parameters())
