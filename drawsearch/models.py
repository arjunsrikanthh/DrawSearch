"""Small neural networks used by the DrawSearch project."""

import torch.nn as nn


class CompactCNNV1(nn.Module):
    """The authors' two-convolution doodle model, expressed in modern PyTorch."""

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 14, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(14, 14, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class CompactCNNV2(nn.Module):
    """One-convolution simplification of V1, matching the paper's stated V2."""

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 14, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))
