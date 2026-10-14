#!/usr/bin/env python3
"""Run a cheap architecture/data smoke test for the isolated MobileNet experiment.

This deliberately performs one forward pass only.  It does not train, download
weights, write metrics, or modify the existing demo models.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drawsearch.transfer_model import (  # noqa: E402
    FrozenMobileNetV3Small,
    pretrained_weights_are_cached,
    total_parameter_count,
    trainable_parameter_count,
)


def main() -> None:
    data_dir = ROOT / "data/processed"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        images = torch.from_numpy(archive["images"][:4].reshape(-1, 1, 28, 28)).float() / 255.0
        labels = torch.from_numpy(archive["labels"][:4]).long()

    torch.manual_seed(int(manifest["seed"]))
    model = FrozenMobileNetV3Small(len(manifest["class_order"]), pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (4, len(manifest["class_order"]))
    assert labels.shape == (4,)
    frozen = sum(1 for parameter in model.backbone.features.parameters() if not parameter.requires_grad)
    assert frozen > 0
    print("Transfer-learning smoke test passed")
    print(f"  input={tuple(images.shape)} -> logits={tuple(logits.shape)}")
    print(f"  classes={manifest['class_order']}")
    print(f"  total_parameters={total_parameter_count(model):,}")
    print(f"  trainable_parameters={trainable_parameter_count(model):,} (classifier head only)")
    print(f"  cached_pretrained_weights={pretrained_weights_are_cached()}")
    print("  training_started=False")


if __name__ == "__main__":
    main()
