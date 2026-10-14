#!/usr/bin/env python3
"""Train the isolated MobileNet transfer-learning experiment.

The existing compact CNN and shared metrics file are deliberately untouched.
This trainer uses the project's fixed train/validation/test archives, chooses
the checkpoint using validation accuracy, and evaluates the test set only after
that choice. Run with --smoke-test first; full training requires --train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drawsearch.transfer_model import (  # noqa: E402
    FrozenMobileNetV3Small,
    pretrained_weights_are_cached,
    total_parameter_count,
    trainable_parameter_count,
)
from train_models import load_splits, select_device  # noqa: E402


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    pixels = torch.from_numpy(images.reshape(-1, 1, 28, 28)).float() / 255.0
    targets = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(pixels, targets), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(loss_function(logits, labels).item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += len(labels)
    return total_loss / examples, correct / examples


def run_smoke(manifest: dict, splits: dict[str, dict[str, np.ndarray]]) -> None:
    model = FrozenMobileNetV3Small(len(manifest["class_order"]), pretrained=False)
    images = torch.from_numpy(splits["train"]["images"][:4].reshape(-1, 1, 28, 28)).float() / 255.0
    labels = torch.from_numpy(splits["train"]["labels"][:4]).long()
    model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    assert logits.shape == (4, len(manifest["class_order"]))
    assert trainable_parameter_count(model) == 5760 + 10  # 576 -> 10 weights and 10 biases
    assert all(not p.requires_grad for p in model.backbone.features.parameters())
    assert all(not p.requires_grad for p in model.backbone.classifier[:-1].parameters())
    assert labels.shape == (4,)
    print("Transfer-learning trainer smoke test passed")
    print(f"  logits={tuple(logits.shape)}")
    print(f"  total_parameters={total_parameter_count(model):,}")
    print(f"  trainable_parameters={trainable_parameter_count(model):,} (new head only)")
    print(f"  cached_pretrained_weights={pretrained_weights_are_cached()}")
    print(f"  one_training_batch_loss={loss.item():.4f}")
    print("  full_training_started=False")


def train(args: argparse.Namespace, manifest: dict, splits: dict[str, dict[str, np.ndarray]]) -> None:
    if args.pretrained and not pretrained_weights_are_cached():
        raise SystemExit("--pretrained requested, but weights are not cached. See PROJECT_DETAILS.md.")
    seed = int(manifest["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(args.device)
    model = FrozenMobileNetV3Small(len(manifest["class_order"]), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loaders = {
        name: make_loader(splits[name]["images"], splits[name]["labels"], args.batch_size, name == "train")
        for name in ("train", "val", "test")
    }
    output_dir = ROOT / "outputs/transfer_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mobilenet_v3_small_head.pt"
    best_accuracy = -1.0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            examples += len(labels)
        val_loss, val_accuracy = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": running_loss / examples, "validation_loss": val_loss, "validation_accuracy": val_accuracy}
        history.append(record)
        print(f"epoch={epoch:02d} train_loss={record['train_loss']:.4f} val_accuracy={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "manifest_sha256": None}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = evaluate(model, loaders["test"], device)
    record = {
        "model": "FrozenMobileNetV3Small",
        "pretrained": args.pretrained,
        "device": str(device),
        "class_order": manifest["class_order"],
        "input": "grayscale 28x28 -> repeated RGB -> resized 224x224 -> ImageNet normalization",
        "checkpoint_selection": "highest validation accuracy",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_parameters": total_parameter_count(model),
        "trainable_parameters": trainable_parameter_count(model),
        "training_seconds": time.perf_counter() - start,
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "test_set_policy": "Test evaluated once after validation-based checkpoint selection.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved separate transfer-learning metrics to {output_dir / 'metrics.json'}")
    print(f"test_accuracy={test_accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny forward pass and stop.")
    parser.add_argument("--train", action="store_true", help="Run full train/validation/test experiment.")
    parser.add_argument("--pretrained", action="store_true", help="Use cached ImageNet MobileNet weights.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.smoke_test == args.train:
        parser.error("Choose exactly one of --smoke-test or --train.")
    manifest, splits, _ = load_splits(ROOT / "data/processed")
    if args.smoke_test:
        run_smoke(manifest, splits)
    else:
        train(args, manifest, splits)


if __name__ == "__main__":
    main()
