#!/usr/bin/env python3
"""Train the one-convolution CNN V2 on the project's saved data splits."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from drawsearch.models import CompactCNNV2
from train_models import (
    evaluate_cnn,
    load_splits,
    measure_cnn_latency,
    save_confusion_plot,
    score_predictions,
    select_device,
    synchronize,
)


def train(args):
    data_dir = ROOT / "data/processed"
    manifest, splits, manifest_sha256 = load_splits(data_dir)
    results_path = ROOT / "outputs/metrics/results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("dataset_manifest_sha256") != manifest_sha256:
        raise ValueError("Existing model results use a different dataset manifest; compare runs separately.")
    if "compact_cnn" not in results.get("models", {}):
        raise ValueError("Train the baseline and CNN V1 first so the comparison uses the same data record.")

    class_names = manifest["class_order"]
    seed = int(manifest["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)
    print("Training the paper's one-convolution CNN V2 on the saved split.", flush=True)

    x_train = torch.from_numpy(
        splits["train"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    x_val = torch.from_numpy(
        splits["val"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    x_test = torch.from_numpy(
        splits["test"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    y_train = torch.from_numpy(splits["train"]["labels"].astype(np.int64, copy=False))
    y_val = torch.from_numpy(splits["val"]["labels"].astype(np.int64, copy=False))
    y_test = torch.from_numpy(splits["test"]["labels"].astype(np.int64, copy=False))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1024, shuffle=False)

    model = CompactCNNV2(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    validation_history = []
    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, actual, predictions = evaluate_cnn(model, val_loader, device, loss_function)
        val_scores, _ = score_predictions(actual, predictions, class_names)
        val_accuracy = val_scores["accuracy"]
        validation_history.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        print(f"epoch {epoch:02d}/{args.epochs}: val_accuracy={val_accuracy:.4f}", flush=True)

        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"Early stop after {args.patience} epochs without validation improvement.", flush=True)
                break

    synchronize(device)
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("CNN V2 training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    _, val_actual, val_predictions = evaluate_cnn(model, val_loader, device, loss_function)
    val_scores, _ = score_predictions(val_actual, val_predictions, class_names)

    checkpoint_path = ROOT / "outputs/models/compact_cnn_v2.pt"
    torch.save(
        {
            "architecture": "CompactCNNV2",
            "state_dict": best_state,
            "class_order": class_names,
            "input_shape": [1, 28, 28],
            "pixel_normalization": "uint8 / 255.0; white strokes on black background",
            "best_epoch": best_epoch,
            "validation_accuracy": best_accuracy,
        },
        checkpoint_path,
    )

    # Evaluate the held-out test split only after selecting this model by validation.
    test_loss, test_actual, test_predictions = evaluate_cnn(model, test_loader, device, loss_function)
    test_scores, test_matrix = score_predictions(test_actual, test_predictions, class_names)
    latency_ms = measure_cnn_latency(model, device, x_val[:1])
    save_confusion_plot(
        test_matrix, class_names, "Compact CNN V2",
        ROOT / "outputs/figures/confusion_compact_cnn_v2.png",
    )

    results["models"]["compact_cnn_v2"] = {
        "architecture": "One 14-filter 3x3 convolution and max-pooling layer, then a 128-unit dense layer and ten logits.",
        "settings": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
        },
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "validation_history": validation_history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "validation": {key: value for key, value in val_scores.items() if key != "confusion_matrix"},
        "test": {key: value for key, value in test_scores.items() if key != "confusion_matrix"},
        "test_loss": test_loss,
        "inference_latency_ms_per_drawing": latency_ms,
    }
    results["test_set_policy"] = (
        "CNN V2 checkpoint selected using training and validation only; test metrics calculated after selection. "
        "The same held-out test split was used for the previously reported baseline and CNN V1."
    )
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"CNN V2: validation accuracy {val_scores['accuracy']:.4f}; "
        f"test accuracy {test_scores['accuracy']:.4f}; test macro-F1 {test_scores['macro_f1']:.4f}; "
        f"training {training_seconds:.2f}s; inference {latency_ms:.3f}ms/drawing"
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Updated metrics: {results_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience, and batch-size must all be positive")
    train(args)


if __name__ == "__main__":
    main()
