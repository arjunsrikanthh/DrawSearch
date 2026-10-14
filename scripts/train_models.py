#!/usr/bin/env python3
"""Train the Logistic Regression baseline and compact CNN on fixed data splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from drawsearch.models import CompactCNNV1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits(data_dir: Path):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_data = {}
    for split_name in ("train", "val", "test"):
        archive_path = data_dir / f"{split_name}.npz"
        metadata = manifest["split_archives"][split_name]
        if sha256_file(archive_path) != metadata["sha256"]:
            raise ValueError(f"{archive_path} does not match the hash in manifest.json")
        with np.load(archive_path, allow_pickle=False) as archive:
            split_data[split_name] = {
                "images": archive["images"],
                "labels": archive["labels"],
                "source_indices": archive["source_indices"],
            }
    class_ids = list(range(len(manifest["class_order"])))
    for split_name, data in split_data.items():
        if data["images"].shape[1:] != (784,) or data["images"].dtype != np.uint8:
            raise ValueError(f"Unexpected images in {split_name}.npz")
        if len(data["labels"]) != manifest["split_archives"][split_name]["drawings"]:
            raise ValueError(f"Unexpected number of drawings in {split_name}.npz")
        if set(np.unique(data["labels"])) != set(class_ids):
            raise ValueError(f"Unexpected or missing class ID in {split_name}.npz")
    return manifest, split_data, sha256_file(manifest_path)


def score_predictions(y_true, y_pred, class_names):
    label_ids = list(range(len(class_names)))
    precision, recall, f1, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(supports[index]),
        }
        for index, name in enumerate(class_names)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }, matrix


def save_confusion_plot(matrix, class_names, model_name, destination: Path):
    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=class_names
    )
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    axis.set_title(f"{model_name} — test set")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(requested)


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def measure_logistic_latency(model, sample, repeats=100):
    for _ in range(10):
        model.predict(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    return (time.perf_counter() - start) * 1000 / repeats


def evaluate_cnn(model, loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    prediction_parts = []
    actual_parts = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += float(loss_function(logits, labels).item()) * len(labels)
            total_examples += len(labels)
            prediction_parts.append(logits.argmax(dim=1).cpu().numpy())
            actual_parts.append(labels.cpu().numpy())
    return total_loss / total_examples, np.concatenate(actual_parts), np.concatenate(prediction_parts)


def measure_cnn_latency(model, device, sample, repeats=100):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeats


def train(args):
    data_dir = ROOT / "data/processed"
    manifest, splits, manifest_sha256 = load_splits(data_dir)
    class_names = manifest["class_order"]
    seed = int(manifest["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Using saved class order: {', '.join(class_names)}", flush=True)

    y_train = splits["train"]["labels"]
    y_val = splits["val"]["labels"]
    y_test = splits["test"]["labels"]
    metric_records = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_manifest_sha256": manifest_sha256,
        "test_set_policy": "Training and model checkpoint selection use train/validation only; test metrics are calculated after training.",
        "device": str(device),
        "models": {},
    }

    # The baseline sees the same 28x28 drawing as 784 normalized pixel values.
    print("\nTraining Logistic Regression baseline...", flush=True)
    x_train_flat = splits["train"]["images"].astype(np.float32) / 255.0
    x_val_flat = splits["val"]["images"].astype(np.float32) / 255.0
    logistic = LogisticRegression(
        solver="lbfgs",
        max_iter=args.max_iter,
        tol=1e-3,
        random_state=seed,
    )
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        logistic.fit(x_train_flat, y_train)
        logistic_training_seconds = time.perf_counter() - start
        convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    val_logistic = logistic.predict(x_val_flat)
    logistic_val_scores, _ = score_predictions(y_val, val_logistic, class_names)
    logistic_path = ROOT / "outputs/models/logistic_regression.joblib"
    logistic_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, logistic_path)
    print(
        f"  validation accuracy={logistic_val_scores['accuracy']:.4f}; "
        f"training={logistic_training_seconds:.1f}s",
        flush=True,
    )
    metric_records["models"]["logistic_regression"] = {
        "input": "flattened 784 pixels, normalized to [0, 1]",
        "settings": {"solver": "lbfgs", "max_iter": args.max_iter, "tolerance": 1e-3},
        "training_seconds": logistic_training_seconds,
        "convergence_warnings": convergence_warnings,
        "checkpoint": str(logistic_path.relative_to(ROOT)),
        "checkpoint_bytes": logistic_path.stat().st_size,
        "validation": {key: value for key, value in logistic_val_scores.items() if key != "confusion_matrix"},
        "inference_latency_ms_per_drawing": measure_logistic_latency(logistic, x_val_flat[:1]),
    }
    del x_train_flat, x_val_flat

    # The CNN keeps the image's 2-D layout so it can learn local stroke patterns.
    print("\nTraining compact two-convolution CNN...", flush=True)
    x_train_cnn = torch.from_numpy(
        splits["train"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    x_val_cnn = torch.from_numpy(
        splits["val"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    x_test_cnn = torch.from_numpy(
        splits["test"]["images"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    )
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64, copy=False))
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64, copy=False))
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64, copy=False))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train_cnn, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(TensorDataset(x_val_cnn, y_val_tensor), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_cnn, y_test_tensor), batch_size=1024, shuffle=False)

    cnn = CompactCNNV1(len(class_names)).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    validation_history = []
    synchronize(device)
    cnn_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        cnn.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = cnn(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_actual, val_predictions = evaluate_cnn(cnn, val_loader, device, loss_function)
        val_scores, _ = score_predictions(val_actual, val_predictions, class_names)
        val_accuracy = val_scores["accuracy"]
        validation_history.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        print(f"  epoch {epoch:02d}/{args.epochs}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}", flush=True)

        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in cnn.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"  stopping early after {args.patience} epochs without validation improvement.", flush=True)
                break

    synchronize(device)
    cnn_training_seconds = time.perf_counter() - cnn_start
    if best_state is None:
        raise RuntimeError("CNN training did not produce a validation checkpoint.")
    cnn.load_state_dict(best_state)
    _, y_val_rechecked, val_cnn_predictions = evaluate_cnn(cnn, val_loader, device, loss_function)
    val_cnn_scores, _ = score_predictions(y_val_rechecked, val_cnn_predictions, class_names)
    cnn_path = ROOT / "outputs/models/compact_cnn.pt"
    cnn_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "CompactCNNV1",
            "state_dict": best_state,
            "class_order": class_names,
            "input_shape": [1, 28, 28],
            "pixel_normalization": "uint8 / 255.0; white strokes on black background",
            "best_epoch": best_epoch,
            "validation_accuracy": best_accuracy,
        },
        cnn_path,
    )
    metric_records["models"]["compact_cnn"] = {
        "architecture": "Two 14-filter 3x3 convolutions, max-pooling after each, 128-unit dense layer, ten logits.",
        "settings": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.patience,
        },
        "best_epoch": best_epoch,
        "training_seconds": cnn_training_seconds,
        "validation_history": validation_history,
        "checkpoint": str(cnn_path.relative_to(ROOT)),
        "checkpoint_bytes": cnn_path.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in cnn.parameters()),
        "validation": {key: value for key, value in val_cnn_scores.items() if key != "confusion_matrix"},
        "inference_latency_ms_per_drawing": None,
    }

    # The held-out test split is read only after the two models are trained.
    x_test_flat = splits["test"]["images"].astype(np.float32) / 255.0
    logistic_test = logistic.predict(x_test_flat)
    logistic_test_scores, logistic_matrix = score_predictions(y_test, logistic_test, class_names)
    metric_records["models"]["logistic_regression"]["test"] = {
        key: value for key, value in logistic_test_scores.items() if key != "confusion_matrix"
    }
    metric_records["models"]["logistic_regression"]["test"]["confusion_matrix"] = logistic_test_scores["confusion_matrix"]
    save_confusion_plot(logistic_matrix, class_names, "Logistic Regression", ROOT / "outputs/figures/confusion_logistic_regression.png")

    cnn_test_loss, cnn_test_actual, cnn_test_predictions = evaluate_cnn(cnn, test_loader, device, loss_function)
    cnn_test_scores, cnn_matrix = score_predictions(cnn_test_actual, cnn_test_predictions, class_names)
    metric_records["models"]["compact_cnn"]["test"] = {
        key: value for key, value in cnn_test_scores.items() if key != "confusion_matrix"
    }
    metric_records["models"]["compact_cnn"]["test"]["loss"] = cnn_test_loss
    metric_records["models"]["compact_cnn"]["test"]["confusion_matrix"] = cnn_test_scores["confusion_matrix"]
    metric_records["models"]["compact_cnn"]["inference_latency_ms_per_drawing"] = measure_cnn_latency(
        cnn, device, x_val_cnn[:1]
    )
    save_confusion_plot(cnn_matrix, class_names, "Compact CNN", ROOT / "outputs/figures/confusion_compact_cnn.png")

    results_path = ROOT / "outputs/metrics/results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metric_records, indent=2) + "\n", encoding="utf-8")
    for model_name, metrics in metric_records["models"].items():
        print(
            f"\n{model_name}: validation accuracy {metrics['validation']['accuracy']:.4f}; "
            f"test accuracy {metrics['test']['accuracy']:.4f}; "
            f"test macro-F1 {metrics['test']['macro_f1']:.4f}",
            flush=True,
        )
    print(f"\nSaved metrics to {results_path}")
    print(f"Saved models under {ROOT / 'outputs/models'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum CNN training epochs (default: 10).")
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many unimproved validation epochs.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=200, help="Logistic Regression maximum iterations.")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.max_iter) < 1:
        parser.error("epochs, patience, batch-size, and max-iter must all be positive")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
