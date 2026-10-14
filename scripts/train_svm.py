#!/usr/bin/env python3
"""Train a small SVM comparison on DrawSearch's fixed train/validation/test splits.

The script keeps model selection honest: it tries the requested C values on the
training split, chooses the best validation result, and evaluates the test
split only once after that choice.  It writes a separate report so the
existing Logistic Regression/CNN results remain unchanged.
"""

import argparse
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
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from scripts.train_models import load_splits, score_predictions


def parse_c_values(value: str) -> list[float]:
    """Parse a comma-separated list such as ``0.01,0.1,1``."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("C values must be numbers separated by commas") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("C values must be positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--c-values",
        type=parse_c_values,
        default=[0.01, 0.1],
        help="Validation candidates; keep the list short because each full-data fit is CPU-intensive.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/metrics/svm_results.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/models/linear_svm.joblib")
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")

    data_dir = ROOT / "data/processed"
    manifest, splits, manifest_sha256 = load_splits(data_dir)
    class_names = manifest["class_order"]
    seed = int(manifest["seed"])
    x_train = splits["train"]["images"].astype(np.float32) / 255.0
    x_val = splits["val"]["images"].astype(np.float32) / 255.0
    x_test = splits["test"]["images"].astype(np.float32) / 255.0
    y_train = splits["train"]["labels"]
    y_val = splits["val"]["labels"]
    y_test = splits["test"]["labels"]

    print(f"Using {len(x_train):,} training, {len(x_val):,} validation, and {len(x_test):,} test drawings.", flush=True)
    print(f"Classes: {', '.join(class_names)}", flush=True)
    print(f"Trying LinearSVC C values: {args.c_values}", flush=True)
    candidates = []
    for c_value in args.c_values:
        model = LinearSVC(
            C=c_value,
            max_iter=args.max_iter,
            # With 80,000 examples and 784 pixels, the primal solver is the
            # faster and more memory-friendly choice on this Mac.
            dual=False,
            random_state=seed,
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        validation_predictions = model.predict(x_val)
        validation_scores, _ = score_predictions(y_val, validation_predictions, class_names)
        convergence_warnings = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        candidate = {
            "C": c_value,
            "training_seconds": elapsed,
            "convergence_warnings": convergence_warnings,
            "validation": {
                key: value for key, value in validation_scores.items() if key != "confusion_matrix"
            },
        }
        candidates.append(candidate)
        print(
            f"  C={c_value:g}: validation accuracy={validation_scores['accuracy']:.4f}; "
            f"macro-F1={validation_scores['macro_f1']:.4f}; training={elapsed:.1f}s",
            flush=True,
        )

    # Ties use macro-F1, then the smaller C.  Both choices use validation only.
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["validation"]["accuracy"],
            candidates[index]["validation"]["macro_f1"],
            -candidates[index]["C"],
        ),
    )
    best_c = candidates[best_index]["C"]
    best_model = LinearSVC(C=best_c, max_iter=args.max_iter, dual=False, random_state=seed)
    refit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        best_model.fit(x_train, y_train)
    refit_seconds = time.perf_counter() - refit_start
    refit_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]

    # This is intentionally the first and only test evaluation in this script.
    test_predictions = best_model.predict(x_test)
    test_scores, _ = score_predictions(y_test, test_predictions, class_names)
    test_scores.pop("confusion_matrix", None)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.checkpoint)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_manifest_sha256": manifest_sha256,
        "model": "LinearSVC",
        "input": "flattened 784 pixels, normalized to [0, 1]",
        "class_order": class_names,
        "selection_policy": "C was chosen by validation accuracy, with validation macro-F1 and then smaller C as tie-breakers; the test split was evaluated only after selection.",
        "settings": {
            "candidate_C_values": args.c_values,
            "max_iter": args.max_iter,
            "dual": False,
            "random_state": seed,
        },
        "candidates": candidates,
        "selected_C": best_c,
        "selected_model": {
            "refit_training_seconds": refit_seconds,
            "convergence_warnings": refit_warnings,
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "parameter_count": int(best_model.coef_.size + best_model.intercept_.size),
            "validation": candidates[best_index]["validation"],
            "test": test_scores,
        },
        "limitations": [
            "This is a linear SVM comparison; no RBF-kernel SVM was run because full-data RBF training is substantially more expensive on 80,000 drawings.",
            "The test metrics are for this fixed project split and should be compared with paper metrics only after matching split size, categories, and metric type.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected C={best_c:g}; test accuracy={test_scores['accuracy']:.4f}; "
        f"test macro-F1={test_scores['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Saved metrics to {args.output}", flush=True)
    print(f"Saved checkpoint to {args.checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
