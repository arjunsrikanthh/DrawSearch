#!/usr/bin/env python3
"""Check the toolchain using synthetic data only; no dataset or weights download."""

import argparse
from datetime import datetime
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "numpy": "numpy", "Pillow": "PIL", "pandas": "pandas",
    "matplotlib": "matplotlib", "scikit-learn": "sklearn", "joblib": "joblib",
    "PyYAML": "yaml", "torch": "torch", "torchvision": "torchvision",
    "gradio": "gradio",
}


def synthetic_cnn_check(torch, device):
    """Exercise common CNN ops, gradients, optimizer, and CPU checkpoint reload."""
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), torch.nn.Flatten(),
            torch.nn.Linear(8 * 14 * 14, 10),
        )

    torch.manual_seed(42)
    model = make_model().to(device)
    x_cpu = torch.rand(8, 1, 28, 28)
    x = x_cpu.to(device)
    y = torch.arange(8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite synthetic loss on {device}")
        loss.backward()
        optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        expected = model(x).cpu()
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "synthetic.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        restored = make_model()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        restored.eval()
        with torch.no_grad():
            actual = restored(x_cpu)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    return {"status": "passed", "synthetic_steps": 2,
            "elapsed_seconds": round(elapsed, 4), "cpu_checkpoint_reload": "passed",
            "note": "Random tensors; not a project model, accuracy, or runtime benchmark."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--check-ui", action="store_true", help="Start a temporary localhost ImageEditor, check HTTP, then close it.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/setup/environment.json")
    args = parser.parse_args()
    free = shutil.disk_usage(ROOT).free
    report = {
        "checked_at_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "free_disk_gib": round(free / 1024 ** 3, 1),
        "packages": {}, "checks": {},
        "evidence_scope": "Environment setup only. No dataset, trained model, or prediction accuracy.",
    }
    if sys.platform == "darwin":
        for key, query in [("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            result = subprocess.run(["sysctl", "-n", query], capture_output=True, text=True)
            if result.returncode == 0:
                report[key] = result.stdout.strip()
    failures = []

    def check(name, action):
        try:
            result = action()
            report["checks"][name] = result or {"status": "passed"}
            print(f"PASS {name}", flush=True)
        except Exception as error:
            report["checks"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            failures.append(name)
            print(f"FAIL {name}: {error}", flush=True)

    def imports():
        for package, module in PACKAGES.items():
            import_module(module)
            report["packages"][package] = metadata.version(package)
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ is required; Python 3.12 is the tested setup.")
    check("package_imports", imports)

    def configuration():
        import yaml
        config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        if len(config["categories"]) != 10 or len(set(config["categories"])) != 10:
            raise ValueError("data.yaml must list ten distinct classes in label order")
        if sum(config["split_counts_per_class"].values()) != config["samples_per_class"]:
            raise ValueError("Split counts disagree with samples_per_class")
        return {"status": "passed", "class_order": config["categories"],
                "note": "data.yaml is the single source of truth for label order."}
    check("configuration", configuration)

    def classical():
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 8))
        y = np.arange(30) % 3
        model = LogisticRegression(max_iter=100).fit(x, y)
        if model.predict_proba(x[:2]).shape != (2, 3):
            raise RuntimeError("Unexpected sklearn output shape")
        return {"status": "passed", "note": "Synthetic compatibility check only."}
    check("sklearn_fit_predict", classical)

    if "package_imports" not in failures:
        import torch
        report["mps_built"] = torch.backends.mps.is_built()
        report["mps_available"] = torch.backends.mps.is_available()
        device = "mps" if args.device == "auto" and report["mps_available"] else args.device
        if device == "auto":
            device = "cpu"
        report["selected_device"] = device
        check("cnn_cpu", lambda: synthetic_cnn_check(torch, "cpu"))
        if device == "mps":
            if report["mps_available"]:
                check("cnn_mps", lambda: synthetic_cnn_check(torch, "mps"))
            else:
                check("cnn_mps", lambda: (_ for _ in ()).throw(RuntimeError("MPS explicitly requested but unavailable")))

        def transfer():
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2(weights=None).eval()
            model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
            for parameter in model.features.parameters():
                parameter.requires_grad = False
            model.to(device)
            with torch.no_grad():
                prediction = model(torch.zeros(1, 3, 224, 224, device=device))
            if prediction.shape != (1, 10) or not torch.isfinite(prediction).all().item():
                raise RuntimeError("Unexpected MobileNetV2 output")
            return {"status": "passed", "device": device, "weights": "random",
                    "note": "Architecture compatibility only; pretrained weights not downloaded."}
        check("mobilenet_architecture", transfer)

        def canvas():
            import gradio as gr
            with gr.Blocks(analytics_enabled=False) as demo:
                gr.ImageEditor(label="Setup check canvas", type="pil", image_mode="RGBA", sources=())
            if not any(component["type"] == "imageeditor" for component in demo.config["components"]):
                raise RuntimeError("ImageEditor absent from Blocks configuration")
            if args.check_ui:
                try:
                    _, local_url, _ = demo.launch(server_name="127.0.0.1", share=False,
                        prevent_thread_lock=True, quiet=True, inbrowser=False)
                    with urlopen(local_url, timeout=10) as response:
                        if response.status != 200:
                            raise RuntimeError(f"UI returned HTTP {response.status}")
                finally:
                    demo.close()
            return {"status": "passed", "component": "ImageEditor",
                    "localhost_http_check": "passed" if args.check_ui else "not_requested",
                    "note": "Component/server check only; browser drawing and prediction remain untested."}
        check("gradio_canvas", canvas)

    report["status"] = "failed" if failures else "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    print(f"Overall: {report['status']}; device: {report.get('selected_device', 'unavailable')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
