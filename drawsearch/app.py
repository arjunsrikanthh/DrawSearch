"""Local drawing-canvas application for the trained compact CNN."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from .models import CompactCNNV1
from .preprocessing import prepare_canvas_image


ROOT = Path(__file__).resolve().parents[1]


def load_classifier(device_name: str = "auto"):
    """Load the saved CNN and its class labels for local inference."""
    checkpoint_path = ROOT / "outputs/models/compact_cnn.pt"
    manifest_path = ROOT / "data/processed/manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Prepare the data and train the models before starting the drawing app.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = manifest["class_order"]
    if checkpoint["class_order"] != class_names:
        raise ValueError("The CNN labels do not match the current dataset manifest.")
    model = CompactCNNV1(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names, device


def predict_canvas(canvas_value, model, class_names, device):
    """Return all class probabilities for Gradio's editor value."""
    if isinstance(canvas_value, dict):
        canvas_value = canvas_value.get("composite")
    pixels = prepare_canvas_image(canvas_value)
    if pixels is None:
        return {}, "Draw one object in the canvas first."

    tensor = torch.from_numpy(pixels.astype(np.float32) / 255.0).reshape(1, 1, 28, 28).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    best_id = int(probabilities.argmax())
    display_scores = {name: float(probabilities[index]) for index, name in enumerate(class_names)}
    return display_scores, f"Top guess: **{class_names[best_id]}** ({probabilities[best_id]:.1%})."


def build_app(model, class_names, device):
    """Build the local web interface. Gradio handles the browser-to-Python call."""
    with gr.Blocks(title="DrawSearch — doodle classifier") as demo:
        gr.Markdown(
            "# DrawSearch\n"
            "Draw one object, then select **Classify drawing**. The model knows "
            "only the ten categories listed below, so unsupported objects receive "
            "the closest available class."
        )
        canvas = gr.Sketchpad(
            label="Drawing canvas",
            image_mode="RGB",
            type="numpy",
            canvas_size=(400, 400),
            fixed_canvas=True,
            brush=gr.Brush(default_size=8, colors=["#000000"], default_color="#000000", color_mode="fixed"),
            layers=False,
        )
        button = gr.Button("Classify drawing", variant="primary")
        prediction = gr.Label(label="Top three guesses", num_top_classes=3)
        status = gr.Markdown("The classifier runs locally on this computer.")
        gr.Markdown("**Supported categories:** " + ", ".join(class_names))
        button.click(
            fn=lambda drawing: predict_canvas(drawing, model, class_names, device),
            inputs=canvas,
            outputs=[prediction, status],
        )
    return demo
