"""Convert a browser canvas drawing into the pixels expected by our CNN."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def prepare_canvas_image(
    canvas_image,
    *,
    output_size: int = 28,
    margin: int = 3,
    background_threshold: int = 245,
) -> np.ndarray | None:
    """Center a black-on-white canvas sketch and return white-on-black pixels.

    Returns ``None`` for an empty canvas. Drawings are cropped to their ink,
    scaled without changing aspect ratio, and padded to match Quick, Draw!'s
    small centered bitmap format.
    """
    if canvas_image is None:
        return None

    array = np.asarray(canvas_image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a grayscale or color image; received shape {array.shape}.")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 color channels; received shape {array.shape}.")

    image = Image.fromarray(array.astype(np.uint8, copy=False))
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    gray = ImageOps.grayscale(image)

    # Quick, Draw! stores white strokes on black. Gradio's simple sketch canvas
    # uses black strokes on white, so invert it after identifying the drawing.
    ink_mask = np.asarray(gray) < background_threshold
    if not ink_mask.any():
        return None

    ys, xs = np.nonzero(ink_mask)
    bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    white_ink = ImageOps.invert(gray).crop(bounds)
    available = output_size - 2 * margin
    if available < 1:
        raise ValueError("The margin must leave at least one pixel for the drawing.")

    scale = min(available / white_ink.width, available / white_ink.height)
    resized_size = (
        max(1, round(white_ink.width * scale)),
        max(1, round(white_ink.height * scale)),
    )
    white_ink = white_ink.resize(resized_size, Image.Resampling.LANCZOS)
    normalized = Image.new("L", (output_size, output_size), 0)
    offset = ((output_size - resized_size[0]) // 2, (output_size - resized_size[1]) // 2)
    normalized.paste(white_ink, offset)
    return np.asarray(normalized, dtype=np.uint8)
