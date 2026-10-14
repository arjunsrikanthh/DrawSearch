"""Checks for the browser-canvas to Quick, Draw! pixel conversion."""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from drawsearch.preprocessing import prepare_canvas_image


class CanvasPreprocessingTests(unittest.TestCase):
    def test_blank_white_canvas_is_rejected(self):
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertIsNone(prepare_canvas_image(blank))

    def test_black_canvas_stroke_becomes_white_ink_on_black_28_pixel_image(self):
        canvas = Image.new("RGB", (100, 100), "white")
        drawing = ImageDraw.Draw(canvas)
        drawing.line((20, 50, 80, 50), fill="black", width=5)

        pixels = prepare_canvas_image(np.asarray(canvas))

        self.assertEqual(pixels.shape, (28, 28))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(int(pixels[0, 0]), 0)
        self.assertEqual(int(pixels.max()), 255)
        self.assertGreater(np.count_nonzero(pixels), 0)


if __name__ == "__main__":
    unittest.main()
