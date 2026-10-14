"""Basic checks that the compact CNN maps one 28x28 image to ten class scores."""

import unittest

import torch

from drawsearch.models import CompactCNNV1, CompactCNNV2


class CompactCNNTests(unittest.TestCase):
    def test_output_has_one_score_per_class(self):
        model = CompactCNNV1(number_of_classes=10)
        with torch.no_grad():
            scores = model(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(scores.shape), (4, 10))

    def test_one_convolution_v2_outputs_one_score_per_class(self):
        model = CompactCNNV2(number_of_classes=10)
        with torch.no_grad():
            scores = model(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(scores.shape), (4, 10))
        self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()), 1)


if __name__ == "__main__":
    unittest.main()
