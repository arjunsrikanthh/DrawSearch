"""Small checks that protect the reproducible source-row splitting rules."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from drawsearch.data import prepare_dataset, split_source_indices


class SplitSourceIndicesTests(unittest.TestCase):
    def test_split_sizes_are_exact_and_source_rows_do_not_overlap(self):
        splits = split_source_indices(
            source_count=1_000,
            sample_count=100,
            split_counts={"train": 80, "val": 10, "test": 10},
            seed=42,
        )

        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 80,
            "val": 10,
            "test": 10,
        })
        combined = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(combined)), 100)

    def test_same_seed_produces_the_same_split(self):
        first = split_source_indices(1_000, 100, {"train": 80, "val": 10, "test": 10}, 42)
        second = split_source_indices(1_000, 100, {"train": 80, "val": 10, "test": 10}, 42)

        for split_name in first:
            np.testing.assert_array_equal(first[split_name], second[split_name])

    def test_rejects_impossible_sizes(self):
        with self.assertRaises(ValueError):
            split_source_indices(20, 30, {"train": 24, "val": 3, "test": 3}, 42)

    def test_small_local_dataset_creates_archives_manifest_and_sample_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw_dir = root / "data/raw"
            raw_dir.mkdir(parents=True)
            rng = np.random.default_rng(5)
            for category in ("cat", "dog"):
                np.save(raw_dir / f"{category}.npy", rng.integers(0, 256, (20, 784), dtype=np.uint8))

            config = {
                "dataset": "test_fixture",
                "source_base_url": "https://example.invalid",
                "categories": ["cat", "dog"],
                "seed": 42,
                "samples_per_class": 10,
                "split_counts_per_class": {"train": 8, "val": 1, "test": 1},
                "image_shape": [28, 28],
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
            }
            config_path = root / "tiny.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            # A checked-in provenance manifest can exist before a fresh clone
            # has downloaded or generated its ignored split archives.
            (root / "data/processed").mkdir(parents=True)
            (root / "data/processed/manifest.json").write_text("{}\n", encoding="utf-8")

            manifest_path, inspection_path = prepare_dataset(config_path, root=root)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["class_order"], ["cat", "dog"])
            self.assertEqual(manifest["split_archives"]["train"]["drawings"], 16)
            self.assertEqual(manifest["split_archives"]["val"]["drawings"], 2)
            self.assertEqual(manifest["split_archives"]["test"]["drawings"], 2)
            self.assertEqual(inspection["pixel_dtype"], "uint8")
            self.assertTrue((root / inspection["sample_grid"]).is_file())

            with np.load(root / "data/processed/train.npz") as train:
                self.assertEqual(train["images"].shape, (16, 784))
                self.assertEqual(set(np.unique(train["labels"])), {0, 1})


if __name__ == "__main__":
    unittest.main()
