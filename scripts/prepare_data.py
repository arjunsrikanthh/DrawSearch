#!/usr/bin/env python3
"""Download Quick, Draw! drawings and make the project's fixed dataset splits."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drawsearch.data import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data.yaml",
        help="Dataset settings file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing prepared splits and rebuild them from the raw files.",
    )
    args = parser.parse_args()
    prepare_dataset(args.config.resolve(), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
