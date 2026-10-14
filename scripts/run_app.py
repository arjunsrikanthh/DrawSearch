#!/usr/bin/env python3
"""Start the trained DrawSearch classifier on your own computer."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drawsearch.app import build_app, load_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--host", default="127.0.0.1", help="Local address to serve (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, class_names, device = load_classifier(args.device)
    app = build_app(model, class_names, device)
    print(f"DrawSearch is starting at http://{args.host}:{args.port} using {device}.", flush=True)
    app.launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
