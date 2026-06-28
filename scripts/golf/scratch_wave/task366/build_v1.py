#!/usr/bin/env python3
"""Build the fresh-safe exact task366 reference candidate.

This delegates to the prior spec-derived robust builder under
scripts/golf/scratch/task366/build_full2.py and writes the candidate into this
scratch_wave directory, keeping all outputs for this run local to the allowed
task366 scratch path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import onnx


ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "scripts/golf/scratch/task366"
OUT = Path(__file__).with_name("cand.onnx")


def main() -> None:
    sys.path.insert(0, str(SRC_DIR))
    import build_full2  # noqa: PLC0415

    graph_builder = build_full2.build()
    model = build_full2.make_final(graph_builder)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
