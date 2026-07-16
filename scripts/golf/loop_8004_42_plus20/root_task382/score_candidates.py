#!/usr/bin/env python3
"""Official-like cost comparison for current and repaired task382 models."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8004.50.zip"
REPAIRED = HERE / "task382_truthful_shapes.onnx"


def main() -> int:
    with zipfile.ZipFile(BASE) as archive:
        baseline = onnx.load_model_from_string(archive.read("task382.onnx"))
    results = {}
    with tempfile.TemporaryDirectory(prefix="task382_score_") as workdir:
        for label, model in (("baseline", baseline), ("repaired", onnx.load(REPAIRED))):
            results[label] = scoring.score_and_verify(
                model, 382, workdir, label=label, require_correct=False
            )
    (HERE / "cost_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
