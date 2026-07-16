#!/usr/bin/env python3
"""Measure the repaired truthful-shape task196 controls."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.scoring import score_and_verify  # noqa: E402

MODELS = {
    "truthful_authority": HERE / "truthful_authority.onnx",
    "truthful_historical_968": HERE / "truthful_historical_968.onnx",
    "sound_bitset_u16": ROOT / "scripts/golf/scratch_codex/task196/agent_bitset/candidate_bitset_u16.onnx",
}


def main() -> None:
    results = {}
    for label, path in MODELS.items():
        results[label] = score_and_verify(
            onnx.load(path),
            196,
            str(HERE / "score_work"),
            label=label,
            require_correct=False,
        )
    (HERE / "truthful_costs.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
