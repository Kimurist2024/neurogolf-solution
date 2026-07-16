#!/usr/bin/env python3
"""Readable rule audit and constructive task319 non-identifiability witness."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from fresh_verify import raw_p319  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def task319_collision() -> dict[str, Any]:
    generator = importlib.import_module("task_ce602527")
    common = {
        "width": 15,
        "height": 18,
        "rows": [0, 1, 2, 2, 0, 1, 2, 2],
        "cols": [1, 2, 0, 1, 1, 2, 0, 1],
        "idxs": [0, 0, 0, 0, 1, 1, 1, 1],
        "magrow": 5,
        "magcol": 11,
        "magcolor": 2,
        "bgcolor": 9,
    }
    first = generator.generate(brows=[5, 0], bcols=[1, 7], colors=[5, 4], **common)
    second = generator.generate(brows=[0, 5], bcols=[7, 1], colors=[4, 5], **common)
    assert first["input"] == second["input"]
    assert first["output"] != second["output"]
    return {
        "same_input": True,
        "different_output": True,
        "input_sha256": digest(first["input"]),
        "output_a_sha256": digest(first["output"]),
        "output_b_sha256": digest(second["output"]),
        "output_a": first["output"],
        "output_b": second["output"],
        "consequence": (
            "No deterministic ONNX can be correct for every valid generator call; "
            "the latent identity of sprite0 is not always observable from the grid."
        ),
    }


def main() -> int:
    dataset = json.loads((ROOT / "inputs/neurogolf-2026/task319.json").read_text())
    known_total = known_right = 0
    first_failure = None
    for subset in ("train", "test", "arc-gen"):
        for index, pair in enumerate(dataset[subset]):
            known_total += 1
            got = raw_p319(pair["input"])
            if got == pair["output"]:
                known_right += 1
            elif first_failure is None:
                first_failure = {"subset": subset, "index": index}
    report = {
        "task319": {
            "classification": "Type D: global magnification cross-correlation plus data-dependent crop",
            "readable_rule": [
                "background is the most frequent color",
                "magnified color is the most frequent non-background color",
                "score each remaining color by the maximum translation-aligned overlap between its 2x expansion and the magnified pixels",
                "select the highest score with the raw solver's count tie-break and crop that color's bounding box",
            ],
            "known_rule_check": {
                "right": known_right,
                "total": known_total,
                "first_failure": first_failure,
            },
            "constructive_generator_collision": task319_collision(),
            "fresh_rule_check_source": "audit/fresh_two_seed.json",
        },
        "task367": {
            "classification": "Type B/D bounded structural scan: hollow rectangle interiors with connector-line rejection",
            "readable_rule": [
                "input colors are black 0 and gray 5",
                "find the interiors of 2-4 hollow gray rectangles, including one-column left/right clipping",
                "exclude gray connector corridors by detecting corner/endpoints and propagating each top-frame interior span downward through black cells",
                "recolor exactly those interior black cells yellow 4",
            ],
            "generator_bounds": {
                "grid_width_height": [10, 20],
                "box_count": [2, 4],
                "box_width_height": [3, 7],
                "horizontal_clip": "at most one column at either side",
            },
            "prior_independent_true_rule_evidence": (
                "scripts/golf/scratch_claude/task367/REPORT.md and "
                "scripts/golf/loop_8004_42_plus20/agent_rebuild_mid5/REPORT.md"
            ),
        },
    }
    (HERE / "audit/rule_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
