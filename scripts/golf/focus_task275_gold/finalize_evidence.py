#!/usr/bin/env python3
"""Aggregate the official gate and eight deterministic fresh chunks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = 275
AUTHORITY_COST = 428
CANDIDATE = HERE / "task275_diagonal_reuse_cost419_c7ddaab77f6d.onnx"
EXPECTED_SHA256 = "c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2"
SEEDS = (275_419_001, 275_419_002)


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LANE = import_path(
    "task275_gold_finalize_support",
    ROOT / "scripts/golf/cost351_500_gold_loop/worker.py",
)


def load_seed(seed: int) -> dict[str, Any]:
    chunks = [
        json.loads((HERE / f"fresh_seed{seed}_chunk{chunk}.json").read_text())
        for chunk in range(4)
    ]
    expected_ranges = [(chunk * 500, (chunk + 1) * 500) for chunk in range(4)]
    actual_ranges = [(int(row["start"]), int(row["stop"])) for row in chunks]
    stream_hashes = {row["generation"]["case_stream_sha256"] for row in chunks}
    passed = bool(
        actual_ranges == expected_ranges
        and len(stream_hashes) == 1
        and all(
            row["candidate_sha256"] == EXPECTED_SHA256
            and int(row["seed"]) == seed
            and int(row["case_count"]) == 500
            and row["pass"]
            and int(row["runtime"]["right"]) == 500
            and int(row["runtime"]["wrong"]) == 0
            and int(row["runtime"]["errors"]) == 0
            and int(row["runtime"]["nonfinite_cases"]) == 0
            and int(row["runtime"]["runtime_shape_mismatches"]) == 0
            and int(row["runtime"]["small_positive_elements_0_to_0_25"]) == 0
            for row in chunks
        )
    )
    return {
        "seed": seed,
        "case_stream_sha256": next(iter(stream_hashes)) if len(stream_hashes) == 1 else None,
        "case_count": sum(int(row["case_count"]) for row in chunks),
        "right": sum(int(row["runtime"]["right"]) for row in chunks),
        "wrong": sum(int(row["runtime"]["wrong"]) for row in chunks),
        "errors": sum(int(row["runtime"]["errors"]) for row in chunks),
        "minimum_positive": min(float(row["runtime"]["minimum_positive"]) for row in chunks),
        "maximum_nonpositive": max(
            float(row["runtime"]["maximum_nonpositive"]) for row in chunks
        ),
        "chunks": chunks,
        "pass": passed,
    }


def main() -> int:
    digest = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError((digest, EXPECTED_SHA256))
    official = LANE.official_gate(CANDIDATE, TASK, AUTHORITY_COST)
    fresh = [load_seed(seed) for seed in SEEDS]
    passed = bool(
        official["pass"]
        and int(official["candidate_cost"]) == 419
        and all(row["pass"] and row["right"] == 2_000 for row in fresh)
    )
    gain = math.log(AUTHORITY_COST / 419)
    payload = {
        "task": TASK,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest,
        "authority": "submission_base_8012.23.zip::task275.onnx",
        "authority_cost": AUTHORITY_COST,
        "candidate_cost": 419,
        "score_gain": gain,
        "official_gate": official,
        "fresh": fresh,
        "absolute_gate": (
            "official gold exact + strict checker/static shape + margin + score; "
            "two independent deterministic fresh streams, each 2000/2000"
        ),
        "lane_root_writes": [],
        "pass": passed,
    }
    (HERE / "final_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("absolute audit rejected candidate")

    report = f"""# task275 gold-exact diagonal-reuse result

- Authority: cost {AUTHORITY_COST}
- Candidate: cost 419
- Projected gain: +{gain:.12f}
- Candidate SHA-256: `{digest}`
- Official gold: exact pass
- Strict checker/static shape: pass
- Minimum positive raw margin: {official['minimum_positive']}
- Fresh seed {SEEDS[0]}: 2000/2000, errors 0
- Fresh seed {SEEDS[1]}: 2000/2000, errors 0
- Small positives in `(0, 0.25)`: 0
- Non-finite cases / runtime shape mismatches: 0 / 0
- Root submission/CSV/score-pointer writes by this lane: none

The final Einsum reuses the same learned 3x3 color map in both former T/W
roles and reads its diagonal with repeated subscript `aa` as the W-row scale.
This preserves the required sign-rank three while removing W's nine
parameters.  The spatial router is unchanged.

The earlier `task275_diag_color_cost413_351d0b2a8557.onnx` experiment is
rejected: it incorrectly collapsed quotient/remainder distractor colors and
failed official gold at train[1].  It is not an admission.
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pass": True, "cost": 419, "gain": gain}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
