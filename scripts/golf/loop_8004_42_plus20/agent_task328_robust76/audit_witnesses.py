#!/usr/bin/env python3
"""Fix legal finite-support witnesses for the task328 scaling impossibility."""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import random
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC = importlib.util.spec_from_file_location("orbit_audit", HERE / "audit_orbits.py")
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task328_r001.onnx"
SCALE14 = HERE / "task328_scale2p14.onnx"
SCALE32 = HERE / "task328_scale2p32.onnx"


def run(path: Path, example: dict) -> dict:
    benchmark = AUDIT.scoring.convert_to_numpy(example)
    assert benchmark is not None
    expected = benchmark["output"].astype(bool)
    rows = {}
    for disabled, threads, label in AUDIT.CONFIGS:
        session = AUDIT.make_session(path, disabled, threads)
        raw = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
        true_raw = raw[expected]
        false_raw = raw[~expected]
        finite = np.isfinite(raw)
        rows[label] = {
            "correct": bool(np.array_equal(raw > 0, expected)),
            "nonfinite_values": int(raw.size - np.count_nonzero(finite)),
            "min_true": float(true_raw.min()) if np.isfinite(true_raw).all() else None,
            "true_below_0_25": int(np.count_nonzero(true_raw < 0.25)),
            "true_below_1": int(np.count_nonzero(true_raw < 1.0)),
            "false_positive": int(np.count_nonzero(false_raw > 0)),
            "max_false": float(false_raw.max()) if np.isfinite(false_raw).all() else None,
            "max_abs_finite": float(np.abs(raw[finite]).max(initial=0.0)),
        }
    return rows


def high_witness() -> tuple[dict, dict]:
    """Find the highest-magnitude case in the first checkpointed 25 orbits."""
    cases = AUDIT.canonical_cases()[:25]
    session = AUDIT.make_session(SOURCE, True, 1)
    best = None
    best_abs = -1.0
    for index, (example, meta) in enumerate(cases, start=1):
        benchmark = AUDIT.scoring.convert_to_numpy(example)
        assert benchmark is not None
        raw = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
        value = float(np.abs(raw[np.isfinite(raw)]).max(initial=0.0))
        if value > best_abs:
            best_abs = value
            best = (example, {"orbit_case": index, **meta, "source_max_abs": value})
    assert best is not None
    return best


def low_witnesses() -> list[tuple[dict, dict]]:
    random.seed(328260000)
    np.random.seed(328260000)
    generator = importlib.import_module("task_d22278a0")
    return [(generator.generate(), {"seed": 328260000, "case": index}) for index in range(1, 5)]


def main() -> int:
    high_example, high_meta = high_witness()
    low_rows = []
    for example, meta in low_witnesses():
        low_rows.append({"meta": meta, "scale2p14": run(SCALE14, example)})
    output = {
        "source_min_true_retained": {
            "value": 7.316870026530253e-11,
            "source": "scripts/golf/loop_7999_13/lane_b26/winner_manifest.json",
            "same_sha256": "4d0fc5264833fbf46609fde690ad8635e208a2cec381e749b5707ef828866cb2"
        },
        "source_high_witness": high_meta,
        "scale2p14_low_seed_cases": low_rows,
        "scale2p32_high_witness": {
            "meta": high_meta,
            "configs": run(SCALE32, high_example),
        },
    }
    (HERE / "witness_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
