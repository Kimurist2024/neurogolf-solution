#!/usr/bin/env python3
"""Materialize deterministic fresh counterexamples for the task218 incumbent."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402

from audit_exact_lane import make_session  # noqa: E402


def labels(raw: np.ndarray, height: int, width: int) -> list[list[int]]:
    positive = raw[0, :, :height, :width] > 0
    out = np.zeros((height, width), dtype=np.int64)
    for row in range(height):
        for col in range(width):
            hits = np.flatnonzero(positive[:, row, col])
            out[row, col] = int(hits[0]) if len(hits) == 1 else -1
    return out.tolist()


def main() -> int:
    generator = importlib.import_module("task_90c28cc7")
    common = importlib.import_module("common")
    model = HERE / "base/task218.onnx"
    session = make_session(model.read_bytes(), True, 1)
    records = []
    seed = 21_839_401
    random.seed(seed)
    common.random.seed(seed)
    for index in range(1_500):
        example = generator.generate()
        converted = scoring.convert_to_numpy(example)
        assert converted is not None
        raw = np.asarray(
            session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: converted["input"]},
            )[0]
        )
        expected_mask = converted["output"].astype(bool)
        if raw.shape == expected_mask.shape and np.array_equal(raw > 0, expected_mask):
            continue
        expected = np.asarray(example["output"], dtype=np.int64)
        grid = np.asarray(example["input"], dtype=np.int64)
        records.append(
            {
                "index": index,
                "input_rows": ["".join(str(int(x)) for x in row) for row in grid],
                "expected": expected.tolist(),
                "observed_labels": labels(raw, *expected.shape),
                "threshold_differences": int(np.count_nonzero((raw > 0) != expected_mask)),
            }
        )
    payload = {"seed": seed, "attempts": 1_500, "mismatches": len(records), "records": records}
    path = HERE / "evidence/task218_counterexamples.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
