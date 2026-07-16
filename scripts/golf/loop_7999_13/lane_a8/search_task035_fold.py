#!/usr/bin/env python3
"""Search which task035 7x feature lanes can be removed and refit exactly."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from scipy.optimize import Bounds, LinearConstraint, milp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


GROUP_COLUMNS = {
    "left2": 0,
    "core0b": 2,
    "core1b": 3,
    "core2b": 4,
    "core3b": 5,
    "right2": 9,
}


def collect() -> list[tuple[int, int, int, tuple[int, ...]]]:
    model = onnx.load(HERE / "baseline" / "task035.onnx")
    model.graph.output.extend(
        [helper.make_tensor_value_info("pack", TensorProto.UINT8, [1, 1, 10, 20])]
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options)
    examples = scoring.load_examples(35)
    samples: set[tuple[int, int, int, tuple[int, ...]]] = set()
    for key in ("train", "test", "arc-gen"):
        for item in examples[key]:
            benchmark = scoring.convert_to_numpy(item)
            if benchmark is None:
                continue
            pack = session.run(["pack"], {"input": benchmark["input"]})[0]
            expected = benchmark["output"].astype(np.uint8)
            for row in range(10):
                for col in range(10):
                    x = int(pack[0, 0, row, 2 * col])
                    z = int(pack[0, 0, row, 2 * col + 1])
                    labels = tuple(int(expected[0, ch, row, col]) for ch in range(10))
                    samples.add((col, x, z, labels))
    return sorted(samples)


def fit_channel(points: set[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    # Variables are integer effective weights a,b and ordinary int32 bias B.
    # Positive: a*dx+b*dz+B >= 1. Negative (including QConv pad): <= 0.
    rows = []
    lo = []
    hi = []
    for dx, dz, label in sorted(points):
        rows.append([dx, dz, 1])
        if label:
            lo.append(1)
            hi.append(np.inf)
        else:
            lo.append(-np.inf)
            hi.append(0)
    rows.append([0, 0, 1])
    lo.append(-np.inf)
    hi.append(0)
    result = milp(
        c=np.zeros(3),
        integrality=np.ones(3),
        bounds=Bounds([-128, -128, -1000000], [127, 127, 0]),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi)),
        options={"time_limit": 5.0},
    )
    if not result.success or result.x is None:
        return None
    answer = tuple(int(round(value)) for value in result.x)
    a, b, bias = answer
    for dx, dz, label in points:
        score = a * dx + b * dz + bias
        if (score >= 1) != bool(label):
            raise AssertionError((answer, dx, dz, label, score))
    if bias > 0:
        raise AssertionError(answer)
    return answer


def main() -> None:
    samples = collect()
    names = tuple(GROUP_COLUMNS)
    attempts = []
    winners = []
    for remove_count in range(len(names), -1, -1):
        for removed in itertools.combinations(names, remove_count):
            removed_columns = {GROUP_COLUMNS[name] for name in removed}
            fits = []
            for channel in range(10):
                points = {
                    (x - 128, (0 if col in removed_columns else z) - 128, labels[channel])
                    for col, x, z, labels in samples
                }
                fit = fit_channel(points)
                if fit is None:
                    break
                fits.append(fit)
            row = {"removed": list(removed), "fits": fits if len(fits) == 10 else None}
            attempts.append(row)
            if len(fits) == 10:
                winners.append(row)
        if winners:
            break
    payload = {
        "known_unique_pack_label_samples": len(samples),
        "max_removed": len(winners[0]["removed"]) if winners else None,
        "winners": winners,
        "attempt_count": len(attempts),
    }
    (HERE / "task035_fold_search.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
