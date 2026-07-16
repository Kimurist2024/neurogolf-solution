#!/usr/bin/env python3
"""Record raw-logit differences for task379 factor-reuse failures."""

from __future__ import annotations

import importlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import onnx

from dual_ort_fresh import MAP, ROOT, make_session, scoring


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "lane_tensor_mode_reuse" / "task379_nv_from_qv.onnx"
OUTPUT = HERE / "lane_tensor_mode_reuse" / "task379_failure_logits.json"


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission_base_7999.13.zip") as archive:
        baseline = onnx.load_model(io.BytesIO(archive.read("task379.onnx")))
    candidate = onnx.load(CANDIDATE)
    baseline_session = make_session(baseline, True)
    candidate_session = make_session(candidate, True)
    module = importlib.import_module(f"task_{MAP['379']}")
    random.seed(37_900_379)
    rows: list[dict[str, object]] = []
    for index in range(5000):
        example = module.generate()
        benchmark = scoring.convert_to_numpy(example)
        if not benchmark:
            continue
        raw_base = baseline_session.run(["output"], {"input": benchmark["input"]})[0]
        raw_cand = candidate_session.run(["output"], {"input": benchmark["input"]})[0]
        expected = benchmark["output"].astype(bool)
        predicted = raw_cand > 0
        if np.array_equal(predicted, expected):
            continue
        coordinates = np.argwhere(predicted != expected)
        details = []
        for coordinate in coordinates[:40]:
            key = tuple(int(value) for value in coordinate)
            details.append(
                {
                    "coordinate": list(key),
                    "expected": bool(expected[key]),
                    "baseline": float(raw_base[key]),
                    "candidate": float(raw_cand[key]),
                    "raw_delta": float(raw_cand[key] - raw_base[key]),
                }
            )
        rows.append(
            {
                "index": index,
                "mismatch_count": int(coordinates.shape[0]),
                "details": details,
                "max_abs_raw_delta": float(np.max(np.abs(raw_cand - raw_base))),
            }
        )
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"failures": len(rows), "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
