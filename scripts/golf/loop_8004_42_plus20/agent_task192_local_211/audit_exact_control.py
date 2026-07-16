#!/usr/bin/env python3
"""Known-only audit for the cost-gated task192 exact ArgMax control."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1149.onnx.fallback"
CANDIDATE = HERE / "candidates/task192_center_direct_argmax.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def session(path: Path, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main() -> None:
    payload = scoring.load_examples(192)
    cases = [row for split in ("train", "test", "arc-gen") for row in payload.get(split, [])]
    result = {"known_count": len(cases), "fresh_skipped": "candidate cost1143 is not below current1138", "configs": {}}
    for disable, threads, label in CONFIGS:
        old, new = session(SOURCE, disable, threads), session(CANDIDATE, disable, threads)
        row = {"raw_equal": 0, "sign_equal": 0, "right": 0, "errors": 0, "nonfinite": 0}
        for case in cases:
            benchmark = scoring.convert_to_numpy(case)
            if benchmark is None:
                row["errors"] += 1
                continue
            try:
                a = np.asarray(old.run(None, {old.get_inputs()[0].name: benchmark["input"]})[0])
                b = np.asarray(new.run(None, {new.get_inputs()[0].name: benchmark["input"]})[0])
            except Exception:  # noqa: BLE001
                row["errors"] += 1
                continue
            row["raw_equal"] += int(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
            row["sign_equal"] += int(np.array_equal(a > 0, b > 0))
            row["right"] += int(np.array_equal(b > 0, benchmark["output"].astype(bool)))
            row["nonfinite"] += int(a.size - np.count_nonzero(np.isfinite(a)))
            row["nonfinite"] += int(b.size - np.count_nonzero(np.isfinite(b)))
        row["pass"] = bool(
            row["raw_equal"] == row["sign_equal"] == row["right"] == len(cases)
            and row["errors"] == row["nonfinite"] == 0
        )
        result["configs"][label] = row
    result["pass"] = all(row["pass"] for row in result["configs"].values())
    (HERE / "audit_exact_control.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
