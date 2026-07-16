#!/usr/bin/env python3
"""Lightweight execution screen for non-lower exact candidates (not a deep gate)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE / "base"

sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402


def options() -> ort.SessionOptions:
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    value.intra_op_num_threads = value.inter_op_num_threads = 1
    value.log_severity_level = 4
    return value


def session(path: Path) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitizer rejected")
    return ort.InferenceSession(model.SerializeToString(), options(), providers=["CPUExecutionProvider"])


def first_known(task: int) -> np.ndarray:
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                return converted["input"]
    raise RuntimeError(task)


def main() -> None:
    result = json.loads((HERE / "profile_results.json").read_text())
    source_sessions = {task: session(BASE / f"task{task:03d}.onnx") for task in (74, 200, 211)}
    inputs = {task: first_known(task) for task in source_sessions}
    source_outputs = {
        task: np.asarray(source_sessions[task].run(["output"], {"input": inputs[task]})[0])
        for task in source_sessions
    }
    rows = []
    for row in result["rows"]:
        task = int(row["task"])
        entry: dict[str, Any] = {"label": row["label"], "task": task}
        try:
            value = np.asarray(session(REPO / row["path"]).run(["output"], {"input": inputs[task]})[0])
            source = source_outputs[task]
            entry.update(
                {
                    "runtime_ok": True,
                    "shape": list(value.shape),
                    "raw_equal": bool(np.array_equal(source, value)),
                    "threshold_equal": bool(np.array_equal(source > 0, value > 0)),
                    "source_nonfinite": int(source.size - np.count_nonzero(np.isfinite(source))),
                    "candidate_nonfinite": int(value.size - np.count_nonzero(np.isfinite(value))),
                }
            )
        except BaseException as exc:
            entry.update({"runtime_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(entry)
    payload = {
        "mode": "ORT_DISABLE_ALL_threads1_first_convertible_known_ONLY",
        "not_a_deep_gate": True,
        "rows": rows,
        "runtime_ok_count": sum(bool(row["runtime_ok"]) for row in rows),
        "raw_equal_count": sum(bool(row.get("raw_equal")) for row in rows),
        "threshold_equal_count": sum(bool(row.get("threshold_equal")) for row in rows),
    }
    (HERE / "one_known_screen.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("runtime_ok_count", "raw_equal_count", "threshold_equal_count")}, indent=2))


if __name__ == "__main__":
    main()
