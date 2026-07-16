#!/usr/bin/env python3
"""Score and known-dual audit local exact task196 probes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

MODELS = {
    "authority": HERE / "baseline_task196.onnx",
    "g_bool_cast": HERE / "probe_g_bool_cast.onnx",
    "g_bool_greater": HERE / "probe_g_bool_greater.onnx",
}
MODES = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "default": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def session(path: Path, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    assert model is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def main() -> None:
    examples = scoring.load_examples(196)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    rows = []
    for mode, level in MODES.items():
        sessions = {label: session(path, level) for label, path in MODELS.items()}
        for label in MODELS:
            row = {
                "mode": mode,
                "label": label,
                "correct": 0,
                "wrong": 0,
                "errors": 0,
                "raw_equal_authority": 0,
            }
            for example in known:
                benchmark = scoring.convert_to_numpy(example)
                assert benchmark is not None
                authority_raw = sessions["authority"].run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
                try:
                    raw = sessions[label].run(["output"], {"input": benchmark["input"]})[0]
                    row["correct" if np.array_equal(raw > 0, benchmark["output"] > 0) else "wrong"] += 1
                    row["raw_equal_authority"] += int(np.array_equal(raw, authority_raw))
                except Exception:
                    row["errors"] += 1
            rows.append(row)
    costs = {}
    for label, path in MODELS.items():
        costs[label] = scoring.score_and_verify(
            onnx.load(path),
            196,
            str(HERE / "score_work"),
            label=f"probe_{label}",
            require_correct=False,
        )
    (HERE / "exact_probe_audit.json").write_text(
        json.dumps({"known_cases": len(known), "rows": rows, "costs": costs}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
