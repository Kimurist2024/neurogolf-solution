#!/usr/bin/env python3
"""Run every known pair under both required ORT optimization modes."""

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
TARGETS = (149, 390, 272, 147, 40, 176, 252, 127)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def run(model: onnx.ModelProto, task: int, disable: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    examples = scoring.load_examples(task)
    total = sum(len(examples[name]) for name in ("train", "test", "arc-gen"))
    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {
            "right": 0,
            "wrong": 0,
            "errors": total,
            "total": total,
            "session_error": f"{type(exc).__name__}: {exc}",
        }
    right = wrong = errors = skipped = 0
    first_failure = None
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                skipped += 1
                continue
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {
                        "split": split,
                        "index": index,
                        "kind": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped": skipped,
        "total": right + wrong + errors,
        "first_failure": first_failure,
    }


def main() -> None:
    rows = []
    for task in TARGETS:
        model = onnx.load(HERE / "base" / f"task{task:03d}.onnx")
        row = {
            "task": task,
            "disable_all": run(model, task, True),
            "default": run(model, task, False),
        }
        rows.append(row)
        print(
            f"task{task:03d}: disable={row['disable_all']['right']}/{row['disable_all']['total']} "
            f"default={row['default']['right']}/{row['default']['total']}",
            flush=True,
        )
    result = {"targets_completed": len(rows), "rows": rows}
    (HERE / "known_dual.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
