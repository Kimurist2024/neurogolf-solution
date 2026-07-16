#!/usr/bin/env python3
"""Fail-fast dual-ORT known screen for every B24 lower-cost probe."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def run_known(model: onnx.ModelProto, task: int, disabled: bool) -> dict[str, object]:
    examples = scoring.load_examples(task)
    ordered = [
        (subset, index, example)
        for subset in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[subset])
    ]
    try:
        sess = session(model, disabled)
    except Exception as error:
        return {
            "right": 0,
            "wrong": 0,
            "errors": 1,
            "session_error": f"{type(error).__name__}: {error}",
        }
    right = wrong = errors = 0
    first_failure = None
    # Fail fast after the first semantic mismatch. A candidate must have zero
    # mismatches, so further cases cannot restore eligibility.
    for subset, index, example in ordered:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            raw = sess.run(["output"], {"input": benchmark["input"]})[0]
            if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                right += 1
            else:
                wrong += 1
                first_failure = {"subset": subset, "index": index}
                break
        except Exception as error:
            errors += 1
            first_failure = {
                "subset": subset,
                "index": index,
                "error": f"{type(error).__name__}: {error}",
            }
            break
    complete = right == len(ordered) and wrong == 0 and errors == 0
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "known_total": len(ordered),
        "complete": complete,
        "first_failure": first_failure,
    }


def main() -> int:
    build = json.loads((HERE / "build_manifest.json").read_text())
    rows: list[dict[str, object]] = []
    for built in build["rows"]:
        path = ROOT / built["path"]
        model = onnx.load(path)
        memory, params, cost = (int(value) for value in cost_of(str(path)))
        disabled = run_known(model, int(built["task"]), True)
        default = run_known(model, int(built["task"]), False)
        eligible_known = bool(disabled["complete"] and default["complete"])
        row = {
            **built,
            "actual_memory": memory,
            "actual_params": params,
            "actual_cost": cost,
            "disable_all_known": disabled,
            "default_known": default,
            "eligible_known": eligible_known,
            "decision": "continue_full_audit" if eligible_known else "reject_known",
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "task": row["task"],
                    "path": row["path"],
                    "cost": cost,
                    "disabled": (disabled["right"], disabled["wrong"], disabled["errors"]),
                    "default": (default["right"], default["wrong"], default["errors"]),
                    "decision": row["decision"],
                }
            ),
            flush=True,
        )
    payload = {
        "baseline_sha256": build["baseline_sha256"],
        "candidate_count": len(rows),
        "known_survivors": sum(bool(row["eligible_known"]) for row in rows),
        "rows": rows,
    }
    (HERE / "candidate_screen.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
