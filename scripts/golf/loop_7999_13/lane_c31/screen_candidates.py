#!/usr/bin/env python3
"""Dual-ORT fail-fast then complete-known screen for C31 candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def run(path: Path, task: int, fail_fast: bool) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    examples = scoring.load_examples(task)
    rows = []
    for mode in ("disabled", "default"):
        sess = make_session(model, mode)
        right = wrong = errors = 0
        first_failure = None
        stop = False
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                    if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                        right += 1
                    else:
                        wrong += 1
                        first_failure = first_failure or {"subset": subset, "index": index}
                        stop = fail_fast
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    first_failure = first_failure or {
                        "subset": subset,
                        "index": index,
                        "error": repr(exc),
                    }
                    stop = fail_fast
                if stop:
                    break
            if stop:
                break
        rows.append(
            {
                "mode": mode,
                "right": right,
                "wrong": wrong,
                "errors": errors,
                "first_failure": first_failure,
            }
        )
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fail_fast": fail_fast,
        "known_dual": rows,
    }


def main() -> int:
    inventory = json.loads((HERE / "history_inventory.json").read_text())
    paths = [
        ROOT / row["path"]
        for row in inventory["tasks"]["199"]["models"]
        if row["cost"] < inventory["tasks"]["199"]["baseline_cost"]
    ]
    seen = set()
    paths = [
        path
        for path in paths
        if not (hashlib.sha256(path.read_bytes()).hexdigest() in seen)
        and not seen.add(hashlib.sha256(path.read_bytes()).hexdigest())
    ]
    candidates = [(path, 199) for path in paths]
    result = []
    for path, task in candidates:
        row = run(path, task, fail_fast=True)
        result.append(row)
        print(path.name, row["known_dual"], flush=True)
    (HERE / "candidate_screen.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
