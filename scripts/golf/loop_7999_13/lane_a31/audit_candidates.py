#!/usr/bin/env python3
"""Fast dual-ORT known-set and official-cost screen for A31 probes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known(task: int, model: onnx.ModelProto, disabled: bool) -> dict[str, object]:
    try:
        sess = session(model, disabled)
    except Exception as exc:  # noqa: BLE001
        return {"right": 0, "wrong": 0, "errors": 1, "session_error": repr(exc)}
    right = wrong = errors = 0
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
    return {"right": right, "wrong": wrong, "errors": errors}


def audit(path: Path) -> dict[str, object]:
    task = int(path.name[4:7])
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    result: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "task": task,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "params_static": sum(math.prod(item.dims) for item in model.graph.initializer),
        "nodes": len(model.graph.node),
        "op_histogram": dict(Counter(item.op_type for item in model.graph.node)),
        "strict_value_info": len(inferred.graph.value_info),
        "disable_all": known(task, model, True),
        "default": known(task, model, False),
    }
    total = sum(len(scoring.load_examples(task)[part]) for part in ("train", "test", "arc-gen"))
    perfect = all(
        result[label] == {"right": total, "wrong": 0, "errors": 0}
        for label in ("disable_all", "default")
    )
    result["known_perfect"] = perfect
    if perfect:
        result["score"] = scoring.score_and_verify(
            model, task, str(HERE / "profile"), label=path.stem, require_correct=True
        )
    return result


def main() -> None:
    paths = sorted(HERE.glob("task306_*.onnx"))
    rows = [audit(path) for path in paths]
    (HERE / "candidate_screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["path"], row["params_static"], row["disable_all"], row["default"])


if __name__ == "__main__":
    main()
