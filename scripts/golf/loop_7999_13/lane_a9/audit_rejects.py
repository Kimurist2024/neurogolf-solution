#!/usr/bin/env python3
"""Record dual-ORT known and structural evidence for A9 rejected candidates."""

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
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


CANDIDATES = {
    25: HERE / "task025_causal_mask.onnx",
    268: HERE / "task268_cast_bool.onnx",
}


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(task: int, session: ort.InferenceSession) -> dict[str, object]:
    subsets: dict[str, dict[str, int]] = {}
    for name in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in scoring.load_examples(task)[name]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                got = session.run(["output"], {"input": benchmark["input"]})[0] > 0.0
                if np.array_equal(got, benchmark["output"] > 0.0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001 - errors are audit data
                errors += 1
        subsets[name] = {"right": right, "wrong": wrong, "errors": errors}
    subsets["total"] = {
        key: sum(int(subsets[name][key]) for name in ("train", "test", "arc-gen"))
        for key in ("right", "wrong", "errors")
    }
    return subsets


def structure(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return {
        "checker_full": "pass",
        "strict_shape_inference": "pass",
        "domains": sorted({item.domain for item in inferred.opset_import}),
        "functions": len(inferred.functions),
        "sparse_initializers": len(inferred.graph.sparse_initializer),
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    rows: dict[str, object] = {}
    for task, path in CANDIDATES.items():
        model = onnx.load(path)
        row: dict[str, object] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "structure": structure(model),
        }
        row["profile"] = scoring.score_and_verify(
            copy.deepcopy(model),
            task,
            str(HERE / "score_work"),
            label=f"a9_reject_{task:03d}",
            require_correct=False,
        )
        for disable_all, label in ((True, "ort_disable_all"), (False, "ort_default")):
            try:
                row[label] = known(task, make_session(model, disable_all))
            except Exception as exc:  # noqa: BLE001 - errors are audit data
                row[label] = {"session_error": f"{type(exc).__name__}: {exc}"}
        rows[str(task)] = row
        print(task, json.dumps(row, sort_keys=True), flush=True)
    (HERE / "rejected_candidate_audit.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
