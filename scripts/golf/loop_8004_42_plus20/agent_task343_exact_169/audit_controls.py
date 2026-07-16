#!/usr/bin/env python3
"""Strict structural, four-mode known, and 10k multi-seed fresh task343 audit."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTROLS = HERE / "controls"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_d8c310e9")
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
MODES = (
    ("disable_t1", True, 1),
    ("disable_t4", True, 4),
    ("default_t1", False, 1),
    ("default_t4", False, 4),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = load_module(
    "task343_exact169_auditor",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(model: onnx.ModelProto, disable_all: bool, threads: int):
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def update_numeric(row: dict[str, Any], raw: np.ndarray) -> None:
    row["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(raw)))
    positive = raw[raw > 0]
    if positive.size:
        current = float(np.min(positive))
        row["min_positive"] = (
            current if row["min_positive"] is None else min(row["min_positive"], current)
        )
    row["small_positive_0_to_0_25"] += int(
        np.count_nonzero((raw > 0) & (raw < 0.25))
    )


def counters() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite_values": 0,
        "small_positive_0_to_0_25": 0,
        "min_positive": None,
        "first_failure": None,
    }


def run_one(session: ort.InferenceSession, benchmark: dict[str, np.ndarray]):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(session.run([output_name], {input_name: benchmark["input"]})[0])


def known_four(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    examples = scoring.load_examples(343)
    for name, disable, threads in MODES:
        row = counters()
        try:
            session = make_session(model, disable, threads)
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    try:
                        raw = run_one(session, benchmark)
                        update_numeric(row, raw)
                        if np.array_equal(raw > 0, benchmark["output"] > 0):
                            row["right"] += 1
                        else:
                            row["wrong"] += 1
                            if row["first_failure"] is None:
                                row["first_failure"] = {"subset": subset}
                    except Exception as exc:  # noqa: BLE001
                        row["errors"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "subset": subset,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
        result[name] = row
        print("known", name, row["right"], row["wrong"], row["errors"], flush=True)
    return result


def fresh_exact_four(model: onnx.ModelProto) -> list[dict[str, Any]]:
    sessions = {
        name: make_session(model, disable, threads)
        for name, disable, threads in MODES
    }
    runs: list[dict[str, Any]] = []
    for seed in (343_169_101, 343_169_102):
        rows = {name: counters() for name in sessions}
        generation_errors = 0
        conversion_skips = 0
        for case in range(5000):
            try:
                random.seed(seed + case)
                example = GEN.generate()
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                conversion_skips += 1
                continue
            for name, session in sessions.items():
                row = rows[name]
                try:
                    raw = run_one(session, benchmark)
                    update_numeric(row, raw)
                    if np.array_equal(raw > 0, benchmark["output"] > 0):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {"case": case}
                except Exception as exc:  # noqa: BLE001
                    row["errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "case": case,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        run = {
            "seed": seed,
            "requested": 5000,
            "generation_errors": generation_errors,
            "conversion_skips": conversion_skips,
            "modes": rows,
        }
        runs.append(run)
        print(
            "fresh",
            seed,
            {name: (row["right"], row["wrong"], row["errors"]) for name, row in rows.items()},
            flush=True,
        )
    return runs


def counterexample_run(model: onnx.ModelProto, seed: int, requested: int = 5000):
    session = make_session(model, True, 1)
    row = counters()
    for case in range(requested):
        random.seed(seed + case)
        benchmark = scoring.convert_to_numpy(GEN.generate())
        if benchmark is None:
            continue
        raw = run_one(session, benchmark)
        update_numeric(row, raw)
        if np.array_equal(raw > 0, benchmark["output"] > 0):
            row["right"] += 1
        else:
            row["wrong"] += 1
            if row["first_failure"] is None:
                row["first_failure"] = {"case": case}
    row.update(seed=seed, requested=requested)
    return row


def structural(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    ops = Counter(node.op_type for node in model.graph.node)
    row: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path.read_bytes()),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_histogram": dict(ops),
        "finite_initializers": all(
            np.all(np.isfinite(np.asarray(numpy_helper.to_array(item))))
            for item in model.graph.initializer
            if np.issubdtype(np.asarray(numpy_helper.to_array(item)).dtype, np.number)
        ),
        "lookup_ops": [
            node.op_type for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax"}
        ],
        "banned_ops": [
            node.op_type for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        ],
        "custom_domains": sorted(
            {item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}}
            | {node.domain for node in model.graph.node if node.domain not in {"", "ai.onnx"}}
        ),
        "nested_graph_attributes": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node for attr in node.attribute
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "conv_bias_findings": AUDITOR.conv_bias_findings(model),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_checker=False, full_checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
    try:
        row["runtime_shape_trace"] = AUDITOR.runtime_shape_trace(343, model)
    except Exception as exc:  # noqa: BLE001
        row["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
    with tempfile.TemporaryDirectory(prefix="task343_exact169_") as work:
        row["actual_profile"] = scoring.score_and_verify(
            copy.deepcopy(model), 343, work, path.stem, require_correct=False
        )
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    exact_path = CONTROLS / "exact178.onnx"
    authority_path = CONTROLS / "authority173.onnx"
    bad_a_path = CONTROLS / "bad172_classifier_a.onnx"
    bad_b_path = CONTROLS / "bad172_classifier_b.onnx"
    exact = onnx.load(exact_path)
    output: dict[str, Any] = {
        "task": 343,
        "generator": "inputs/arc-gen-repo/tasks/task_d8c310e9.py",
        "controls": {
            "authority173": structural(authority_path),
            "bad172_classifier_a": structural(bad_a_path),
            "bad172_classifier_b": structural(bad_b_path),
            "exact178": structural(exact_path),
        },
    }
    output["exact178_known_four"] = known_four(exact)
    (HERE / "control_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    output["exact178_fresh_four"] = fresh_exact_four(exact)
    output["reproduced_counterexamples"] = {
        "authority173": counterexample_run(onnx.load(authority_path), 343_799_445),
        "bad172_classifier_a": counterexample_run(onnx.load(bad_a_path), 90_343_001),
        "bad172_classifier_b": counterexample_run(onnx.load(bad_b_path), 343_799_445),
    }
    (HERE / "control_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "exact_cost": output["controls"]["exact178"]["actual_profile"]["cost"],
                "known": {
                    name: (row["right"], row["wrong"], row["errors"])
                    for name, row in output["exact178_known_four"].items()
                },
                "fresh": [
                    {
                        "seed": run["seed"],
                        "modes": {
                            name: (row["right"], row["wrong"], row["errors"])
                            for name, row in run["modes"].items()
                        },
                    }
                    for run in output["exact178_fresh_four"]
                ],
                "counterexamples": output["reproduced_counterexamples"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
