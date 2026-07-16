#!/usr/bin/env python3
"""Strict task226 proof and runtime audit for the six-probe rebuild."""

from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "golf"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))

from lib import scoring  # noqa: E402
import check_conv_bias  # noqa: E402


TASK = 226
CANDIDATE = HERE / "task226_sixbit.onnx"
BASE_ZIPS = [ROOT / "submission_base_8004.50.zip", ROOT / "submission_base_8005.16.zip"]
OUT = HERE / "task226_strict_audit.json"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
FRESH_SEEDS = [22_650_001, 22_650_002]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected the model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def run_one(session: ort.InferenceSession, example: dict[str, object]) -> tuple[bool, np.ndarray]:
    converted = scoring.convert_to_numpy(example)
    if converted is None:
        raise RuntimeError("example does not fit the official 30x30 carrier")
    raw = session.run(["output"], {"input": converted["input"]})[0]
    return bool(np.array_equal(raw > 0, converted["output"] > 0)), raw


def static_gate(model: onnx.ModelProto) -> tuple[dict[str, object], onnx.ModelProto]:
    row: dict[str, object] = {}
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    row["checker_full"] = "PASS"
    row["strict_shape_inference_data_prop"] = "PASS"
    row["standard_domains"] = all(op.domain in {"", "ai.onnx"} for op in model.opset_import)
    row["functions"] = len(model.functions)
    row["banned_ops"] = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
    ]
    row["nested_graphs"] = [
        node.op_type
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    row["conv_bias_ub"] = [list(item) for item in check_conv_bias.check_model(model)]
    bad_shapes = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(
        inferred.graph.output
    ):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or any(
            not dim.HasField("dim_value")
            or dim.HasField("dim_param")
            or dim.dim_value <= 0
            for dim in tensor_type.shape.dim
        ):
            bad_shapes.append(value.name)
    row["static_shape_failures"] = bad_shapes
    row["pass"] = bool(
        row["standard_domains"]
        and not row["functions"]
        and not row["banned_ops"]
        and not row["nested_graphs"]
        and not row["conv_bias_ub"]
        and not bad_shapes
    )
    return row, inferred


def runtime_shape_gate(
    inferred: onnx.ModelProto, example: dict[str, object], disable_all: bool
) -> dict[str, object]:
    shape_by_name: dict[str, tuple[int, ...]] = {}
    type_by_name: dict[str, onnx.ValueInfoProto] = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        shape_by_name[value.name] = tuple(
            int(dim.dim_value) for dim in value.type.tensor_type.shape.dim
        )
        type_by_name[value.name] = value

    exposed = copy.deepcopy(inferred)
    names = []
    seen = set()
    for node in exposed.graph.node:
        for name in node.output:
            if name and name not in seen and name in type_by_name:
                names.append(name)
                seen.add(name)
    del exposed.graph.output[:]
    exposed.graph.output.extend(copy.deepcopy(type_by_name[name]) for name in names)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(exposed.SerializeToString(), options)
    converted = scoring.convert_to_numpy(example)
    assert converted is not None
    values = session.run(names, {"input": converted["input"]})
    mismatches = [
        {"name": name, "inferred": list(shape_by_name[name]), "runtime": list(value.shape)}
        for name, value in zip(names, values)
        if tuple(value.shape) != shape_by_name[name]
    ]
    return {
        "mode": "disable_all" if disable_all else "default",
        "outputs_exposed": len(names),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    candidate_bytes = CANDIDATE.read_bytes()
    candidate = onnx.load_model_from_string(candidate_bytes)
    structural, inferred = static_gate(candidate)

    baselines = []
    baseline_model = None
    for path in BASE_ZIPS:
        with zipfile.ZipFile(path) as archive:
            data = archive.read("task226.onnx")
        baselines.append({"zip": path.name, "member_sha256": sha256(data), "bytes": len(data)})
        baseline_model = onnx.load_model_from_string(data)
    assert baseline_model is not None

    generator = importlib.import_module("task_941d9a10")
    candidate_sessions = {
        "disable_all": make_session(candidate, True),
        "default": make_session(candidate, False),
    }
    baseline_sessions = {
        "disable_all": make_session(baseline_model, True),
        "default": make_session(baseline_model, False),
    }

    known = []
    examples = scoring.load_examples(TASK)
    for subset in ("train", "test", "arc-gen"):
        known.extend(examples.get(subset, []))
    known_rows: dict[str, object] = {}
    min_positive = None
    near_margin = 0
    for mode, session in candidate_sessions.items():
        right = errors = 0
        for example in known:
            try:
                ok, raw = run_one(session, example)
                right += int(ok)
                positive = raw[raw > 0]
                if positive.size:
                    value = float(positive.min())
                    min_positive = value if min_positive is None else min(min_positive, value)
                    near_margin += int(np.count_nonzero(positive < 0.25))
            except Exception:  # noqa: BLE001
                errors += 1
        known_rows[mode] = {
            "first100_right": min(right, 100) if len(known) == 100 else None,
            "all_total": len(known),
            "all_right": right,
            "runtime_errors": errors,
        }
    # Re-run the explicitly requested first-100 slice; the all-right summary
    # above must not be used as a proxy when a later known case fails.
    for mode, session in candidate_sessions.items():
        first100_right = first100_errors = 0
        for example in known[:100]:
            try:
                ok, _ = run_one(session, example)
                first100_right += int(ok)
            except Exception:  # noqa: BLE001
                first100_errors += 1
        known_rows[mode]["first100_right"] = first100_right
        known_rows[mode]["first100_total"] = min(100, len(known))
        known_rows[mode]["first100_runtime_errors"] = first100_errors

    # Exhaust all 17 valid widths and all eight valid heights.  This is the
    # complete generator domain, not a sample, and therefore proves private
    # correctness for the task226 generator.
    wides = []
    for length in (3, 5):
        for values in itertools.product(range(1, 5), repeat=length):
            if sum(values) + length - 1 == 10:
                wides.append(values)
    talls = []
    for length in (3, 5):
        for values in itertools.product(range(1, 4), repeat=length):
            if sum(values) + length - 1 == 10:
                talls.append(values)
    exhaustive: dict[str, object] = {
        "valid_wides": len(wides),
        "valid_talls": len(talls),
        "total": len(wides) * len(talls),
    }
    for mode in candidate_sessions:
        right = raw_equal = errors = 0
        for wide in wides:
            for tall in talls:
                example = generator.generate(wides=list(wide), talls=list(tall))
                try:
                    ok, candidate_raw = run_one(candidate_sessions[mode], example)
                    _, baseline_raw = run_one(baseline_sessions[mode], example)
                    right += int(ok)
                    raw_equal += int(np.array_equal(candidate_raw, baseline_raw, equal_nan=True))
                except Exception:  # noqa: BLE001
                    errors += 1
        exhaustive[mode] = {
            "right": right,
            "raw_equal_to_baseline": raw_equal,
            "runtime_errors": errors,
        }

    fresh: dict[str, object] = {}
    for seed in FRESH_SEEDS:
        random.seed(seed)
        rows = {mode: {"right": 0, "wrong": 0, "runtime_errors": 0} for mode in candidate_sessions}
        generation_errors = 0
        for _ in range(5000):
            try:
                example = generator.generate()
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            for mode, session in candidate_sessions.items():
                try:
                    ok, _ = run_one(session, example)
                    rows[mode]["right" if ok else "wrong"] += 1
                except Exception:  # noqa: BLE001
                    rows[mode]["runtime_errors"] += 1
        fresh[str(seed)] = {"generated": 5000 - generation_errors, "generation_errors": generation_errors, **rows}

    witness = generator.generate(wides=list(wides[0]), talls=list(talls[0]))
    runtime_shapes = [
        runtime_shape_gate(inferred, witness, True),
        runtime_shape_gate(inferred, witness, False),
    ]
    result = {
        "task": TASK,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": sha256(candidate_bytes),
        "baselines": baselines,
        "rebase_compatible": len({row["member_sha256"] for row in baselines}) == 1,
        "structural": structural,
        "runtime_shapes": runtime_shapes,
        "known": known_rows,
        "exhaustive_generator_domain": exhaustive,
        "fresh_5000_x_2_seeds": fresh,
        "margin": {"min_positive": min_positive, "cells_in_open_0_025": near_margin},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
