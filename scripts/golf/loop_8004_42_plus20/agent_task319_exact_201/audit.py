#!/usr/bin/env python3
"""Fail-closed audit for task319 exact-regolf probes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK = 319
AUTHORITY_COST = 1003
FRESH_SEED = 319_201_001
FRESH_COUNT = 2_000
FOUR_CONFIGS = (
    (True, 1, "disable_all_t1"),
    (True, 4, "disable_all_t4"),
    (False, 1, "default_t1"),
    (False, 4, "default_t4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import known_score  # noqa: E402
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRICT = load_module(
    "task319_201_strict_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_A_115/run_exact_audit.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def converted_known() -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    for examples in scoring.load_examples(TASK).values():
        for example in examples:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def converted_fresh() -> tuple[list[dict[str, np.ndarray]], int, int]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[str(TASK)]}")
    random.seed(FRESH_SEED)
    np.random.seed(FRESH_SEED & 0xFFFFFFFF)
    rows: list[dict[str, np.ndarray]] = []
    attempts = generation_errors = 0
    while len(rows) < FRESH_COUNT:
        attempts += 1
        try:
            converted = scoring.convert_to_numpy(generator.generate())
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if converted is not None:
            rows.append(converted)
    return rows, attempts, generation_errors


def run_three(
    rows: list[dict[str, np.ndarray]], sessions: dict[str, ort.InferenceSession]
) -> dict[str, Any]:
    stats = {
        label: {
            "right": 0, "wrong": 0, "errors": 0,
            "nonfinite_output_values": 0, "first_failure": None,
        }
        for label in sessions
    }
    raw_equal = {
        label: {"equal": 0, "different": 0, "first_difference": None}
        for label in sessions if label != "authority"
    }
    for case, benchmark in enumerate(rows):
        raw: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            item = stats[label]
            try:
                output = np.asarray(session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0])
                raw[label] = output
                if output.dtype.kind in "fc":
                    item["nonfinite_output_values"] += int(np.count_nonzero(~np.isfinite(output)))
                if np.array_equal(output > 0, benchmark["output"] > 0):
                    item["right"] += 1
                else:
                    item["wrong"] += 1
                    if item["first_failure"] is None:
                        item["first_failure"] = {
                            "case": case,
                            "different_cells": int(np.count_nonzero((output > 0) != (benchmark["output"] > 0))),
                        }
            except Exception as exc:  # noqa: BLE001
                item["errors"] += 1
                if item["first_failure"] is None:
                    item["first_failure"] = {
                        "case": case,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        if "authority" not in raw:
            continue
        for label, item in raw_equal.items():
            if label not in raw:
                continue
            if np.array_equal(raw[label], raw["authority"]):
                item["equal"] += 1
            else:
                item["different"] += 1
                if item["first_difference"] is None:
                    delta = raw[label].astype(np.int64) - raw["authority"].astype(np.int64)
                    item["first_difference"] = {
                        "case": case,
                        "different_values": int(np.count_nonzero(delta)),
                        "max_abs_delta": int(np.max(np.abs(delta))),
                    }
    return {"models": stats, "raw_equal_to_authority": raw_equal}


def four_config_audit(
    payloads: dict[str, bytes], known: list[dict[str, np.ndarray]],
    fresh: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for disabled, threads, key in FOUR_CONFIGS:
        sessions: dict[str, ort.InferenceSession] = {}
        session_errors: dict[str, str] = {}
        for label, data in payloads.items():
            try:
                sessions[label] = make_session(data, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                session_errors[label] = f"{type(exc).__name__}: {exc}"
        result[key] = {
            "session_errors": session_errors,
            "known": run_three(known, sessions),
            "fresh": run_three(fresh, sessions),
        }
    return result


def all_intermediate_trace(
    data: bytes, rows: list[dict[str, np.ndarray]], disable_all: bool,
) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    expected = {
        item.name: [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
        for item in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for item in traced.graph.node:
        for output in item.output:
            if output and output in typed and output not in names:
                traced.graph.output.append(copy.deepcopy(typed[output]))
                names.append(output)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    mismatch_map: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = {}
    nonfinite = runtime_errors = 0
    max_sum_bytes = 0
    for benchmark in rows:
        try:
            outputs = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
        except Exception:  # noqa: BLE001
            runtime_errors += 1
            continue
        max_sum_bytes = max(max_sum_bytes, sum(np.asarray(item).nbytes for item in outputs))
        for name, output in zip(names, outputs, strict=True):
            array = np.asarray(output)
            if array.dtype.kind in "fc":
                nonfinite += int(np.count_nonzero(~np.isfinite(array)))
            actual = tuple(int(size) for size in array.shape)
            wanted = tuple(expected[name])
            if actual != wanted:
                key = (name, wanted, actual)
                mismatch_map[key] = mismatch_map.get(key, 0) + 1
    mismatches = [
        {"tensor": name, "declared": list(wanted), "runtime": list(actual), "cases": count}
        for (name, wanted, actual), count in sorted(mismatch_map.items())
    ]
    return {
        "cases": len(rows),
        "runtime_tensors": len(names),
        "runtime_errors": runtime_errors,
        "nonfinite_values": nonfinite,
        "max_sum_tensor_bytes": max_sum_bytes,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches and runtime_errors == 0,
    }


def main() -> None:
    manifest = json.loads((HERE / "build.json").read_text())
    rows: list[dict[str, Any]] = []
    payload_by_label: dict[str, bytes] = {}
    for source in manifest["candidates"]:
        path = ROOT / source["path"]
        data = path.read_bytes()
        if digest(data) != source["sha256"]:
            raise RuntimeError(f"hash drift: {path}")
        payload_by_label[source["label"]] = data
        row = dict(source)
        row["structure"] = STRICT.structure(data)
        try:
            with tempfile.TemporaryDirectory(prefix=f"task319_201_{source['label']}_") as workdir:
                row["official_profile"] = scoring.score_and_verify(
                    onnx.load_model_from_string(data), TASK, workdir,
                    label=f"task319_201_{source['label']}", require_correct=False,
                )
        except Exception as exc:  # noqa: BLE001
            row["official_profile"] = None
            row["official_profile_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    labels = (
        "authority",
        "combined_runnable",
        "combined_runnable_honest_metadata",
        "truthful_free_input",
    )
    payloads = {label: payload_by_label[label] for label in labels}
    known = converted_known()
    fresh, attempts, generation_errors = converted_fresh()
    four = four_config_audit(payloads, known, fresh)

    traces: dict[str, Any] = {}
    trace_rows = known[:2] + fresh[:30]
    for label in labels:
        traces[label] = {
            "disable_all": all_intermediate_trace(payloads[label], trace_rows, True),
            "default": all_intermediate_trace(payloads[label], trace_rows, False),
        }

    strict_lower = [
        row for row in rows
        if row.get("official_profile")
        and int(row["official_profile"]["cost"]) < AUTHORITY_COST
        and bool(row["official_profile"].get("correct"))
    ]
    accepted = []
    for row in strict_lower:
        label = row["label"]
        if label not in traces:
            continue
        if not all(item["truthful"] for item in traces[label].values()):
            continue
        if not row["structure"]["passed"]:
            continue
        accepted.append(label)

    report = {
        "task": TASK,
        "authority_cost": AUTHORITY_COST,
        "fresh_seed": FRESH_SEED,
        "fresh_count": FRESH_COUNT,
        "fresh_attempts": attempts,
        "fresh_generation_errors": generation_errors,
        "candidate_rows": rows,
        "four_config": four,
        "all_intermediate_trace": traces,
        "strict_lower_known_correct_labels": [row["label"] for row in strict_lower],
        "accepted": accepted,
        "decision": "NO_TRUTHFUL_STRICT_LOWER_CANDIDATE" if not accepted else "ACCEPT",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "profiles": {
            row["label"]: row.get("official_profile") for row in rows
        },
        "strict_lower_known_correct": report["strict_lower_known_correct_labels"],
        "trace_truthful": {
            label: {mode: item["truthful"] for mode, item in modes.items()}
            for label, modes in traces.items()
        },
        "accepted": accepted,
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
