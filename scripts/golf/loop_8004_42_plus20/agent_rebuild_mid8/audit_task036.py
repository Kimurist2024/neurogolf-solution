#!/usr/bin/env python3
"""Independent no-promotion audit of the truthful task036 rebuild."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (36, 208, 255, 44)
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
SOURCE_CANDIDATE = (
    ROOT / "scripts/golf/loop_7999_13/lane_b15/candidate_task036_truthful_gather.onnx"
)
BASE_DIR = HERE / "baseline"
CANDIDATE_DIR = HERE / "candidates"
CANDIDATE = CANDIDATE_DIR / "task036_truthful_gather.onnx"
FRESH_SEEDS = (260714036, 910714036)
FRESH_PER_SEED = 5000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict[str, Any]:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            (BASE_DIR / f"task{task:03d}.onnx").write_bytes(
                archive.read(f"task{task:03d}.onnx")
            )
    shutil.copyfile(SOURCE_CANDIDATE, CANDIDATE)
    return {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha256(BASE_ZIP),
        "baseline_members": {
            str(task): {
                "path": str((BASE_DIR / f"task{task:03d}.onnx").relative_to(ROOT)),
                "sha256": sha256(BASE_DIR / f"task{task:03d}.onnx"),
            }
            for task in TASKS
        },
        "candidate": {
            "source": str(SOURCE_CANDIDATE.relative_to(ROOT)),
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
        },
    }


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker_full = True
    except Exception as exc:  # noqa: BLE001
        checker_full = False
        errors.append(f"checker: {type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict_data_prop = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict_data_prop = False
        errors.append(f"shape: {type(exc).__name__}: {exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    nonstatic = [
        value.name
        for value in values
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in value.type.tensor_type.shape.dim)
    ]
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    giant_initializers = [
        {"name": item.name, "elements": int(np.prod(numpy_helper.to_array(item).shape, dtype=np.int64))}
        for item in model.graph.initializer
        if numpy_helper.to_array(item).size > 100_000
    ]
    checks = {
        "checker_full": checker_full,
        "strict_data_prop": strict_data_prop,
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_functions_sparse_nested": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attribute in node.attribute
            )
        ),
        "no_banned_ops": all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        # Campaign-wide definition used by the existing exhaustive miner.
        "no_giant_einsum_max16": max_einsum_inputs <= 16,
        "no_giant_initializer": not giant_initializers,
        "static_positive_shapes": not nonstatic,
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "nonstatic": nonstatic,
        "max_einsum_inputs": max_einsum_inputs,
        "giant_initializers": giant_initializers,
        "ops": dict(Counter(node.op_type for node in model.graph.node)),
    }


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_dual(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    result: dict[str, Any] = {}
    for mode in ("disabled", "default"):
        row: dict[str, Any] = {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        try:
            session = make_session(model, mode)
        except Exception as exc:  # noqa: BLE001
            row["runtime_errors"] = 1
            row["first_failure"] = {"session": f"{type(exc).__name__}: {exc}"}
            result[mode] = row
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = session.run(["output"], {"input": benchmark["input"]})[0]
                    if np.array_equal(raw > 0, benchmark["output"].astype(bool)):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "subset": subset,
                            "index": index,
                        }
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "subset": subset,
                        "index": index,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
        result[mode] = row
    return result


def trace_runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    example = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert example is not None
    arrays = session.run(names, {"input": example["input"]})
    mismatches = [
        {"tensor": name, "declared": dims(typed[name]), "runtime": list(np.asarray(array).shape)}
        for name, array in zip(names, arrays)
        if dims(typed[name]) != list(np.asarray(array).shape)
    ]
    graph_outputs = {value.name for value in model.graph.output}
    truthful_bytes = sum(
        np.asarray(array).nbytes
        for name, array in zip(names, arrays)
        if name not in graph_outputs
    )
    return {
        "trace_outputs": len(names),
        "declared_runtime_mismatches": mismatches,
        "shape_cloak": bool(mismatches),
        "truthful_one_example_intermediate_bytes": int(truthful_bytes),
    }


def actual_score(model: onnx.ModelProto, task: int) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"mid8_{task:03d}_", dir="/tmp") as workdir:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=f"mid8_task{task:03d}", require_correct=False
        )


def fresh_seed(model: onnx.ModelProto, seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module("task_1f85a75f")
    sessions = {mode: make_session(model, mode) for mode in ("disabled", "default")}
    rows = {
        mode: {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        for mode in sessions
    }
    valid = attempts = generation_errors = conversion_skips = 0
    started = time.monotonic()
    while valid < FRESH_PER_SEED:
        attempts += 1
        try:
            case = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        benchmark = scoring.convert_to_numpy(case)
        if benchmark is None:
            conversion_skips += 1
            continue
        valid += 1
        for mode, session in sessions.items():
            row = rows[mode]
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                got = raw > 0
                want = benchmark["output"].astype(bool)
                if np.array_equal(got, want):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "valid_case": valid,
                        "got_shape": list(got.shape),
                        "want_shape": list(want.shape),
                        "different_cells": int(np.count_nonzero(got != want)) if got.shape == want.shape else None,
                    }
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "valid_case": valid,
                    "runtime": f"{type(exc).__name__}: {exc}",
                }
    for row in rows.values():
        row["accuracy"] = row["right"] / valid
    return {
        "seed": seed,
        "requested": FRESH_PER_SEED,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "modes": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    prepared = prepare()
    model = onnx.load(CANDIDATE)
    current_costs = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json").read_text()
    )["costs"]
    score = actual_score(model, 36)
    known = known_dual(model, 36)
    fresh = [fresh_seed(model, seed) for seed in FRESH_SEEDS]
    candidate_cost = score["cost"] if score else None
    base_cost = int(current_costs["36"])
    gain = math.log(base_cost / candidate_cost) if candidate_cost else None
    all_known = all(
        row["right"] == 265 and row["wrong"] == 0 and row["runtime_errors"] == 0
        for row in known.values()
    )
    all_fresh = all(
        row["right"] == FRESH_PER_SEED and row["wrong"] == 0 and row["runtime_errors"] == 0
        for seed_row in fresh
        for row in seed_row["modes"].values()
    )
    structure = structural(model)
    runtime_shapes = trace_runtime_shapes(model, 36)
    accepted = bool(
        score
        and score.get("correct")
        and candidate_cost < base_cost
        and structure["pass"]
        and not runtime_shapes["shape_cloak"]
        and all_known
        and all_fresh
    )
    result = {
        "complete": True,
        "prepared": prepared,
        "task": 36,
        "task_hash": "1f85a75f",
        "rule": (
            "Select the generator's connected special-color object (3..5 by 3..5 support, "
            "protected by a one-cell moat from random background pixels) and return its tight crop."
        ),
        "private_zero_lineage": False,
        "current_cost": base_cost,
        "candidate_score": score,
        "candidate_cost": candidate_cost,
        "projected_score_gain": gain,
        "structure": structure,
        "runtime_shapes": runtime_shapes,
        "known_dual": known,
        "fresh_independent_seeds_dual": fresh,
        "known_complete": all_known,
        "fresh_all_100_percent": all_fresh,
        "accepted": accepted,
    }
    (HERE / "task036_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accepted": accepted,
                "current_cost": base_cost,
                "candidate_cost": candidate_cost,
                "gain": gain,
                "known_complete": all_known,
                "fresh_all_100_percent": all_fresh,
                "structure": structure["pass"],
                "shape_cloak": runtime_shapes["shape_cloak"],
            },
            indent=2,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
