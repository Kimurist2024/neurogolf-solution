#!/usr/bin/env python3
"""Independent, non-promoting audit of root_sweep29 latent-prune candidates.

The source archive and candidate files are read only.  This script writes all
evidence next to itself and never invokes try_candidate or edits a submission.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
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
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent"
TASKS = (10, 28, 60, 175, 229, 232, 304, 315)
VARIANTS = {10: 3, 28: 3, 60: 2, 175: 8, 229: 4, 232: 4, 304: 6, 315: 2}
TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
# None of this wave's tasks appears in docs/golf/private_zero_tasks.md's
# operational 51-task high-risk catalog.  Keep the policy branch explicit.
PRIVATE_ZERO: set[int] = set()
FRESH_SEEDS = (30_010_071, 30_010_072)
FRESH_COUNT = 5000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {
    "TFIDFVECTORIZER",
    "SCATTERELEMENTS",
    "SCATTERND",
    "GATHERND",
    "HARDMAX",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def session(model: onnx.ModelProto, disable: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def examples(task: int) -> list[dict[str, np.ndarray]]:
    return [
        item
        for split in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(split, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]


def known_dual(task: int, model: onnx.ModelProto) -> dict[str, Any]:
    cases = examples(task)
    result: dict[str, Any] = {}
    for label, disable in (("disable_all", True), ("default", False)):
        row: dict[str, Any] = {
            "total": len(cases),
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "first_failure": None,
            "unstable_examples": 0,
            "min_positive": None,
        }
        try:
            runtime = session(model, disable)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = 1
            row["perfect"] = False
            result[label] = row
            continue
        positives: list[float] = []
        for index, case in enumerate(cases, start=1):
            try:
                raw = runtime.run(None, {runtime.get_inputs()[0].name: case["input"]})[0]
                correct = np.array_equal(raw > 0, case["output"].astype(bool))
                row["right" if correct else "wrong"] += 1
                if not correct and row["first_failure"] is None:
                    row["first_failure"] = {"case": index}
                pos = raw[raw > 0]
                if pos.size:
                    positives.append(float(pos.min()))
                row["unstable_examples"] += int(bool(np.any((raw > 0) & (raw < 0.25))))
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
        row["min_positive"] = min(positives) if positives else None
        row["perfect"] = (
            row["right"] == row["total"]
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
            and row["unstable_examples"] == 0
        )
        result[label] = row
    return result


def runtime_shapes(task: int, model: onnx.ModelProto) -> dict[str, Any]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(sanitized), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(sanitized)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    runtime = session(traced, True)
    case = examples(task)[0]
    arrays = runtime.run(names, {runtime.get_inputs()[0].name: case["input"]})
    mismatches = [
        {"tensor": name, "declared": shape(typed[name]), "runtime": list(array.shape)}
        for name, array in zip(names, arrays)
        if shape(typed[name]) != list(array.shape)
    ]
    return {
        "all_node_outputs_traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def structure(task: int, model: onnx.ModelProto) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    static = all(shape(value) is not None for value in values if value.type.HasField("tensor_type"))
    try:
        truth = runtime_shapes(task, model)
    except Exception as exc:  # noqa: BLE001
        truth = {"pass": False, "error": f"{type(exc).__name__}:{exc}"}
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    arrays = [numpy_helper.to_array(init) for init in model.graph.initializer]
    domains = {item.domain for item in model.opset_import} | {node.domain for node in model.graph.node}
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive_shapes": static,
        "truthful_all_runtime_shapes": bool(truth.get("pass")),
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and shape(model.graph.input[0]) == [1, 10, 30, 30]
            and shape(model.graph.output[0]) == [1, 10, 30, 30]
        ),
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "no_banned": all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "no_nested_functions_sparse": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attr in node.attribute
            )
        ),
        "no_lookup_or_cloak_ops": all(node.op_type.upper() not in LOOKUP for node in model.graph.node),
        "no_giant_einsum": max_einsum <= 16,
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(array.dtype.kind not in "fc" or bool(np.isfinite(array).all()) for array in arrays),
    }
    return {
        "nodes": len(model.graph.node),
        "ops": dict(sorted(ops.items())),
        "max_einsum_inputs": max_einsum,
        "runtime_shape_evidence": truth,
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
    }


def fresh_seed(task: int, model: onnx.ModelProto, seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{TASK_MAP[f'{task:03d}']}")
    sessions = {"disable_all": session(model, True), "default": session(model, False)}
    rows = {
        label: {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        for label in sessions
    }
    valid = attempts = generation_errors = 0
    started = time.monotonic()
    while valid < FRESH_COUNT:
        attempts += 1
        try:
            raw_case = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        case = scoring.convert_to_numpy(raw_case)
        if case is None:
            continue
        valid += 1
        for label, runtime in sessions.items():
            row = rows[label]
            try:
                raw = runtime.run(None, {runtime.get_inputs()[0].name: case["input"]})[0]
                correct = np.array_equal(raw > 0, case["output"].astype(bool))
                row["right" if correct else "wrong"] += 1
                if not correct and row["first_failure"] is None:
                    row["first_failure"] = {"case": valid}
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": valid,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
    for row in rows.values():
        row["accuracy"] = row["right"] / valid
    return {
        "seed": seed,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "modes": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def measured_cost(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def main() -> int:
    ort.set_default_logger_severity(4)
    base_sha = digest(BASE_ZIP.read_bytes())
    baseline: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(BASE_ZIP) as archive, tempfile.TemporaryDirectory(prefix="prune30a_base_") as tmp:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            path = Path(tmp) / f"task{task:03d}.onnx"
            path.write_bytes(data)
            baseline[task] = {
                "sha256": digest(data),
                "cost": measured_cost(path),
            }

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        candidates: list[tuple[Path, dict[str, int]]] = []
        for variant in range(1, VARIANTS[task] + 1):
            path = SOURCE / f"task{task:03d}_r{variant:03d}.onnx"
            candidates.append((path, measured_cost(path)))
        candidates.sort(key=lambda item: (item[1]["cost"], item[0].name))
        for order, (path, cost) in enumerate(candidates, start=1):
            data = path.read_bytes()
            model = onnx.load_from_string(data)
            static = structure(task, model)
            known = known_dual(task, model)
            known_pass = all(mode.get("perfect") for mode in known.values())
            lower = 0 < cost["cost"] < baseline[task]["cost"]["cost"]
            pre_fresh_pass = static["pass"] and known_pass and lower
            reasons: list[str] = []
            if not lower:
                reasons.append("actual_cost_not_strictly_lower")
            if not static["pass"]:
                reasons.extend(name for name, passed in static["checks"].items() if not passed)
            if not known_pass:
                reasons.append("known_dual_not_100pct_or_margin_stable")
            row: dict[str, Any] = {
                "task": task,
                "screen_order_within_task": order,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(data),
                "private_zero_classification": task in PRIVATE_ZERO,
                "baseline": baseline[task],
                "actual_cost": cost,
                "gain_if_accepted": (
                    math.log(baseline[task]["cost"]["cost"] / cost["cost"]) if lower else 0.0
                ),
                "structure": static,
                "known_dual": known,
                "known_pass": known_pass,
                "pre_fresh_pass": pre_fresh_pass,
                "pre_fresh_reasons": sorted(set(reasons)),
                "fresh": [],
                "accepted": False,
            }
            if pre_fresh_pass:
                row["fresh"] = [fresh_seed(task, model, seed) for seed in FRESH_SEEDS]
                threshold = 1.0 if task in PRIVATE_ZERO else 0.90
                row["fresh_threshold"] = threshold
                row["accepted"] = all(
                    mode["runtime_errors"] == 0 and mode["accuracy"] >= threshold
                    for seed_row in row["fresh"]
                    for mode in seed_row["modes"].values()
                )
                if not row["accepted"]:
                    row["pre_fresh_reasons"].append("fresh_dual_below_threshold_or_runtime_error")
            rows.append(row)
            print(
                f"task{task:03d} {path.name} order={order} cost={cost['cost']}/"
                f"{baseline[task]['cost']['cost']} known="
                f"{known['disable_all']['right']}/{known['disable_all']['total']}|"
                f"{known['default']['right']}/{known['default']['total']} "
                f"struct={static['pass']} accepted={row['accepted']}",
                flush=True,
            )
            (HERE / "audit_partial.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    accepted = [row for row in rows if row["accepted"]]
    result = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": base_sha,
        "tasks": list(TASKS),
        "policy": {
            "known_dual": 1.0,
            "fresh_non_private": 0.90,
            "fresh_private_zero": 1.0,
            "fresh_seeds": list(FRESH_SEEDS),
            "fresh_count_per_seed": FRESH_COUNT,
            "runtime_errors": 0,
            "strict_data_prop": True,
            "truthful_all_runtime_shapes": True,
            "standard_domains": True,
            "lookup_cloak_giant": False,
            "conv_bias_ub": 0,
        },
        "baseline": {str(task): value for task, value in baseline.items()},
        "rows": rows,
        "accepted": accepted,
        "accepted_count": len(accepted),
        "aggregate_gain": sum(row["gain_if_accepted"] for row in accepted),
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
