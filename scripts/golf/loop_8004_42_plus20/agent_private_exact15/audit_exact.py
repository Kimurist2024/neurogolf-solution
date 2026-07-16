#!/usr/bin/env python3
"""Fail-closed audit of exact micro-rewrites in private-zero lineages.

The only admissible rewrite here is removal of an internal Identity or a
duplicate deterministic producer.  Candidates are compared to the exact
task member in the LB-white 8004.50 archive.  The script never promotes or
modifies a submission archive.
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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE_ZIP = ROOT / "submission_base_8004.50.zip"
CANDIDATE_DIR = HERE / "candidates"
TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
CURRENT_COST_ROWS = json.loads(
    (ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json").read_text()
)["ranked"]
CURRENT_COST = {int(row["task"]): row for row in CURRENT_COST_ROWS}
TARGETS = {
    77: [("identity", 353)],
    102: [("identity", 0), ("identity", 3)],
    169: [("duplicate", 12, 11)],
    187: [("identity", 7)],
    191: [("identity", 0)],
    216: [("identity", 11)],
    285: [("identity", 204)],
    366: [("identity", 81)],
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def prune_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def build_candidate(base: onnx.ModelProto, rewrites: list[tuple]) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    model = copy.deepcopy(base)
    details: list[dict[str, Any]] = []
    # Indices refer to the original model, so process in descending order.
    for spec in sorted(rewrites, key=lambda item: item[1], reverse=True):
        kind = spec[0]
        if kind == "identity":
            index = spec[1]
            node = model.graph.node[index]
            if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
                raise RuntimeError(f"identity precondition failed at {index}")
            old, new = node.output[0], node.input[0]
            replace_uses(model, old, new)
            del model.graph.node[index]
            removed = prune_initializers(model)
            remove_value_info(model, {old, *removed})
            details.append({
                "kind": kind, "index": index, "input": new, "output": old,
                "removed_initializers": removed,
                "proof": "Identity(x)=x for every tensor value and dtype; all consumers are rewired from the Identity output to the same input tensor.",
            })
        elif kind == "duplicate":
            duplicate_index, canonical_index = spec[1], spec[2]
            duplicate = model.graph.node[duplicate_index]
            canonical = model.graph.node[canonical_index]
            left = copy.deepcopy(duplicate)
            right = copy.deepcopy(canonical)
            left.name = right.name = ""
            del left.output[:]
            del right.output[:]
            if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True):
                raise RuntimeError("duplicate producer precondition failed")
            if len(duplicate.output) != len(canonical.output):
                raise RuntimeError("duplicate arity mismatch")
            pairs = list(zip(duplicate.output, canonical.output))
            for old, new in pairs:
                replace_uses(model, old, new)
            del model.graph.node[duplicate_index]
            removed = prune_initializers(model)
            remove_value_info(model, {old for old, _ in pairs} | set(removed))
            details.append({
                "kind": kind, "duplicate_index": duplicate_index,
                "canonical_index": canonical_index, "replacements": dict(pairs),
                "removed_initializers": removed,
                "proof": "The two deterministic nodes have byte-identical op type, domain, inputs and attributes; therefore their outputs are identical for every input and the duplicate consumers can use the canonical output.",
            })
        else:
            raise RuntimeError(kind)
    return model, details


def static_cost(model: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, int]:
    infos = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    excluded = {value.name for value in inferred.graph.input}
    excluded.update(value.name for value in inferred.graph.output)
    excluded.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in excluded or name in seen:
                continue
            seen.add(name)
            value = infos.get(name)
            dims = shape(value) if value is not None else None
            if dims is None:
                raise RuntimeError(f"non-static output {name}")
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    for node in inferred.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_floats":
                params += len(attr.floats)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    error = None
    inferred = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checks["checker_full"] = True
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        checks["strict_data_prop"] = True
    except Exception as exc:
        checks["checker_full"] = False
        checks["strict_data_prop"] = False
        error = f"{type(exc).__name__}: {exc}"
    inspected = inferred if inferred is not None else model
    values = list(inspected.graph.input) + list(inspected.graph.value_info) + list(inspected.graph.output)
    checks["canonical_io"] = (
        len(model.graph.input) == len(model.graph.output) == 1
        and model.graph.input[0].name == "input" and model.graph.output[0].name == "output"
    )
    checks["standard_domains"] = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    checks["no_functions_sparse_nested"] = (
        not model.functions and not model.graph.sparse_initializer and all(
            attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute
        )
    )
    checks["no_banned_ops"] = all(
        node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
        for node in model.graph.node
    )
    checks["no_new_giant_einsum"] = True
    checks["static_positive"] = all(shape(value) is not None for value in values)
    checks["no_external_initializers"] = all(
        item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
        for item in model.graph.initializer
    )
    checks["finite_initializers"] = all(
        array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
        for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
    )
    checks["conv_bias_ub0"] = not check_conv_bias(model)
    cost = None
    if inferred is not None:
        try:
            cost = static_cost(model, inferred)
        except Exception as exc:
            error = f"{error}; cost:{type(exc).__name__}:{exc}" if error else f"cost:{type(exc).__name__}:{exc}"
    return {"checks": checks, "pass": all(checks.values()) and error is None and cost is not None, "error": error, "cost": cost}


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disable_all" else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_examples(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    rows = []
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def fresh_examples(task: int, count: int, seed: int) -> tuple[list[dict[str, np.ndarray]], int, int]:
    module = importlib.import_module(f"task_{TASK_MAP[f'{task:03d}']}")
    random.seed(seed)
    rows = []
    errors = 0
    attempts = 0
    while len(rows) < count and attempts < count * 10:
        attempts += 1
        try:
            converted = scoring.convert_to_numpy(module.generate())
            if converted is not None:
                rows.append(converted)
        except Exception:
            errors += 1
    return rows, errors, attempts


def compare(base_sess: ort.InferenceSession, cand_sess: ort.InferenceSession, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(rows), "base_right": 0, "candidate_right": 0,
        "base_errors": 0, "candidate_errors": 0, "one_sided_errors": 0,
        "raw_bitwise_equal": 0, "decoded_equal": 0, "max_abs_raw_difference": 0.0,
    }
    for row in rows:
        expected = row["output"] > 0
        base_raw = cand_raw = None
        try:
            base_raw = base_sess.run(["output"], {"input": row["input"]})[0]
        except Exception:
            result["base_errors"] += 1
        try:
            cand_raw = cand_sess.run(["output"], {"input": row["input"]})[0]
        except Exception:
            result["candidate_errors"] += 1
        if (base_raw is None) != (cand_raw is None):
            result["one_sided_errors"] += 1
        if base_raw is None or cand_raw is None:
            continue
        base_decoded = base_raw > 0
        cand_decoded = cand_raw > 0
        result["base_right"] += int(np.array_equal(base_decoded, expected))
        result["candidate_right"] += int(np.array_equal(cand_decoded, expected))
        result["raw_bitwise_equal"] += int(np.array_equal(base_raw, cand_raw, equal_nan=True))
        result["decoded_equal"] += int(np.array_equal(base_decoded, cand_decoded))
        if base_raw.shape == cand_raw.shape:
            difference = np.abs(
                np.nan_to_num(base_raw).astype(np.float64, copy=False)
                - np.nan_to_num(cand_raw).astype(np.float64, copy=False)
            )
            result["max_abs_raw_difference"] = max(
                result["max_abs_raw_difference"], float(difference.max(initial=0.0))
            )
        else:
            result["max_abs_raw_difference"] = float("inf")
    return result


def trace_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    declared = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options)
    example = known_examples(task)[0]
    outputs = session.run(names, {"input": example["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
    mismatches = [
        {"tensor": name, "declared": declared[name], "runtime": actual[name]}
        for name in declared if name in actual and declared[name] != actual[name]
    ]
    return {"shape_cloak": bool(mismatches), "mismatch_count": len(mismatches), "mismatches": mismatches[:50]}


def actual_score(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"private_exact15_{task}_{label}_", dir="/tmp") as directory:
        return scoring.score_and_verify(copy.deepcopy(model), task, directory, label=label, require_correct=False)


def perfect_comparison(row: dict[str, Any]) -> bool:
    return (
        row["candidate_errors"] == row["one_sided_errors"] == 0
        and row["candidate_right"] == row["total"]
        and row["raw_bitwise_equal"] == row["total"]
        and row["decoded_equal"] == row["total"]
        and row["max_abs_raw_difference"] == 0.0
    )


def audit_task(task: int, rewrites: list[tuple], fresh_count: int) -> dict[str, Any]:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        base_bytes = archive.read(f"task{task:03d}.onnx")
    base = onnx.load_model_from_string(base_bytes)
    candidate, proof = build_candidate(base, rewrites)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    provisional = CANDIDATE_DIR / f"task{task:03d}_provisional.onnx"
    onnx.save(candidate, provisional)
    base_structure = structural(base)
    candidate_structure = structural(candidate)
    report: dict[str, Any] = {
        "task": task,
        "lb_white_lineage": {
            "archive": "submission_base_8004.50.zip",
            "archive_sha256": sha256_bytes(BASELINE_ZIP.read_bytes()),
            "member": f"task{task:03d}.onnx",
            "member_sha256": sha256_bytes(base_bytes),
            "member_byte_length": len(base_bytes),
        },
        "rewrite": proof,
        "base_structure": base_structure,
        "candidate_structure": candidate_structure,
        "candidate_sha256": sha256_bytes(provisional.read_bytes()),
        "candidate_byte_length": provisional.stat().st_size,
        "incumbent_measured_cost": CURRENT_COST[task],
        "candidate_measured_cost": None,
        "candidate_measured_cost_status": "not_run_unless_mandatory_pre_fresh_gates_pass",
        "known": {},
        "fresh": {},
        "fresh_run": False,
        "verdict": "REJECT",
        "reasons": [],
    }
    if base_structure["cost"] and candidate_structure["cost"]:
        old = base_structure["cost"]["cost"]
        new = candidate_structure["cost"]["cost"]
        report["static_cost"] = {"old": old, "new": new, "reduction": old - new, "projected_gain": math.log(old / new) if new > 0 else None}
    else:
        report["static_cost"] = None
    try:
        report["base_runtime_shapes"] = trace_shapes(base, task)
    except Exception as exc:
        report["base_runtime_shapes"] = {"trace_error": f"{type(exc).__name__}: {exc}"}
    try:
        report["candidate_runtime_shapes"] = trace_shapes(candidate, task)
    except Exception as exc:
        report["candidate_runtime_shapes"] = {"trace_error": f"{type(exc).__name__}: {exc}"}

    known = known_examples(task)
    sessions: dict[str, tuple[ort.InferenceSession, ort.InferenceSession]] = {}
    for mode in ("disable_all", "default"):
        try:
            sessions[mode] = (make_session(base, mode), make_session(candidate, mode))
            report["known"][mode] = compare(*sessions[mode], known)
        except Exception as exc:
            report["known"][mode] = {"session_error": f"{type(exc).__name__}: {exc}"}

    known_gate = all(perfect_comparison(row) for row in report["known"].values() if "session_error" not in row) and len(report["known"]) == 2 and all("session_error" not in row for row in report["known"].values())
    structure_gate = bool(candidate_structure["pass"])
    cost_gate = bool(report["static_cost"] and report["static_cost"]["reduction"] > 0)
    truthful_gate = report["candidate_runtime_shapes"].get("shape_cloak") is False
    if not structure_gate:
        report["reasons"].append("candidate_structural_gate_failed")
    if not cost_gate:
        report["reasons"].append("no_strict_cost_reduction")
    if not truthful_gate:
        report["reasons"].append("candidate_runtime_shapes_not_truthful")
    if not known_gate:
        report["reasons"].append("known_dual_not_raw_exact_and_100pct_correct")

    # Private-zero policy: do not spend fresh generation unless all mandatory
    # pre-fresh gates already pass.
    if structure_gate and cost_gate and truthful_gate and known_gate:
        report["fresh_run"] = True
        fresh_gate = True
        for seed in (80_045_000 + task, 90_045_000 + task):
            rows, generation_errors, attempts = fresh_examples(task, fresh_count, seed)
            seed_row = {"seed": seed, "requested": fresh_count, "generated": len(rows), "generation_errors": generation_errors, "attempts": attempts, "modes": {}}
            for mode in ("disable_all", "default"):
                seed_row["modes"][mode] = compare(*sessions[mode], rows)
                fresh_gate &= perfect_comparison(seed_row["modes"][mode])
            report["fresh"][str(seed)] = seed_row
        if fresh_gate:
            measured = actual_score(candidate, task, "candidate")
            report["candidate_measured_cost"] = measured
            report["candidate_measured_cost_status"] = "measured_after_all_correctness_and_shape_gates"
            measured_cost_gate = bool(
                measured is not None
                and measured["cost"] < report["incumbent_measured_cost"]["cost"]
            )
            if not measured_cost_gate:
                report["reasons"].append("candidate_measured_cost_not_strictly_lower")
                if provisional.exists():
                    provisional.unlink()
                return report
            report["measured_projected_gain"] = math.log(
                report["incumbent_measured_cost"]["cost"] / measured["cost"]
            )
            final = CANDIDATE_DIR / f"task{task:03d}_{report['candidate_sha256'][:16]}_cost{measured['cost']}.onnx"
            provisional.replace(final)
            report["candidate_path"] = str(final.relative_to(ROOT))
            report["verdict"] = "ACCEPT_PRIVATE_EXACT_100"
        else:
            report["reasons"].append("fresh_two_seed_dual_not_raw_exact_and_100pct_correct")

    if report["verdict"] == "REJECT" and provisional.exists():
        provisional.unlink()
    return report


def main() -> int:
    ort.set_default_logger_severity(4)
    rows = []
    for task, rewrites in TARGETS.items():
        print(f"audit task{task:03d}", flush=True)
        try:
            row = audit_task(task, rewrites, fresh_count=5000)
        except Exception as exc:
            row = {"task": task, "verdict": "REJECT", "reasons": [f"audit_exception:{type(exc).__name__}:{exc}"]}
        rows.append(row)
        (HERE / f"task{task:03d}_audit.json").write_text(json.dumps(row, indent=2) + "\n")
        print(task, row["verdict"], row.get("static_cost"), row.get("reasons"), flush=True)
    accepted = [row for row in rows if row.get("verdict") == "ACCEPT_PRIVATE_EXACT_100"]
    result = {
        "baseline": "submission_base_8004.50.zip",
        "baseline_sha256": sha256_bytes(BASELINE_ZIP.read_bytes()),
        "policy": "private-zero requires strict structure, truthful runtime shapes, strict lower actual/static cost, known 100% raw equality and correctness in both ORT modes, plus 5000 fresh x 2 independent seeds x both modes at 100% raw equality and correctness",
        "tasks": list(TARGETS),
        "rows": rows,
        "accepted": accepted,
        "verified_projected_gain": sum(row["measured_projected_gain"] for row in accepted),
        "final_verdict": "WINNERS" if accepted else "NO_GUARANTEED_PRIVATE_EXACT_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
