#!/usr/bin/env python3
"""Read-only audit for tasks 002/088/089/191 against the 8004.50 archive.

This script deliberately never calls try_candidate and never promotes a model.
It inventories the retained loose history, applies structural/actual-cost gates,
and runs fresh validation only if a model survives every cheaper pre-fresh gate.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (2, 88, 89, 191)
HASHES = {2: "00d62c1b", 88: "3de23699", 89: "3e980e27", 191: "7df24a62"}
BASE_COST = {2: 1286, 88: 902, 89: 1361, 191: 897}
PRIVATE_ZERO = {191}
FRESH_SEEDS = (260714009, 910714009)
FRESH_PER_SEED = 5000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def dim_list(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def discover() -> dict[int, list[Path]]:
    """Find retained loose models; ZIP lineages were already deduped into these."""
    roots = [ROOT / "scripts/golf", ROOT / "others", ROOT / "artifacts"]
    found: dict[int, list[Path]] = defaultdict(list)
    for task in TASKS:
        names = (f"task{task:03d}*.onnx",)
        for root in roots:
            if not root.exists():
                continue
            for name in names:
                for path in root.rglob(name):
                    if HERE in path.parents:
                        continue
                    found[task].append(path)
        # Explicit generator-rule-derived controls whose basename is not taskNNN.
    explicit = {
        2: [ROOT / "artifacts/handcrafted/task002.onnx"],
        88: [ROOT / "scripts/golf/scratch/task088/candidate10.onnx"],
        89: [ROOT / "scripts/golf/scratch_claude/task089/rebuild.onnx"],
        191: [ROOT / "scripts/golf/scratch_claude/task191/rebuild.onnx"],
    }
    for task, paths in explicit.items():
        found[task].extend(path for path in paths if path.exists())
    return {task: sorted(set(paths)) for task, paths in found.items()}


def static_audit(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": rel(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    try:
        model = onnx.load(path)
    except Exception as exc:  # noqa: BLE001
        row.update(load=False, reasons=["load"], error=f"{type(exc).__name__}: {exc}")
        return row
    row["load"] = True
    ops = Counter(node.op_type for node in model.graph.node)
    row.update(
        nodes=len(model.graph.node),
        initializers=len(model.graph.initializer),
        params=int(scoring.calculate_params(model)),
        ops=dict(ops),
        max_einsum_inputs=max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    )
    reasons: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["full_check"] = False
        row["full_check_error"] = f"{type(exc).__name__}: {exc}"
        reasons.append("full_check")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row["strict_data_prop"] = False
        row["strict_error"] = f"{type(exc).__name__}: {exc}"
        reasons.append("strict_data_prop")

    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    output_names = {value.name for value in model.graph.output}
    nonstatic: list[str] = []
    unknown_dtype: list[str] = []
    static_memory = 0
    for node in model.graph.node:
        for name in node.output:
            if not name or name in output_names:
                continue
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                nonstatic.append(name)
                continue
            dims = dim_list(value)
            if any(not isinstance(dim, int) or dim <= 0 for dim in dims):
                nonstatic.append(name)
                continue
            itemsize = DTYPE_BYTES.get(int(value.type.tensor_type.elem_type))
            if itemsize is None:
                unknown_dtype.append(name)
                continue
            static_memory += math.prod(dims) * itemsize
    row["nonstatic"] = sorted(set(nonstatic))
    row["unknown_dtype"] = sorted(set(unknown_dtype))
    row["static_memory"] = int(static_memory)
    row["static_cost"] = int(static_memory + row["params"])
    if nonstatic:
        reasons.append("nonstatic_shape")
    if unknown_dtype:
        reasons.append("unknown_dtype")

    standard = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    if not standard:
        reasons.append("nonstandard_domain")
    row["standard_domains"] = standard
    nested = any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    if model.functions or model.graph.sparse_initializer or nested:
        reasons.append("functions_sparse_nested")
    row["functions_sparse_nested"] = bool(model.functions or model.graph.sparse_initializer or nested)
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    if banned:
        reasons.append("banned_op")
    row["banned_ops"] = banned
    if row["max_einsum_inputs"] > 16:
        reasons.append("giant_einsum")
    giant = [
        {"name": init.name, "elements": int(numpy_helper.to_array(init).size)}
        for init in model.graph.initializer
        if numpy_helper.to_array(init).size >= 100_000
    ]
    if giant:
        reasons.append("giant_initializer")
    row["giant_initializers"] = giant
    if ops.get("TfIdfVectorizer", 0):
        reasons.append("lookup")
    row["lookup"] = bool(ops.get("TfIdfVectorizer", 0))
    try:
        bias = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        bias = [{"check_error": f"{type(exc).__name__}: {exc}"}]
    if bias:
        reasons.append("conv_bias_ub")
    row["conv_bias_findings"] = bias
    finite = all(
        arr.dtype.kind not in "fc" or bool(np.isfinite(arr).all())
        for arr in (numpy_helper.to_array(item) for item in model.graph.initializer)
    )
    if not finite:
        reasons.append("nonfinite_initializer")
    row["finite_initializers"] = finite
    row["reasons"] = sorted(set(reasons))
    row["structural_pass"] = not row["reasons"]
    return row


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_dual(task: int, model: onnx.ModelProto) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for disabled, key in ((True, "disable_all"), (False, "default")):
        row = {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        try:
            session = make_session(model, disabled)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = 1
            results[key] = row
            continue
        for split, examples in scoring.load_examples(task).items():
            for index, example in enumerate(examples):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    if np.array_equal(raw > 0, benchmark["output"].astype(bool)):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "split": split,
                            "index": index,
                        }
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "split": split,
                        "index": index,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
        results[key] = row
    return results


def runtime_shape_trace(task: int, model: onnx.ModelProto) -> dict[str, Any]:
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
    session = make_session(traced, True)
    first = next(iter(scoring.load_examples(task)["train"]))
    benchmark = scoring.convert_to_numpy(first)
    assert benchmark is not None
    arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = [
        {
            "tensor": name,
            "declared": dim_list(typed[name]),
            "runtime": list(np.asarray(array).shape),
        }
        for name, array in zip(names, arrays)
        if dim_list(typed[name]) != list(np.asarray(array).shape)
    ]
    output_names = {value.name for value in model.graph.output}
    memory = sum(
        np.asarray(array).nbytes
        for name, array in zip(names, arrays)
        if name not in output_names
    )
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:30],
        "one_example_runtime_memory": int(memory),
        "shape_cloak_free": not mismatches,
    }


def actual_score(task: int, model: onnx.ModelProto, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"mid9_{task:03d}_", dir="/tmp") as workdir:
        try:
            return scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label=label, require_correct=False
            )
        except Exception:
            return None


def fresh_seed(task: int, model: onnx.ModelProto, seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{HASHES[task]}")
    sessions = {
        "disable_all": make_session(model, True),
        "default": make_session(model, False),
    }
    rows = {
        key: {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        for key in sessions
    }
    valid = attempts = generation_errors = 0
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
            continue
        valid += 1
        for key, session in sessions.items():
            row = rows[key]
            try:
                raw = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
                if np.array_equal(raw > 0, benchmark["output"].astype(bool)):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    row["first_failure"] = row["first_failure"] or {"case": valid}
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
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


def main() -> int:
    ort.set_default_logger_severity(4)
    discovered = discover()
    inventory: dict[str, Any] = {"tasks": {}, "total_paths": 0, "total_unique_sha": 0}
    unique: dict[int, dict[str, dict[str, Any]]] = {}
    for task, paths in discovered.items():
        by_sha: dict[str, dict[str, Any]] = {}
        for path in paths:
            try:
                digest = sha256(path)
            except OSError:
                continue
            item = by_sha.setdefault(digest, {"sha256": digest, "sources": []})
            item["sources"].append(rel(path))
            item.setdefault("representative", path)
        unique[task] = by_sha
        inventory["tasks"][str(task)] = {
            "paths": len(paths),
            "unique_sha": len(by_sha),
            "baseline_cost": BASE_COST[task],
            "private_zero_catalog": task in PRIVATE_ZERO,
        }
        inventory["total_paths"] += len(paths)
        inventory["total_unique_sha"] += len(by_sha)
    (HERE / "history_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        baseline_path = HERE / "baseline" / f"task{task:03d}.onnx"
        baseline_sha = sha256(baseline_path)
        # Baseline plus every SHA-distinct history family.
        candidates = [("baseline", baseline_path, ["submission_base_8004.50.zip"])]
        for digest, item in unique[task].items():
            if digest == baseline_sha:
                continue
            candidates.append(("history", item["representative"], item["sources"]))
        for kind, path, sources in candidates:
            audit = static_audit(path)
            row: dict[str, Any] = {
                "task": task,
                "kind": kind,
                "sources": sources,
                "baseline_cost": BASE_COST[task],
                "private_zero_catalog": task in PRIVATE_ZERO,
                "static": audit,
            }
            reasons = list(audit.get("reasons", []))
            if kind != "baseline" and audit.get("static_cost", 10**18) >= BASE_COST[task]:
                reasons.append("static_cost_not_lower")
            should_runtime = kind == "baseline" or (
                audit.get("structural_pass") and audit.get("static_cost", 10**18) < BASE_COST[task]
            )
            if should_runtime:
                model = onnx.load(path)
                row["known"] = known_dual(task, model)
                row["official_like"] = actual_score(task, model, f"mid9_{task:03d}")
                try:
                    row["runtime_shapes"] = runtime_shape_trace(task, model)
                except Exception as exc:  # noqa: BLE001
                    row["runtime_shapes"] = {
                        "shape_cloak_free": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                profile = row.get("official_like")
                if kind != "baseline" and (
                    not profile or int(profile["cost"]) >= BASE_COST[task]
                ):
                    reasons.append("actual_cost_not_lower")
                known = row["known"]
                for mode in ("disable_all", "default"):
                    result = known[mode]
                    if result.get("wrong") or result.get("runtime_errors") or not result.get("right"):
                        reasons.append(f"known_{mode}")
                if not row["runtime_shapes"].get("shape_cloak_free"):
                    reasons.append("runtime_shape_mismatch")
            else:
                row["runtime_skipped"] = "failed static/cheaper prerequisite"
            row["pre_fresh_reasons"] = sorted(set(reasons))
            row["pre_fresh_pass"] = kind != "baseline" and not row["pre_fresh_reasons"]
            if row["pre_fresh_pass"]:
                model = onnx.load(path)
                row["fresh"] = [fresh_seed(task, model, seed) for seed in FRESH_SEEDS]
                threshold = 1.0 if task in PRIVATE_ZERO else 0.90
                row["fresh_threshold"] = threshold
                row["accepted"] = all(
                    mode["runtime_errors"] == 0 and mode["accuracy"] >= threshold
                    for seed_row in row["fresh"]
                    for mode in seed_row["modes"].values()
                )
            else:
                row["fresh"] = []
                row["accepted"] = False
            rows.append(row)
            print(task, kind, audit.get("sha256", "")[:12], row["pre_fresh_reasons"][:5], flush=True)
            (HERE / "audit_partial.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    accepted = [row for row in rows if row["accepted"]]
    result = {
        "baseline_zip": "submission_base_8004.50.zip",
        "tasks": list(TASKS),
        "baseline_costs": {str(k): v for k, v in BASE_COST.items()},
        "private_zero_catalog": sorted(PRIVATE_ZERO),
        "policy": {
            "known": 1.0,
            "fresh_non_private": 0.90,
            "fresh_private_zero": 1.0,
            "fresh_seeds": list(FRESH_SEEDS),
            "fresh_per_seed": FRESH_PER_SEED,
        },
        "rows": rows,
        "accepted": accepted,
        "accepted_count": len(accepted),
        "aggregate_gain": sum(
            math.log(BASE_COST[row["task"]] / row["official_like"]["cost"])
            for row in accepted
        ),
    }
    (HERE / "audit_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
