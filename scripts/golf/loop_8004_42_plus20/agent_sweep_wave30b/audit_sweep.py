#!/usr/bin/env python3
"""Fail-closed, non-promoting audit of sweep29 candidates for eight tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29"
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
TASKS = (199, 70, 333, 165, 169, 328, 379, 13)
PRIVATE_RISK = {70, 169}
GIANT_EINSUM_MIN_INPUTS = 15
MAX_BYTES = 1_440_000
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def candidates() -> dict[int, list[Path]]:
    return {
        199: sorted((SOURCE / "prune_latent").glob("task199_r*.onnx")),
        70: sorted((SOURCE / "prune_latent").glob("task070_r*.onnx"))
        + [SOURCE / "fuse_cooccur/task070_r01.onnx", SOURCE / "fuse_unique/task070_r01.onnx"],
        333: sorted((SOURCE / "prune_latent").glob("task333_r*.onnx")),
        165: [SOURCE / "exact_cse/task165.onnx"],
        169: [SOURCE / "exact_cse/task169.onnx"],
        328: [SOURCE / "reuse_contract/task328_r001.onnx"],
        379: [SOURCE / "reuse_contract/task379_r001.onnx"],
        13: sorted((SOURCE / "reuse_contract").glob("task013_r*.onnx")),
    }


def tensor_dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def static_audit(data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "serialized_bytes": len(data),
        "full_check": False,
        "strict_data_prop": False,
        "all_node_outputs_static_positive": False,
        "standard_domains": False,
        "conv_bias_ub0": False,
        "reasons": [],
    }
    if len(data) > MAX_BYTES:
        row["reasons"].append("file_too_large")
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["full_check_error"] = f"{type(exc).__name__}: {exc}"
        row["reasons"].append("full_check")
        return row
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row["strict_error"] = f"{type(exc).__name__}: {exc}"
        row["reasons"].append("strict_data_prop")

    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    giant_initializers = []
    finite = True
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        if array.size >= 10_000:
            giant_initializers.append({"name": item.name, "elements": int(array.size)})
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            finite = False
    nested = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    standard = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    row.update(
        {
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "op_histogram": dict(sorted(ops.items())),
            "max_einsum_inputs": max_einsum,
            "giant_einsum": max_einsum >= GIANT_EINSUM_MIN_INPUTS,
            "giant_initializers": giant_initializers,
            "lookup": bool(ops.get("TfIdfVectorizer") or ops.get("Hardmax")),
            "nested_graph_count": nested,
            "functions_count": len(model.functions),
            "sparse_initializer_count": len(model.graph.sparse_initializer),
            "banned_ops": banned,
            "finite_initializers": finite,
            "standard_domains": standard,
        }
    )
    if row["giant_einsum"]:
        row["reasons"].append("giant_einsum")
    if giant_initializers:
        row["reasons"].append("giant_initializer")
    if row["lookup"]:
        row["reasons"].append("lookup")
    if nested or model.functions or model.graph.sparse_initializer:
        row["reasons"].append("nested_functions_or_sparse")
    if banned:
        row["reasons"].append("banned_op")
    if not finite:
        row["reasons"].append("nonfinite_initializer")
    if not standard:
        row["reasons"].append("nonstandard_domain")

    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    output_names = {value.name for value in inferred.graph.output}
    nonstatic: list[str] = []
    memory = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in output_names:
                continue
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                nonstatic.append(name)
                continue
            dims = tensor_dims(value)
            if any(dim is None or dim <= 0 for dim in dims):
                nonstatic.append(name)
                continue
            try:
                itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)).itemsize
            except Exception:  # noqa: BLE001
                nonstatic.append(name)
                continue
            memory += math.prod(dims) * itemsize
    row["nonstatic_node_outputs"] = sorted(set(nonstatic))
    row["all_node_outputs_static_positive"] = not nonstatic
    row["static_memory"] = int(memory)
    row["params"] = int(scoring.calculate_params(model))
    row["static_cost"] = row["static_memory"] + row["params"]
    if nonstatic:
        row["reasons"].append("nonstatic_node_output")
    try:
        findings = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        findings = [{"check_error": f"{type(exc).__name__}: {exc}"}]
    row["conv_bias_findings"] = findings
    row["conv_bias_ub0"] = not findings
    if findings:
        row["reasons"].append("conv_bias_ub")
    row["reasons"] = sorted(set(row["reasons"]))
    row["pre_runtime_structural_pass"] = not row["reasons"]
    return row


def make_session(data: bytes, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_dual(task: int, data: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    total = sum(len(examples[split]) for split in ("train", "test", "arc-gen"))
    report: dict[str, Any] = {"total_examples": total, "modes": {}}
    for disable_all, mode in ((True, "disable_all"), (False, "default")):
        row: dict[str, Any] = {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        try:
            session = make_session(data, disable_all)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = total
            report["modes"][mode] = row
            continue
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[split]):
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
                        row["first_failure"] = row["first_failure"] or {"split": split, "index": index}
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "split": split,
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        report["modes"][mode] = row
    report["perfect"] = all(
        row["right"] == total and row["wrong"] == row["runtime_errors"] == 0
        for row in report["modes"].values()
    )
    return report


def runtime_shape_trace(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
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
    first = scoring.load_examples(task)["train"][0]
    benchmark = scoring.convert_to_numpy(first)
    assert benchmark is not None
    session = make_session(traced.SerializeToString(), True)
    arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = [
        {"name": name, "declared": tensor_dims(typed[name]), "actual": list(np.asarray(array).shape)}
        for name, array in zip(names, arrays)
        if tensor_dims(typed[name]) != list(np.asarray(array).shape)
    ]
    return {"traced": len(names), "mismatch_count": len(mismatches), "mismatches": mismatches[:50], "truthful": not mismatches}


def profiler_cost(data: bytes, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"wave30b_{task:03d}_", dir="/tmp") as workdir:
        path = Path(workdir) / f"{label}.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> int:
    ort.set_default_logger_severity(4)
    baseline: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            baseline[task] = {
                "sha256": sha256(data),
                "profile": profiler_cost(data, task, "baseline"),
            }

    task_rows: dict[str, Any] = {}
    accepted: list[dict[str, Any]] = []
    for task, paths in candidates().items():
        base_cost = baseline[task]["profile"]["cost"]
        rows = []
        for path in paths:
            data = path.read_bytes()
            static = static_audit(data)
            profile = profiler_cost(data, task, path.stem)
            gain = math.log(base_cost / profile["cost"]) if 0 < profile["cost"] < base_cost else 0.0
            reasons = list(static["reasons"])
            if profile["cost"] < 0 or profile["cost"] >= base_cost:
                reasons.append("not_strictly_cheaper_actual_cost")
            row: dict[str, Any] = {
                "path": relative(path),
                "sha256": sha256(data),
                "baseline_cost": base_cost,
                "profile": profile,
                "projected_gain": gain,
                "private_risk": task in PRIVATE_RISK,
                "static": static,
            }
            # Giant/lookup/structurally invalid candidates are fail-closed before
            # expensive correctness/fresh work.  Only non-giant exact-CSE leads
            # reach dual-ORT known and runtime-shape gates in this batch.
            if not reasons:
                row["known_dual"] = known_dual(task, data)
                if not row["known_dual"]["perfect"]:
                    reasons.append("known_dual_not_100_or_runtime_error")
                try:
                    row["runtime_shape_trace"] = runtime_shape_trace(task, data)
                except Exception as exc:  # noqa: BLE001
                    row["runtime_shape_trace"] = {
                        "truthful": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if not row["runtime_shape_trace"].get("truthful", False):
                    reasons.append("runtime_shapes_not_truthful_or_trace_runtime_error")
            else:
                row["runtime_and_known_skipped"] = "failed fail-closed structural/cost prerequisite"
            if not reasons:
                row["fresh"] = {"status": "would_be_required_but_not_reached_in_this_batch"}
                reasons.append("fresh_not_completed")
            else:
                row["fresh"] = {"status": "not_run_after_mandatory_pre_fresh_gate_failure"}
            row["reasons"] = sorted(set(reasons))
            row["accepted"] = False
            rows.append(row)
            print(task, path.name, profile["cost"], row["reasons"][:3], flush=True)
        task_rows[str(task)] = {
            "baseline": baseline[task],
            "private_risk": task in PRIVATE_RISK,
            "candidate_count": len(rows),
            "candidates": rows,
            "accepted_count": 0,
            "verdict": "REJECT_ALL",
        }
        (HERE / "audit_partial.json").write_text(
            json.dumps({"tasks": task_rows}, indent=2) + "\n", encoding="utf-8"
        )

    result = {
        "baseline_zip": relative(BASE_ZIP),
        "baseline_zip_sha256": sha256(BASE_ZIP.read_bytes()),
        "tasks": task_rows,
        "policy": {
            "known_dual": 1.0,
            "fresh_non_private_dual_each_seed": 0.90,
            "fresh_private_risk_dual_each_seed": 1.0,
            "runtime_errors": 0,
            "giant_einsum_min_inputs": GIANT_EINSUM_MIN_INPUTS,
            "lookup_cloak_giant": "forbidden",
            "strict_data_prop_and_truthful_runtime_shapes": True,
            "standard_domains": True,
            "conv_family_bias_ub": 0,
        },
        "accepted": accepted,
        "accepted_count": len(accepted),
        "aggregate_gain": 0.0,
        "zip_merge_performed": False,
        "verdict": "NO_SAFE_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
