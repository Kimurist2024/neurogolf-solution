#!/usr/bin/env python3
"""Policy and cost audit for task034/374/025/250 against 8004.50.

All evidence is written under this lane.  The script never promotes a model or
modifies an archive/artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (34, 374, 25, 250)
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
CONTROLS = {
    34: ROOT / "scripts/golf/scratch/task034/cand_v14.onnx",
    374: ROOT / "scripts/golf/scratch/task374/candidate_v5.onnx",
    25: ROOT / "scripts/golf/scratch_codex/task025/candidate_v22_conventional_spec.onnx",
    250: ROOT / "scripts/golf/scratch_codex/task250/groundup_485.onnx",
}
CANDIDATES = {
    250: HERE / "task250_direct_roi_rc.onnx",
}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TFIDFVECTORIZER", "SCATTERELEMENTS", "SCATTERND", "HARDMAX"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape_of(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0 or dim.HasField("dim_param"):
            return None
        dims.append(int(dim.dim_value))
    return dims


def make_session(model: onnx.ModelProto, disable: bool, *, profile_prefix: str | None = None):
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    return sanitized, ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def runtime_shape_truth(path: Path, task: int) -> dict[str, Any]:
    model = onnx.load(path)
    with tempfile.TemporaryDirectory(prefix=f"mid14_truth_{task:03d}_") as tmp:
        sanitized, session = make_session(
            model, True, profile_prefix=str(Path(tmp) / "truth")
        )
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(sanitized), strict_mode=True, data_prop=True
        )
        declared = {
            value.name: shape_of(value)
            for value in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        }
        example = next(
            raw
            for subset in ("train", "test", "arc-gen")
            for raw in scoring.load_examples(task).get(subset, [])
            if scoring.convert_to_numpy(raw) is not None
        )
        item = scoring.convert_to_numpy(example)
        assert item is not None
        output = session.run(None, {"input": item["input"]})[0]
        output_declared = shape_of(inferred.graph.output[0])
        trace_path = Path(session.end_profiling())
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actual: dict[str, list[int]] = {}
        for event in trace:
            args = event.get("args", {})
            shapes = args.get("output_type_shape")
            if event.get("cat") != "Node" or not shapes:
                continue
            node_name = str(event.get("name", "")).replace("_kernel_time", "")
            node = next((item for item in inferred.graph.node if item.name == node_name), None)
            if node is None:
                continue
            for index, shape_dict in enumerate(shapes):
                if index >= len(node.output) or not node.output[index] or not shape_dict:
                    continue
                dims = next(iter(shape_dict.values()))
                actual.setdefault(node.output[index], [int(value) for value in dims])
        mismatches = {
            name: {"declared": declared.get(name), "runtime": dims}
            for name, dims in actual.items()
            if declared.get(name) != dims
        }
        if output_declared != list(output.shape):
            mismatches["output"] = {
                "declared": output_declared,
                "runtime": list(output.shape),
            }
        return {
            "pass": not mismatches,
            "profiled_node_outputs": len(actual),
            "mismatches": mismatches,
        }


def structure(path: Path, task: int) -> dict[str, Any]:
    model = onnx.load(path)
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
    static = all(shape_of(value) is not None for value in values if value.type.HasField("tensor_type"))
    domains = sorted({node.domain for node in model.graph.node} | {item.domain for item in model.opset_import})
    try:
        truth = runtime_shape_truth(path, task)
    except Exception as exc:  # noqa: BLE001
        truth = {"pass": False, "error": f"{type(exc).__name__}:{exc}"}
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    input_shape = shape_of(model.graph.input[0]) if len(model.graph.input) == 1 else None
    output_shape = shape_of(model.graph.output[0]) if len(model.graph.output) == 1 else None
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive": static,
        "truthful_runtime_shapes": bool(truth.get("pass")),
        "canonical_io": input_shape == [1, 10, 30, 30] and output_shape == [1, 10, 30, 30],
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
        "no_lookup": all(node.op_type.upper() not in LOOKUP for node in model.graph.node),
        "no_shape_cloak": all(node.op_type != "CenterCropPad" for node in model.graph.node),
        "no_giant_einsum": max_einsum <= 16,
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
        ),
    }
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "params_static": sum(math.prod(item.dims) for item in model.graph.initializer),
        "ops": dict(sorted(ops.items())),
        "domains": domains,
        "max_einsum_inputs": max_einsum,
        "runtime_truth": truth,
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
    }


def profile(path: Path, task: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"mid14_score_{task:03d}_") as tmp:
        result = scoring.score_and_verify(
            onnx.load(path), task, tmp, label="audit", require_correct=False
        )
    if result is None:
        return {"error": "score_and_verify returned None"}
    return {
        "memory": int(result["memory"]),
        "params": int(result["params"]),
        "cost": int(result["cost"]),
        "score": float(result["score"]),
        "known_correct": bool(result["correct"]),
    }


def known_dual(path: Path, task: int) -> dict[str, Any]:
    model = onnx.load(path)
    converted = [
        item
        for subset in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(subset, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]
    result: dict[str, Any] = {}
    for mode, disable in (("disable_all", True), ("default", False)):
        row = {"right": 0, "wrong": 0, "runtime_errors": 0, "total": len(converted)}
        try:
            _, session = make_session(model, disable)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}:{exc}"
            result[mode] = row
            continue
        for item in converted:
            try:
                raw = session.run(None, {"input": item["input"]})[0]
                if np.array_equal((raw > 0).astype(np.float32), item["output"]):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
        result[mode] = row
    return result


def history_union() -> dict[str, Any]:
    screen = json.loads(
        (HERE.parent / "agent_clean95_all/screen_results.json").read_text(encoding="utf-8")
    )["rows"]
    union = json.loads(
        (HERE.parent / "agent_clean95_all/inventory_union.json").read_text(encoding="utf-8")
    )
    rows = {
        str(task): [item for item in screen if int(item.get("task", -1)) == task]
        for task in TASKS
    }
    return {
        "source": "agent_clean95_all exhaustive accepted-history + loose-archive union",
        "union_counts": union["counts"],
        "unique_by_task": {str(task): union["unique_by_task"].get(str(task), 0) for task in TASKS},
        "rows": rows,
    }


def main() -> None:
    current = json.loads((HERE.parent / "current_costs_8004_50.json").read_text(encoding="utf-8"))["ranked"]
    current_by_task = {int(item["task"]): item for item in current}
    tasks: dict[str, Any] = {}
    for task in TASKS:
        baseline = HERE / "baseline" / f"task{task:03d}.onnx"
        baseline_profile = profile(baseline, task)
        entry: dict[str, Any] = {
            "baseline": {
                "path": str(baseline.relative_to(ROOT)),
                "reported": current_by_task[task],
                "profile": baseline_profile,
                "structure": structure(baseline, task),
                "known_dual": known_dual(baseline, task),
            },
            "honest_spec_control": {
                "path": str(CONTROLS[task].relative_to(ROOT)),
                "profile": profile(CONTROLS[task], task),
                "structure": structure(CONTROLS[task], task),
                "known_dual": known_dual(CONTROLS[task], task),
            },
        }
        if task in CANDIDATES:
            candidate = CANDIDATES[task]
            candidate_profile = profile(candidate, task)
            entry["new_candidate"] = {
                "path": str(candidate.relative_to(ROOT)),
                "profile": candidate_profile,
                "structure": structure(candidate, task),
                "known_dual": known_dual(candidate, task),
                "strictly_cheaper": candidate_profile.get("cost", 10**18)
                < baseline_profile.get("cost", -1),
            }
        tasks[str(task)] = entry
    payload = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha256(BASE_ZIP),
        "tasks": tasks,
        "history": history_union(),
    }
    (HERE / "audit_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(HERE / "audit_results.json")


if __name__ == "__main__":
    main()
