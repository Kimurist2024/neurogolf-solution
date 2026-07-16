#!/usr/bin/env python3
"""Exact scalar/initializer alias scan on the immutable 8012.15 authority."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
AUTHORITY_LB = 8012.15
TASKS = (5, 54, 62, 80, 156, 182, 221, 297, 349, 382)
WORKERS = 10
KNOWN_BLACK = {70, 134, 202, 343}
PRIVATE_ZERO = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000

# Targets consumed by any of these operators/roles are never removed.  This
# is deliberately broader than the brief's minimum protection list.
PROTECTED_OPS = {
    "Conv", "ConvTranspose", "ConvInteger", "QLinearConv", "QLinearMatMul",
    "MatMul", "MatMulInteger", "Gemm", "Einsum",
    "Concat", "Scatter", "ScatterElements", "ScatterND",
    "Reshape", "Expand", "ConstantOfShape", "Slice", "Pad", "Resize", "Tile",
    "Squeeze", "Unsqueeze", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceL1",
    "ReduceL2", "TopK", "Split", "Gather", "GatherElements", "GatherND",
    "CenterCropPad", "RoiAlign", "MaxRoiPool", "OneHot", "Shape", "Size",
}
ELEMENTWISE = {
    "Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "GreaterOrEqual",
    "Less", "LessOrEqual", "Max", "Min", "And", "Or", "Xor", "BitwiseAnd",
    "BitwiseOr", "BitwiseXor", "Mod", "BitShift", "Sum", "Clip", "Where",
}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_support():
    path = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
    name = f"restart414_support_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.POLICY_THRESHOLD = THRESHOLD
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


def tensor_key(tensor: onnx.TensorProto) -> tuple[int, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(numpy_helper.to_array(tensor))
    return int(tensor.data_type), tuple(int(dim) for dim in tensor.dims), array.tobytes()


def element_count(tensor: onnx.TensorProto) -> int:
    return int(math.prod(tensor.dims)) if tensor.dims else 1


def uses(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                result[name].append(
                    {
                        "node_index": node_index,
                        "node": node.name or (node.output[0] if node.output else ""),
                        "op": node.op_type,
                        "input_index": input_index,
                    }
                )
    return result


def protected_names(model: onnx.ModelProto) -> tuple[set[str], list[dict[str, Any]]]:
    initializers = {item.name for item in model.graph.initializer}
    protected: set[str] = set()
    reasons: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in PROTECTED_OPS:
            continue
        for input_index, name in enumerate(node.input):
            if name in initializers:
                protected.add(name)
                reasons.append(
                    {
                        "initializer": name,
                        "op": node.op_type,
                        "node_index": node_index,
                        "input_index": input_index,
                    }
                )
    return protected, reasons


def exact_scalar_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.size != 1 or right.size == 0:
        return False
    return bool(np.all(right == left.reshape(-1)[0]))


def find_plans(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inits = {item.name: item for item in model.graph.initializer}
    arrays = {name: np.asarray(numpy_helper.to_array(item)) for name, item in inits.items()}
    use_map = uses(model)
    protected, protection_reasons = protected_names(model)
    graph_io = {item.name for item in [*model.graph.input, *model.graph.output]}
    plans_by_target: dict[str, dict[str, Any]] = {}

    # 1. Byte/dtype/shape-identical duplicates.  Even these are skipped when
    # the removed name is protected by a weight/shape-sensitive consumer.
    groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for name, item in inits.items():
        groups[tensor_key(item)].append(name)
    for key, names in groups.items():
        if len(names) < 2:
            continue
        names.sort(key=lambda name: (name in protected, element_count(inits[name]), name))
        source = names[0]
        for target in names[1:]:
            if target in protected or target in graph_io or not use_map[target]:
                continue
            plans_by_target[target] = {
                "kind": "exact_duplicate",
                "source": source,
                "target": target,
                "removed_elements": element_count(inits[target]),
                "proof": {
                    "rule": "same dtype, serialized shape, and tensor bytes",
                    "dtype": TensorProto.DataType.Name(key[0]),
                    "shape": list(key[1]),
                    "tensor_bytes_sha256": sha256(key[2]),
                    "target_uses": use_map[target],
                },
            }

    # 2. CastLike's second input contributes only its element type.  Reuse an
    # existing initializer of the same element type and delete the template.
    for target, item in inits.items():
        if target in plans_by_target or target in protected or target in graph_io:
            continue
        target_uses = use_map[target]
        if not target_uses or not all(
            row["op"] == "CastLike" and row["input_index"] == 1 for row in target_uses
        ):
            continue
        sources = [
            name for name, source in inits.items()
            if name != target and int(source.data_type) == int(item.data_type)
        ]
        if not sources:
            continue
        sources.sort(
            key=lambda name: (
                all(
                    row["op"] == "CastLike" and row["input_index"] == 1
                    for row in use_map[name]
                ) if use_map[name] else True,
                element_count(inits[name]),
                name,
            )
        )
        source = sources[0]
        plans_by_target[target] = {
            "kind": "castlike_type_template",
            "source": source,
            "target": target,
            "removed_elements": element_count(item),
            "proof": {
                "rule": "CastLike input[1] uses element type only; source dtype is identical",
                "dtype": TensorProto.DataType.Name(item.data_type),
                "target_shape": list(item.dims),
                "source_shape": list(inits[source].dims),
                "target_uses": target_uses,
            },
        }

    # 3. A uniform tensor can be replaced by an existing equal scalar only
    # when every use is a multidirectional elementwise-broadcast position.
    for target, item in inits.items():
        if target in plans_by_target or target in protected or target in graph_io:
            continue
        target_array = arrays[target]
        if target_array.size <= 1 or not np.all(np.isfinite(target_array)):
            continue
        target_uses = use_map[target]
        if not target_uses:
            continue
        allowed = True
        for row in target_uses:
            if row["op"] not in ELEMENTWISE:
                allowed = False
                break
            if row["op"] == "Where" and row["input_index"] == 0:
                allowed = False
                break
            if row["op"] == "Clip" and row["input_index"] == 0:
                allowed = False
                break
        if not allowed:
            continue
        sources = [
            name for name, array in arrays.items()
            if name != target
            and arrays[name].size == 1
            and exact_scalar_equal(arrays[name], target_array)
        ]
        if not sources:
            continue
        sources.sort(key=lambda name: (name in protected, name))
        source = sources[0]
        scalar_bytes = np.ascontiguousarray(arrays[source]).tobytes()
        plans_by_target[target] = {
            "kind": "uniform_elementwise_broadcast",
            "source": source,
            "target": target,
            "removed_elements": element_count(item),
            "proof": {
                "rule": "uniform target equals existing scalar; all consumers broadcast elementwise",
                "dtype": str(target_array.dtype),
                "target_shape": list(target_array.shape),
                "source_shape": list(arrays[source].shape),
                "scalar_bytes_hex": scalar_bytes.hex(),
                "target_uses": target_uses,
            },
        }

    plans = sorted(
        plans_by_target.values(),
        key=lambda row: (-int(row["removed_elements"]), row["kind"], row["target"]),
    )
    audit = {
        "initializer_count": len(inits),
        "protected_initializer_count": len(protected),
        "protected_initializers": sorted(protected),
        "protection_reasons": protection_reasons,
        "plan_count": len(plans),
    }
    return plans, audit


def build_candidate(
    original: onnx.ModelProto, plans: list[dict[str, Any]]
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(original)
    originals = {item.name: item.SerializeToString() for item in original.graph.initializer}
    protected, _ = protected_names(original)
    removed: set[str] = set()
    used_sources: set[str] = set()
    applied: list[dict[str, Any]] = []
    for plan in plans:
        source = str(plan["source"])
        target = str(plan["target"])
        if (
            target in removed
            or source in removed
            or target in protected
            or target in used_sources
        ):
            continue
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == target:
                    node.input[index] = source
        keep = [item for item in model.graph.initializer if item.name != target]
        if len(keep) == len(model.graph.initializer):
            continue
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        keep_vi = [item for item in model.graph.value_info if item.name != target]
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep_vi)
        removed.add(target)
        used_sources.add(source)
        applied.append(plan)
    current = {item.name: item.SerializeToString() for item in model.graph.initializer}
    protection_failures = [
        name for name in sorted(protected)
        if name not in current or current[name] != originals[name]
    ]
    return model, {
        "plans": applied,
        "removed_initializers": sorted(removed),
        "removed_elements": sum(int(row["removed_elements"]) for row in applied),
        "protected_initializer_integrity": not protection_failures,
        "protected_initializer_failures": protection_failures,
        "node_count_unchanged": len(model.graph.node) == len(original.graph.node),
    }


def value_signature(value: onnx.ValueInfoProto) -> tuple[int, tuple[int | None, ...]]:
    tensor = value.type.tensor_type
    dims = tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in tensor.shape.dim
    )
    return int(tensor.elem_type), dims


def inferred_signatures(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int | None, ...]]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    return {
        value.name: value_signature(value)
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }


def generate_candidates(
    task: int, authority_data: bytes
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = onnx.load_model_from_string(authority_data)
    plans, plan_audit = find_plans(original)
    variants: list[tuple[str, list[dict[str, Any]]]] = [
        (f"single_{index:02d}", [plan]) for index, plan in enumerate(plans, 1)
    ]
    if len(plans) > 1:
        variants.append(("combined", plans))
    seen = {sha256(authority_data)}
    candidates: list[dict[str, Any]] = []
    original_signatures = inferred_signatures(original)
    errors: list[dict[str, Any]] = []
    for label, selected in variants:
        try:
            model, build = build_candidate(original, selected)
            data = model.SerializeToString()
            digest = sha256(data)
            if digest in seen or not build["plans"]:
                continue
            seen.add(digest)
            full_check = strict_shape = False
            full_error = strict_error = None
            try:
                onnx.checker.check_model(copy.deepcopy(model), full_check=True)
                full_check = True
            except Exception as exc:  # noqa: BLE001
                full_error = f"{type(exc).__name__}: {exc}"
            try:
                candidate_signatures = inferred_signatures(model)
                strict_shape = True
            except Exception as exc:  # noqa: BLE001
                candidate_signatures = {}
                strict_error = f"{type(exc).__name__}: {exc}"
            common_names = set(original_signatures) & set(candidate_signatures)
            shape_signature_equal = bool(
                strict_shape
                and all(original_signatures[name] == candidate_signatures[name] for name in common_names)
            )
            candidates.append(
                {
                    "task": task,
                    "label": label,
                    "sha256": digest,
                    "data": data,
                    "file_bytes": len(data),
                    "build": build,
                    "proof_complete": bool(
                        build["protected_initializer_integrity"]
                        and build["node_count_unchanged"]
                        and full_check
                        and strict_shape
                        and shape_signature_equal
                    ),
                    "generation_structure": {
                        "full_check": full_check,
                        "full_error": full_error,
                        "strict_shape": strict_shape,
                        "strict_error": strict_error,
                        "common_inferred_tensor_count": len(common_names),
                        "inferred_shape_dtype_equal_for_common_tensors": shape_signature_equal,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
    return candidates, {"plans": plans, "plan_audit": plan_audit, "errors": errors}


def runtime_row_pass(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("accuracy", 0.0)) >= THRESHOLD
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def compact_runtime(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error",
        "first_shape_mismatch", "optimization", "threads",
    )
    result = {key: row.get(key) for key in keys if key in row}
    result["policy90"] = runtime_row_pass(row)
    if row.get("session_error"):
        result["session_error"] = row["session_error"]
    return result


def evaluate_four(support, data: bytes, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: compact_runtime(row)
        for name, row in support.evaluate_four(data, cases).items()
    }


def four_clean(rows: dict[str, dict[str, Any]]) -> bool:
    return bool(
        len(rows) == 4
        and all(
            row.get("errors") == 0
            and row.get("nonfinite_cases") == 0
            and row.get("nonfinite_elements") == 0
            and row.get("runtime_shape_mismatches") == 0
            and row.get("small_positive_elements_0_to_0_25") == 0
            and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
            and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
            and not row.get("session_error")
            for row in rows.values()
        )
    )


def raw_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left.keys() == right.keys()
        and all(
            left[name].get("total") == right[name].get("total")
            and left[name].get("raw_sha256") == right[name].get("raw_sha256")
            and left[name].get("sign_sha256") == right[name].get("sign_sha256")
            for name in left
        )
    )


def mandatory_structure(support, task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    generic = support.structural_audit(task, model, data)
    try:
        trace = support.runtime_shape_trace(task, copy.deepcopy(model))
        trace["shape_cloak_findings"] = len(trace.get("declared_actual_mismatches", []))
        trace["truthful"] = bool(
            not trace.get("error") and trace["shape_cloak_findings"] == 0
        )
    except Exception as exc:  # noqa: BLE001
        trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    reasons: list[str] = []
    if not generic.get("full_check"):
        reasons.append("full_checker")
    if not generic.get("strict_shape_data_prop"):
        reasons.append("strict_shape")
    if not generic.get("canonical_io"):
        reasons.append("noncanonical_io")
    if generic.get("missing_node_outputs") or generic.get("nonstatic_node_outputs"):
        reasons.append("untyped_or_nonstatic")
    if generic.get("banned_ops"):
        reasons.append("banned_ops")
    if generic.get("nonstandard_domains") or generic.get("nested_graphs") or generic.get("functions"):
        reasons.append("nonstandard_or_nested")
    if generic.get("sparse_initializers") or generic.get("external_initializers"):
        reasons.append("sparse_or_external")
    if generic.get("nonfinite_initializers"):
        reasons.append("nonfinite_initializer")
    if generic.get("conv_bias_ub_findings"):
        reasons.append("conv_bias_ub")
    if not trace.get("truthful"):
        reasons.append("runtime_shape_cloak")
    return {
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "full_check": generic.get("full_check"),
        "strict_shape_data_prop": generic.get("strict_shape_data_prop"),
        "banned_ops": generic.get("banned_ops"),
        "nonfinite_initializers": generic.get("nonfinite_initializers"),
        "conv_bias_ub_findings": generic.get("conv_bias_ub_findings"),
        "runtime_shape_trace": trace,
    }


def profile(support, task: int, data: bytes, label: str) -> dict[str, Any] | None:
    try:
        return support.official_profile(task, onnx.load_model_from_string(data), label)
    except Exception:  # noqa: BLE001
        return None


def worker_main(args: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    index = int(args["worker_index"])
    tasks = [int(task) for task in args["tasks"]]
    models = {int(task): data for task, data in args["authority"].items()}
    candidates = {int(task): rows for task, rows in args["candidates"].items()}
    support = load_support()
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    results: list[dict[str, Any]] = []
    candidate_dir = HERE / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        authority_data = models[task]
        authority_profile = profile(
            support, task, authority_data, f"restart414_authority_{task:03d}"
        )
        if authority_profile is None:
            raise RuntimeError(f"authority task{task:03d} unscorable")
        prelim: list[dict[str, Any]] = []
        for ordinal, candidate in enumerate(candidates[task], 1):
            data = candidate["data"]
            row = {key: value for key, value in candidate.items() if key != "data"}
            row["profile"] = profile(
                support, task, data, f"restart414_w{index}_{task:03d}_{ordinal}"
            )
            row["structure"] = mandatory_structure(support, task, data)
            row["strict_lower_actual"] = bool(
                row["profile"] is not None
                and int(row["profile"]["cost"]) < int(authority_profile["cost"])
            )
            row["preliminary_pass"] = bool(
                row["proof_complete"]
                and row["build"]["protected_initializer_integrity"]
                and row["structure"]["pass"]
                and row["strict_lower_actual"]
            )
            row["data"] = data
            prelim.append(row)
        prelim.sort(
            key=lambda row: (
                10**18 if row["profile"] is None else int(row["profile"]["cost"]),
                str(row["sha256"]),
            )
        )
        eligible = [row for row in prelim if row["preliminary_pass"]]
        audits: list[dict[str, Any]] = []
        finalist = None
        if eligible:
            known_cases, known_counts = support.known_cases(task)
            fresh_sets = []
            for seed in (414_200_000 + task, 414_300_000 + task):
                fresh_sets.append(support.fresh_cases(task, seed, task_map))
            corpora = [("known", known_cases, known_counts)] + [
                (f"fresh_{generation['seed']}", cases, generation)
                for cases, generation in fresh_sets
            ]
            authority_runtime = {
                label: {"meta": meta, "runtime": evaluate_four(support, authority_data, cases)}
                for label, cases, meta in corpora
            }
            for rank, row in enumerate(eligible, 1):
                candidate_runtime = {
                    label: {"meta": meta, "runtime": evaluate_four(support, row["data"], cases)}
                    for label, cases, meta in corpora
                }
                comparisons = {
                    label: raw_equal(
                        candidate_runtime[label]["runtime"],
                        authority_runtime[label]["runtime"],
                    )
                    for label in candidate_runtime
                }
                all_candidate_rows = [
                    runtime_row
                    for corpus in candidate_runtime.values()
                    for runtime_row in corpus["runtime"].values()
                ]
                runtime_clean = all(
                    four_clean(corpus["runtime"])
                    for corpus in candidate_runtime.values()
                )
                normal_policy90 = bool(
                    runtime_clean and all(item["policy90"] for item in all_candidate_rows)
                )
                exact_authority = bool(
                    runtime_clean
                    and all(comparisons.values())
                    and row["proof_complete"]
                    and row["build"]["protected_initializer_integrity"]
                )
                classification = None
                if exact_authority:
                    classification = "EXACT_AUTHORITY_EQUIVALENT"
                elif normal_policy90:
                    classification = "POLICY90_NONEXACT"
                audit = {
                    **{key: value for key, value in row.items() if key != "data"},
                    "rank": rank,
                    "authority_runtime": authority_runtime,
                    "candidate_runtime": candidate_runtime,
                    "raw_equivalence_by_corpus": comparisons,
                    "all_audit_raw_authority_equivalent": all(comparisons.values()),
                    "runtime_clean": runtime_clean,
                    "normal_policy90": normal_policy90,
                    "exact_authority_equivalent": exact_authority,
                    "classification": classification,
                    "admitted": classification is not None,
                }
                audits.append(audit)
                print(
                    json.dumps(
                        {
                            "worker": index,
                            "pid": os.getpid(),
                            "task": task,
                            "rank": rank,
                            "cost": row["profile"]["cost"],
                            "raw_equal": comparisons,
                            "policy90": normal_policy90,
                            "class": classification,
                        }
                    ),
                    flush=True,
                )
                if classification is not None:
                    output = candidate_dir / (
                        f"task{task:03d}_cost{int(row['profile']['cost'])}_"
                        f"{row['sha256'][:12]}_{classification}.onnx"
                    )
                    output.write_bytes(row["data"])
                    audit["saved_path"] = rel(output)
                    audit["saved_sha256"] = sha256(output.read_bytes())
                    audit["projected_gain"] = math.log(
                        int(authority_profile["cost"]) / int(row["profile"]["cost"])
                    )
                    finalist = audit
                    break
        result = {
            "task": task,
            "authority": {
                "sha256": sha256(authority_data),
                "bytes": len(authority_data),
                "profile": authority_profile,
            },
            "generated_candidates": len(candidates[task]),
            "preliminary_eligible": len(eligible),
            "preliminary": [
                {key: value for key, value in row.items() if key != "data"}
                for row in prelim
            ],
            "audits": audits,
            "finalist": finalist,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "worker": index,
                    "pid": os.getpid(),
                    "task_done": task,
                    "generated": len(candidates[task]),
                    "eligible": len(eligible),
                    "winner": None if finalist is None else {
                        "cost": finalist["profile"]["cost"],
                        "class": finalist["classification"],
                    },
                }
            ),
            flush=True,
        )
    return {
        "worker_index": index,
        "pid": os.getpid(),
        "tasks": tasks,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA changed")
    if set(TASKS) & (KNOWN_BLACK | PRIVATE_ZERO):
        raise RuntimeError("target set contains catalog/black task")
    costs = {
        int(row["task"].removeprefix("task")): int(row["cost"])
        for row in csv.DictReader((ROOT / "all_scores.csv").open())
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    generated: dict[int, list[dict[str, Any]]] = {}
    generation: dict[int, dict[str, Any]] = {}
    for task in TASKS:
        generated[task], generation[task] = generate_candidates(task, authority[task])

    # The user explicitly requested ten-way exploration.  There are exactly
    # ten targets, so each spawned worker normally owns one task.  The generic
    # balancing below still behaves correctly if TASKS changes later.
    groups: list[list[int]] = [[] for _ in range(WORKERS)]
    loads = [0 for _ in range(WORKERS)]
    for task in sorted(TASKS, key=lambda item: (-len(generated[item]), item)):
        target = min(
            range(WORKERS), key=lambda item: (loads[item], len(groups[item]), item)
        )
        groups[target].append(task)
        loads[target] += max(1, len(generated[task]))
    for group in groups:
        group.sort()
    inventory = {
        "authority": {"zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256, "lb": AUTHORITY_LB},
        "tasks": list(TASKS),
        "costs": {str(task): costs[task] for task in TASKS},
        "known_black_excluded": sorted(KNOWN_BLACK),
        "private_zero_catalog": {
            "path": "docs/golf/private_zero_tasks.md",
            "sha256": sha256((ROOT / "docs/golf/private_zero_tasks.md").read_bytes()),
        },
        "generation": generation,
        "generated_candidate_count": sum(len(rows) for rows in generated.values()),
        "generated_by_task": {str(task): len(generated[task]) for task in TASKS},
        "worker_groups": [
            {"worker_index": index, "tasks": group, "candidate_load": loads[index]}
            for index, group in enumerate(groups)
        ],
        "protected_ops": sorted(PROTECTED_OPS),
        "elementwise_broadcast_ops": sorted(ELEMENTWISE),
    }
    (HERE / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        json.dumps(
            {
                "generated": inventory["generated_by_task"],
                "groups": inventory["worker_groups"],
            },
            indent=2,
        ),
        flush=True,
    )

    args = [
        {
            "worker_index": index,
            "tasks": group,
            "authority": {task: authority[task] for task in group},
            "candidates": {task: generated[task] for task in group},
        }
        for index, group in enumerate(groups)
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=WORKERS) as pool:
        workers = pool.map(worker_main, args)
    workers.sort(key=lambda row: int(row["worker_index"]))
    for worker in workers:
        (HERE / f"worker_{worker['worker_index']}_evidence.json").write_text(
            json.dumps(worker, indent=2) + "\n"
        )
    task_results = sorted(
        [row for worker in workers for row in worker["results"]],
        key=lambda row: int(row["task"]),
    )
    finalists = [row["finalist"] for row in task_results if row["finalist"] is not None]
    gain = sum(float(row["projected_gain"]) for row in finalists)
    payload = {
        "lane": rel(HERE),
        "authority": inventory["authority"],
        "workers_requested": WORKERS,
        "worker_pids": [int(worker["pid"]) for worker in workers],
        "worker_groups": inventory["worker_groups"],
        "tasks": task_results,
        "summary": {
            "tasks": len(TASKS),
            "generated_candidates": inventory["generated_candidate_count"],
            "preliminary_eligible": sum(row["preliminary_eligible"] for row in task_results),
            "fully_audited": sum(len(row["audits"]) for row in task_results),
            "admitted": len(finalists),
            "exact_authority_equivalent": sum(
                row["classification"] == "EXACT_AUTHORITY_EQUIVALENT" for row in finalists
            ),
            "policy90_nonexact": sum(
                row["classification"] == "POLICY90_NONEXACT" for row in finalists
            ),
            "projected_gain": gain,
            "projected_lb_if_all_hold": AUTHORITY_LB + gain,
        },
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "root/all_scores/others unchanged; lane only",
    }
    (HERE / "evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "authority": inventory["authority"],
        "admission_policy": "EXACT_AUTHORITY_EQUIVALENT or POLICY90; not an LB guarantee until measured",
        "candidates": [
            {
                "task": row["task"],
                "path": row["saved_path"],
                "sha256": row["saved_sha256"],
                "authority_cost": next(
                    result["authority"]["profile"]["cost"]
                    for result in task_results if result["task"] == row["task"]
                ),
                "candidate_cost": row["profile"]["cost"],
                "projected_gain": row["projected_gain"],
                "classification": row["classification"],
                "plans": row["build"]["plans"],
            }
            for row in finalists
        ],
        "projected_gain": gain,
        "projected_lb_if_all_hold": AUTHORITY_LB + gain,
        "root_all_scores_others_modified": False,
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# scalar / initializer alias 10-worker scan",
        "",
        f"Authority: `submission_base_8012.15.zip` (`{AUTHORITY_SHA256}`)",
        "",
        f"Workers: PIDs {payload['worker_pids']}; generated loads {loads}.",
        "",
        "| task | authority | candidate | gain | classification |",
        "|---:|---:|---:|---:|---|",
    ]
    for result in task_results:
        row = result["finalist"]
        if row is None:
            report.append(
                f"| {result['task']:03d} | {result['authority']['profile']['cost']} | — | — | NO_ADMISSION |"
            )
        else:
            report.append(
                f"| {result['task']:03d} | {result['authority']['profile']['cost']} | "
                f"{row['profile']['cost']} | +{row['projected_gain']:.6f} | {row['classification']} |"
            )
    report.extend(
        [
            "",
            f"Conditional gain: **+{gain:.6f}**",
            f"Conditional projected LB: **{AUTHORITY_LB + gain:.6f}**",
            "",
            "Every admitted exact candidate has a per-replacement mathematical proof and",
            "byte-identical raw outputs to its authority across known plus two fresh 2000-case",
            "seeds under all four ORT configurations. Protected weight/shape initializers",
            "remain byte-identical and present.",
            "",
            "The root authority, all_scores.csv, and others/ were not modified.",
        ]
    )
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n")
    if not finalists:
        readme = HERE / "candidates/README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text("# No admitted candidates\n\nNo candidate passed every required gate.\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
