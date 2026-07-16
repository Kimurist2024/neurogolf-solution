#!/usr/bin/env python3
"""Three-worker exact rewrite census for the immutable 8012.15 authority.

The worker stage is deliberately fail closed.  It only emits a model when a
mechanical rewrite passes the full checker, strict data-propagating shape
inference, the canonical I/O contract, raw equality with the authority on all
known cases, and the official profiler reports a strictly lower cost.
"""

from __future__ import annotations

import copy
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
WORKERS = 3
MIN_COST = 100
MAX_COST = 500

KNOWN_BLACK = {70, 134, 202, 343}
PRIVATE_ZERO = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
ALREADY_ADMITTED = {12, 23, 161, 175, 354, 355}
EXCLUDED = PRIVATE_ZERO | KNOWN_BLACK | ALREADY_ADMITTED

SAFE_OPTIMIZER_PASSES = (
    "extract_constant_to_initializer",
    "eliminate_consecutive_idempotent_ops",
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_shape_gather",
    "eliminate_slice_after_shape",
    "eliminate_nop_transpose",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "eliminate_nop_reshape",
    "eliminate_nop_with_unit",
    "eliminate_common_subexpression",
    "fuse_consecutive_unsqueezes",
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_shape_op",
    "fuse_consecutive_slices",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "fuse_add_bias_into_conv",
    "fuse_bn_into_conv",
    "fuse_pad_into_conv",
    "fuse_pad_into_pool",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_transpose_into_gemm",
    "replace_einsum_with_matmul",
    "fuse_concat_into_reshape",
    "adjust_slice_and_matmul",
)

ELEMENTWISE = {
    "Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "GreaterOrEqual",
    "Less", "LessOrEqual", "Max", "Min", "And", "Or", "Xor", "BitwiseAnd",
    "BitwiseOr", "BitwiseXor", "Mod", "BitShift", "Sum", "Clip", "Where",
}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    f"restart415_exact_{os.getpid()}",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/scan_and_build.py",
)
NOOPS = load_module(
    f"restart415_noops_{os.getpid()}",
    ROOT / "scripts/golf/agent_exact_noop_scan_285/scan.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def costs() -> dict[int, int]:
    result: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            result[int(row["task"].removeprefix("task"))] = int(row["cost"])
    return result


def canonical_io(model: onnx.ModelProto) -> tuple[bool, str]:
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return False, "io_count"
    for value, name in ((model.graph.input[0], "input"), (model.graph.output[0], "output")):
        if value.name != name:
            return False, f"name:{value.name}"
        tensor = value.type.tensor_type
        dims = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        if tensor.elem_type != TensorProto.FLOAT or dims != (1, 10, 30, 30):
            return False, f"contract:{name}:{tensor.elem_type}:{dims}"
    return True, "pass"


def init_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result = {}
    for item in model.graph.initializer:
        try:
            result[item.name] = np.asarray(numpy_helper.to_array(item))
        except Exception:
            pass
    return result


def uses(model: onnx.ModelProto) -> dict[str, list[tuple[int, str, int]]]:
    result: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                result[name].append((node_index, node.op_type, input_index))
    return result


def prune_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def alias_variants(base: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    """Build conservative dtype/value aliases for elementwise and CastLike use."""
    arrays = init_arrays(base)
    use_map = uses(base)
    variants: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    io = {item.name for item in [*base.graph.input, *base.graph.output]}

    def allowed_target(name: str) -> bool:
        rows = use_map.get(name, [])
        if not rows or name in io:
            return False
        for _, op, index in rows:
            if op not in ELEMENTWISE:
                return False
            if op == "Where" and index == 0:
                return False
            if op == "Clip" and index == 0:
                return False
        return True

    for target, target_array in arrays.items():
        if not allowed_target(target) or target_array.size == 0:
            continue
        flat = target_array.reshape(-1)
        if not np.all(flat == flat[0]):
            continue
        sources = [
            name for name, array in arrays.items()
            if name != target and array.size == 1 and array.dtype == target_array.dtype
            and np.array_equal(array.reshape(-1), flat[:1]) and use_map.get(name)
        ]
        if not sources:
            continue
        source = sorted(sources)[0]
        model = copy.deepcopy(base)
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == target:
                    node.input[index] = source
        removed = prune_initializers(model)
        variants.append((
            f"broadcast_alias_{target}", model,
            {"kind": "uniform_elementwise_broadcast", "target": target, "source": source,
             "target_shape": list(target_array.shape), "source_shape": list(arrays[source].shape),
             "removed_initializers": removed},
        ))

    for target, target_array in arrays.items():
        rows = use_map.get(target, [])
        if not rows or target in io or not all(op == "CastLike" and index == 1 for _, op, index in rows):
            continue
        sources = [
            name for name, array in arrays.items()
            if name != target and array.dtype == target_array.dtype and use_map.get(name)
        ]
        if not sources:
            continue
        source = min(sources, key=lambda name: (arrays[name].size, name))
        model = copy.deepcopy(base)
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == target:
                    node.input[index] = source
        removed = prune_initializers(model)
        variants.append((
            f"castlike_alias_{target}", model,
            {"kind": "castlike_dtype_template", "target": target, "source": source,
             "dtype": str(target_array.dtype), "removed_initializers": removed},
        ))
    return variants


def optional_output_variant(base: onnx.ModelProto) -> tuple[str, onnx.ModelProto, dict[str, Any]] | None:
    used = {name for node in base.graph.node for name in node.input if name}
    used.update(item.name for item in base.graph.output)
    model = copy.deepcopy(base)
    removed = []
    for node_index, node in enumerate(model.graph.node):
        for output_index in range(1, len(node.output)):
            name = node.output[output_index]
            if name and name not in used:
                removed.append({"node_index": node_index, "output_index": output_index,
                                "op": node.op_type, "name": name})
                node.output[output_index] = ""
    if not removed:
        return None
    EXACT.remove_value_info(model, {row["name"] for row in removed})
    return "optional_outputs", model, {"kind": "unused_secondary_outputs", "removed": removed}


def einsum_unit_variant(base: onnx.ModelProto) -> tuple[str, onnx.ModelProto, dict[str, Any]] | None:
    """Remove multiplicative size-one +1 operands that cannot supply an output axis."""
    model = copy.deepcopy(base)
    arrays = init_arrays(model)
    actions = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation_attr = next((attr for attr in node.attribute if attr.name == "equation"), None)
        if equation_attr is None:
            continue
        equation = equation_attr.s.decode()
        if "->" not in equation:
            continue
        left, output_term = equation.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input):
            continue
        remove = []
        for index, (term, name) in enumerate(zip(terms, node.input)):
            array = arrays.get(name)
            if array is None or array.size != 1 or not np.all(array == 1):
                continue
            other_labels = set("".join(terms[:index] + terms[index + 1:]).replace(".", ""))
            output_labels = set(output_term.replace(".", ""))
            # An output label must still be sourced by another operand.
            if not output_labels.issubset(other_labels):
                continue
            remove.append(index)
        if not remove or len(remove) == len(terms):
            continue
        kept_terms = [term for index, term in enumerate(terms) if index not in set(remove)]
        kept_inputs = [name for index, name in enumerate(node.input) if index not in set(remove)]
        removed_inputs = [name for index, name in enumerate(node.input) if index in set(remove)]
        equation_attr.s = (",".join(kept_terms) + "->" + output_term).encode()
        del node.input[:]
        node.input.extend(kept_inputs)
        actions.append({"node_index": node_index, "removed_inputs": removed_inputs,
                        "old_equation": equation, "new_equation": equation_attr.s.decode()})
    if not actions:
        return None
    removed_initializers = prune_initializers(model)
    return "einsum_unit_operands", model, {
        "kind": "remove_size1_unit_einsum_operands", "actions": actions,
        "removed_initializers": removed_initializers,
    }


def einsum_precontract_variant(base: onnx.ModelProto) -> tuple[str, onnx.ModelProto, dict[str, Any]] | None:
    """Pre-sum initializer-only labels that occur nowhere else in an Einsum."""
    model = copy.deepcopy(base)
    use_map = uses(model)
    init_map = {item.name: item for item in model.graph.initializer}
    actions = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation_attr = next((attr for attr in node.attribute if attr.name == "equation"), None)
        if equation_attr is None or "->" not in equation_attr.s.decode():
            continue
        equation = equation_attr.s.decode()
        left, output_term = equation.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input) or any("..." in term for term in terms):
            continue
        counts = defaultdict(int)
        for term in terms:
            for label in term:
                counts[label] += 1
        changed_terms = list(terms)
        for input_index, (term, name) in enumerate(zip(terms, node.input)):
            tensor = init_map.get(name)
            if tensor is None or len(use_map.get(name, [])) != 1 or len(term) != len(tensor.dims):
                continue
            removable = [
                axis for axis, label in enumerate(term)
                if counts[label] == 1 and label not in output_term
            ]
            if not removable:
                continue
            array = np.asarray(numpy_helper.to_array(tensor))
            # Preserve the initializer dtype instead of NumPy's default integer
            # widening. Axes are reduced together to avoid index shifts.
            reduced = np.sum(array, axis=tuple(removable), dtype=array.dtype)
            reduced = np.ascontiguousarray(reduced)
            replacement = numpy_helper.from_array(reduced, name)
            tensor.CopyFrom(replacement)
            changed_terms[input_index] = "".join(
                label for axis, label in enumerate(term) if axis not in set(removable)
            )
            actions.append({
                "node_index": node_index, "input_index": input_index, "initializer": name,
                "removed_axes": removable, "old_term": term,
                "new_term": changed_terms[input_index], "old_shape": list(array.shape),
                "new_shape": list(reduced.shape),
            })
        if changed_terms != terms:
            equation_attr.s = (",".join(changed_terms) + "->" + output_term).encode()
            actions[-1]["old_equation"] = equation
            actions[-1]["new_equation"] = equation_attr.s.decode()
    if not actions:
        return None
    return "einsum_precontract", model, {"kind": "initializer_unique_axis_presum", "actions": actions}


def einsum_same_term_fusion_variant(base: onnx.ModelProto) -> tuple[str, onnx.ModelProto, dict[str, Any]] | None:
    """Multiply exclusive same-subscript initializer operands offline."""
    model = copy.deepcopy(base)
    use_map = uses(model)
    init_map = {item.name: item for item in model.graph.initializer}
    actions = []
    removed_names: set[str] = set()
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation_attr = next((attr for attr in node.attribute if attr.name == "equation"), None)
        if equation_attr is None or "->" not in equation_attr.s.decode():
            continue
        equation = equation_attr.s.decode()
        left, output_term = equation.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input):
            continue
        groups: dict[str, list[int]] = defaultdict(list)
        for index, (term, name) in enumerate(zip(terms, node.input)):
            tensor = init_map.get(name)
            if tensor is not None and len(use_map.get(name, [])) == 1 and name not in removed_names:
                groups[term].append(index)
        remove_indices: set[int] = set()
        for term, indices in groups.items():
            if len(indices) < 2:
                continue
            first_index = indices[0]
            first_name = node.input[first_index]
            first = np.asarray(numpy_helper.to_array(init_map[first_name]))
            for other_index in indices[1:]:
                other_name = node.input[other_index]
                other = np.asarray(numpy_helper.to_array(init_map[other_name]))
                if first.shape != other.shape or first.dtype != other.dtype:
                    continue
                combined = np.multiply(first, other, dtype=first.dtype)
                if not np.all(np.isfinite(combined)):
                    continue
                first = np.ascontiguousarray(combined)
                remove_indices.add(other_index)
                removed_names.add(other_name)
                actions.append({"node_index": node_index, "term": term,
                                "kept": first_name, "fused": other_name})
            init_map[first_name].CopyFrom(numpy_helper.from_array(first, first_name))
        if remove_indices:
            kept_terms = [term for index, term in enumerate(terms) if index not in remove_indices]
            kept_inputs = [name for index, name in enumerate(node.input) if index not in remove_indices]
            equation_attr.s = (",".join(kept_terms) + "->" + output_term).encode()
            del node.input[:]
            node.input.extend(kept_inputs)
            actions[-1]["old_equation"] = equation
            actions[-1]["new_equation"] = equation_attr.s.decode()
    if not actions:
        return None
    removed_initializers = prune_initializers(model)
    return "einsum_same_term_fusion", model, {
        "kind": "exclusive_same_subscript_initializer_product", "actions": actions,
        "removed_initializers": removed_initializers,
    }


def centercroppad_noop_variant(base: onnx.ModelProto) -> tuple[str, onnx.ModelProto, dict[str, Any]] | None:
    """Bypass CenterCropPad only when its literal target equals source dimensions."""
    model = copy.deepcopy(base)
    actions = []
    while True:
        try:
            inferred = onnx.shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=True
            )
        except Exception:
            break
        values = {
            item.name: item for item in
            [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        }
        arrays = init_arrays(model)
        chosen = None
        for index, node in enumerate(model.graph.node):
            if node.op_type != "CenterCropPad" or len(node.input) != 2 or len(node.output) != 1:
                continue
            source_value = values.get(node.input[0])
            target = arrays.get(node.input[1])
            if source_value is None or target is None:
                continue
            source_dims = [int(dim.dim_value) for dim in source_value.type.tensor_type.shape.dim]
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            axes = [int(value) for value in attrs.get("axes", range(len(source_dims)))]
            axes = [axis + len(source_dims) if axis < 0 else axis for axis in axes]
            target_dims = [int(value) for value in target.reshape(-1)]
            if len(axes) != len(target_dims) or any(axis >= len(source_dims) for axis in axes):
                continue
            if target_dims != [source_dims[axis] for axis in axes]:
                continue
            if not NOOPS.can_bypass(model, node.output[0], node.input[0]):
                continue
            chosen = (index, node.input[0], "literal CenterCropPad target equals source dimensions")
            break
        if chosen is None:
            break
        action = NOOPS.bypass_node(model, *chosen)
        if action is None:
            break
        actions.append(action)
        actions.extend(NOOPS.cleanup_dead(model))
    if not actions:
        return None
    return "centercroppad_noops", model, {"kind": "literal_noop_centercroppad", "actions": actions}


def optimizer_variants(base: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    result = []
    for name in SAFE_OPTIMIZER_PASSES:
        try:
            model = onnxoptimizer.optimize(copy.deepcopy(base), [name], fixed_point=True)
        except Exception:
            continue
        if model.SerializeToString() != base.SerializeToString():
            result.append((f"opt_{name}", model, {"kind": "onnxoptimizer", "passes": [name]}))
    groups = (
        ("opt_cleanup", ("eliminate_deadend", "eliminate_identity", "eliminate_unused_initializer", "eliminate_duplicate_initializer")),
        ("opt_shape", ("eliminate_shape_gather", "eliminate_slice_after_shape", "eliminate_shape_op", "eliminate_nop_reshape", "eliminate_nop_expand")),
        ("opt_all_safe", SAFE_OPTIMIZER_PASSES),
    )
    for label, passes in groups:
        try:
            model = onnxoptimizer.optimize(copy.deepcopy(base), list(passes), fixed_point=True)
        except Exception:
            continue
        if model.SerializeToString() != base.SerializeToString():
            result.append((label, model, {"kind": "onnxoptimizer", "passes": list(passes)}))
    return result


def structural_variants(base: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    result = []
    for profile in NOOPS.PROFILES:
        try:
            model, actions = NOOPS.transform(base, profile)
        except Exception:
            continue
        if actions:
            result.append((f"mechanical_{profile}", model, {"kind": "mechanical", "actions": actions}))
    result.extend(alias_variants(base))
    optional = optional_output_variant(base)
    if optional is not None:
        result.append(optional)
    einsum_unit = einsum_unit_variant(base)
    if einsum_unit is not None:
        result.append(einsum_unit)
    einsum_precontract = einsum_precontract_variant(base)
    if einsum_precontract is not None:
        result.append(einsum_precontract)
    einsum_fusion = einsum_same_term_fusion_variant(base)
    if einsum_fusion is not None:
        result.append(einsum_fusion)
    center_noop = centercroppad_noop_variant(base)
    if center_noop is not None:
        result.append(center_noop)
    result.extend(optimizer_variants(base))
    return result


def worker(worker_id: int, tasks: list[int], official_costs: dict[int, int]) -> dict[str, Any]:
    candidate_dir = HERE / "candidates" / f"worker_{worker_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baseline_failures = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            base_audit, _ = EXACT.structural_audit(base)
            ok, io_reason = canonical_io(base)
            if not ok:
                baseline_failures.append({"task": task, "canonical_io": io_reason})
                continue
            seeds: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
            if base_audit.get("pass"):
                seeds.append(("", base, {}))
            else:
                baseline_failures.append({"task": task, "audit": base_audit})
                # Advisory value_info is not computational payload.  Clearing
                # it can turn a contradictory legacy annotation into a strict,
                # truthfully inferred graph without changing any operation.
                normalized = copy.deepcopy(base)
                removed_value_info = [item.name for item in normalized.graph.value_info]
                del normalized.graph.value_info[:]
                normalized_audit, _ = EXACT.structural_audit(normalized)
                if normalized_audit.get("pass"):
                    seeds.append((
                        "normalized_", normalized,
                        {"kind": "clear_advisory_value_info", "removed_count": len(removed_value_info)},
                    ))
            seen: set[str] = set()
            variants: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
            for prefix, seed, seed_detail in seeds:
                if prefix:
                    variants.append((prefix.removesuffix("_"), seed, seed_detail))
                for label, candidate, detail in structural_variants(seed):
                    variants.append((prefix + label, candidate, {"seed": seed_detail, **detail}))
            for label, candidate, detail in variants:
                data = candidate.SerializeToString()
                digest = sha256(data)
                if digest == sha256(base_data) or digest in seen:
                    continue
                seen.add(digest)
                audit, _ = EXACT.structural_audit(candidate)
                row: dict[str, Any] = {
                    "task": task, "label": label, "authority_sha256": sha256(base_data),
                    "candidate_sha256": digest, "detail": detail,
                    "authority_static": {key: base_audit.get(key) for key in ("memory", "params", "cost")},
                    "candidate_static": {key: audit.get(key) for key in ("memory", "params", "cost")},
                    "authority_official_cost": official_costs[task],
                    "status": "structural_reject",
                }
                if not audit.get("pass"):
                    row["structural"] = audit
                    rows.append(row)
                    continue
                ok, reason = canonical_io(candidate)
                if not ok:
                    row["status"] = "canonical_io_reject"
                    row["canonical_io"] = reason
                    rows.append(row)
                    continue
                static_reference = (
                    int(base_audit["cost"]) if base_audit.get("pass")
                    else official_costs[task]
                )
                if int(audit["cost"]) >= static_reference:
                    row["status"] = "not_static_lower"
                    rows.append(row)
                    continue
                if not scoring.outputs_bit_identical(base, candidate, task):
                    row["status"] = "known_raw_mismatch"
                    rows.append(row)
                    continue
                with tempfile.TemporaryDirectory(prefix=f"restart415_w{worker_id}_t{task:03d}_", dir="/tmp") as workdir:
                    profile = scoring.score_and_verify(copy.deepcopy(candidate), task, workdir, label, require_correct=False)
                row["official_profile"] = profile
                if profile is None:
                    row["status"] = "official_profile_reject"
                    rows.append(row)
                    continue
                if int(profile["cost"]) >= official_costs[task]:
                    row["status"] = "not_official_lower"
                    rows.append(row)
                    continue
                path = candidate_dir / f"task{task:03d}_{label}_{digest[:12]}.onnx"
                onnx.save(candidate, path)
                row.update({
                    "status": "finalist_known_exact",
                    "path": relative(path),
                    "gain": math.log(official_costs[task] / int(profile["cost"])),
                })
                rows.append(row)
    payload = {
        "worker": worker_id,
        "pid": os.getpid(),
        "tasks": tasks,
        "baseline_failures": baseline_failures,
        "rows": rows,
        "summary": {
            "tasks": len(tasks),
            "baseline_failures": len(baseline_failures),
            "variants": len(rows),
            "finalists_known_exact": sum(row["status"] == "finalist_known_exact" for row in rows),
        },
    }
    (HERE / f"worker_{worker_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def selected_tasks(official_costs: dict[int, int]) -> list[int]:
    return sorted(
        task for task, cost in official_costs.items()
        if MIN_COST <= cost <= MAX_COST and task not in EXCLUDED
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    official_costs = costs()
    tasks = selected_tasks(official_costs)
    partitions = [tasks[index::WORKERS] for index in range(WORKERS)]
    if len(partitions) != 3 or sum(map(len, partitions)) != len(tasks):
        raise RuntimeError("three-worker partition invariant failed")
    if args.worker is not None:
        if not 0 <= args.worker < WORKERS:
            raise RuntimeError("worker must be 0, 1, or 2")
        result = worker(args.worker, partitions[args.worker], official_costs)
        print(json.dumps(result["summary"], indent=2))
        return 0

    processes = []
    handles = []
    for worker_id in range(WORKERS):
        stdout = (HERE / f"worker_{worker_id}.log").open("w")
        handles.append(stdout)
        processes.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(worker_id)],
            cwd=ROOT,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        ))
    failures = []
    for worker_id, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append({"worker": worker_id, "exit_code": code})
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError(f"worker failures: {failures}")
    results = [json.loads((HERE / f"worker_{worker_id}.json").read_text()) for worker_id in range(WORKERS)]
    finalists = [row for result in results for row in result["rows"] if row["status"] == "finalist_known_exact"]
    payload = {
        "authority": {"path": relative(AUTHORITY), "sha256": AUTHORITY_SHA256, "lb": 8012.15},
        "policy": {
            "workers": WORKERS,
            "cost_range": [MIN_COST, MAX_COST],
            "excluded_known_black": sorted(KNOWN_BLACK),
            "excluded_private_zero": sorted(PRIVATE_ZERO),
            "excluded_already_admitted": sorted(ALREADY_ADMITTED),
            "admission_stage": "known raw exact plus official strict-lower; fresh audit still required",
        },
        "tasks": tasks,
        "partitions": partitions,
        "worker_summaries": [item["summary"] for item in results],
        "finalists": finalists,
    }
    (HERE / "scan_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "tasks": len(tasks), "workers": WORKERS,
        "variants": sum(item["summary"]["variants"] for item in results),
        "finalists": len(finalists),
        "finalist_tasks": sorted({row["task"] for row in finalists}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
