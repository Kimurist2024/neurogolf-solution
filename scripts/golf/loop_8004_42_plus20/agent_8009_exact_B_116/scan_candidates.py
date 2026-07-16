#!/usr/bin/env python3
"""Inventory 8009.46 lane-B targets and emit only strict-lower exact rewrites."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxoptimizer
import onnxsim
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (178, 228, 234, 264, 325, 344, 357, 387, 388, 392, 397, 398)
BASELINE_DIR = HERE / "baseline"
CANDIDATE_DIR = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


OPT_PASSES = (
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "extract_constant_to_initializer",
    "eliminate_consecutive_idempotent_ops",
    "eliminate_if_with_const_cond",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_shape_gather",
    "eliminate_slice_after_shape",
    "eliminate_nop_transpose",
    "fuse_consecutive_concats",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_concat_into_reshape",
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
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return default


def tensor_key(item: onnx.TensorProto) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(numpy_helper.to_array(item))
    return (array.dtype.str, tuple(array.shape), array.tobytes())


def node_key(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.CopyFrom(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def official_cost(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="exactB116_", dir="/tmp") as work:
        path = Path(work) / f"{label}.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    row: dict[str, Any] = {
        "full_check": False,
        "strict_data_prop": False,
        "all_node_outputs_static_positive": False,
        "standard_domains": False,
        "finite_initializers": False,
        "conv_bias_ub0": False,
        "reasons": [],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["reasons"].append("checker")
        row["checker_error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["reasons"].append("strict_data_prop")
        row["strict_error"] = f"{type(exc).__name__}: {exc}"
        return row
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    nonstatic: list[str] = []
    for node in inferred.graph.node:
        for name in node.output:
            if not name:
                continue
            value = typed.get(name)
            shape = dims(value) if value is not None else []
            if value is None or any(dim is None or dim <= 0 for dim in shape):
                nonstatic.append(name)
    standard = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    finite = True
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            finite = False
    nested = sum(
        attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    )
    try:
        findings = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        findings = [{"check_error": f"{type(exc).__name__}: {exc}"}]
    row.update(
        {
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "value_info_count": len(model.graph.value_info),
            "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "standard_domains": standard,
            "finite_initializers": finite,
            "nested_graph_count": nested,
            "function_count": len(model.functions),
            "sparse_initializer_count": len(model.graph.sparse_initializer),
            "nonstatic_node_outputs": sorted(set(nonstatic)),
            "all_node_outputs_static_positive": not nonstatic,
            "declared_output_shapes": {value.name: dims(value) for value in inferred.graph.output},
            "conv_bias_findings": findings,
            "conv_bias_ub0": not findings,
        }
    )
    if nonstatic:
        row["reasons"].append("nonstatic")
    if not standard:
        row["reasons"].append("nonstandard_domain")
    if not finite:
        row["reasons"].append("nonfinite_initializer")
    if nested or model.functions or model.graph.sparse_initializer:
        row["reasons"].append("nested_function_sparse")
    if findings:
        row["reasons"].append("conv_bias_ub")
    row["pass"] = not row["reasons"]
    return row


def graph_inventory(model: onnx.ModelProto) -> dict[str, Any]:
    used = Counter(value for node in model.graph.node for value in node.input if value)
    outputs = {value.name for value in model.graph.output}
    needed = set(outputs)
    live: set[int] = set()
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(name and name in needed for name in node.output):
            live.add(index)
            needed.update(name for name in node.input if name)
    aliases: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        aliases[tensor_key(item)].append(item.name)
    node_aliases: dict[bytes, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        node_aliases[node_key(node)].append(index)
    return {
        "nodes": [
            {
                "index": index,
                "name": node.name,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for index, node in enumerate(model.graph.node)
        ],
        "initializers": [
            {
                "name": item.name,
                "dtype": TensorProto.DataType.Name(item.data_type),
                "shape": list(item.dims),
                "elements": int(np.asarray(numpy_helper.to_array(item)).size),
            }
            for item in model.graph.initializer
        ],
        "dead_node_indices": [index for index in range(len(model.graph.node)) if index not in live],
        "unused_initializers": [item.name for item in model.graph.initializer if item.name not in needed],
        "initializer_alias_groups": [names for names in aliases.values() if len(names) > 1],
        "duplicate_node_groups": [indices for indices in node_aliases.values() if len(indices) > 1],
        "unused_optional_outputs": [
            {"node": index, "output_index": output_index, "name": name}
            for index, node in enumerate(model.graph.node)
            for output_index, name in enumerate(node.output)
            if output_index > 0 and name and name not in outputs and used[name] == 0
        ],
    }


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def prune_unused_initializers(model: onnx.ModelProto) -> None:
    used = {value for node in model.graph.node for value in node.input if value}
    used.update(value.name for value in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def clean_value_info(model: onnx.ModelProto, removed: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def transform_alias_initializers(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    canonical: dict[tuple[str, tuple[int, ...], bytes], str] = {}
    replacements: dict[str, str] = {}
    for item in candidate.graph.initializer:
        key = tensor_key(item)
        if key in canonical:
            replacements[item.name] = canonical[key]
        else:
            canonical[key] = item.name
    if not replacements:
        return None
    for node in candidate.graph.node:
        for index, value in enumerate(node.input):
            if value in replacements:
                node.input[index] = replacements[value]
    kept = [item for item in candidate.graph.initializer if item.name not in replacements]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    clean_value_info(candidate, set(replacements))
    return candidate


def transform_dead(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    needed = {value.name for value in candidate.graph.output}
    live: list[int] = []
    for index in range(len(candidate.graph.node) - 1, -1, -1):
        node = candidate.graph.node[index]
        if any(name and name in needed for name in node.output):
            live.append(index)
            needed.update(name for name in node.input if name)
    live.reverse()
    unused = {item.name for item in candidate.graph.initializer if item.name not in needed}
    dead_outputs = {
        name
        for index, node in enumerate(candidate.graph.node)
        if index not in set(live)
        for name in node.output
        if name
    }
    if len(live) == len(candidate.graph.node) and not unused:
        return None
    nodes = [candidate.graph.node[index] for index in live]
    inits = [item for item in candidate.graph.initializer if item.name in needed]
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(inits)
    clean_value_info(candidate, dead_outputs | unused)
    return candidate


def transform_cse(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    outputs = {value.name for value in candidate.graph.output}
    canonical: dict[bytes, int] = {}
    remove: list[int] = []
    removed_names: set[str] = set()
    for index, node in enumerate(candidate.graph.node):
        key = node_key(node)
        if key not in canonical:
            canonical[key] = index
            continue
        first = candidate.graph.node[canonical[key]]
        if len(first.output) != len(node.output) or any(name in outputs for name in node.output if name):
            continue
        pairs = [(old, new) for old, new in zip(node.output, first.output) if old and new]
        if len(pairs) != len([name for name in node.output if name]):
            continue
        for old, new in pairs:
            replace_uses(candidate, old, new)
            removed_names.add(old)
        remove.append(index)
    if not remove:
        return None
    nodes = [node for index, node in enumerate(candidate.graph.node) if index not in set(remove)]
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    prune_unused_initializers(candidate)
    clean_value_info(candidate, removed_names)
    return candidate


def transform_optional(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    used = Counter(value for node in candidate.graph.node for value in node.input if value)
    outputs = {value.name for value in candidate.graph.output}
    removed: set[str] = set()
    for node in candidate.graph.node:
        for index in range(1, len(node.output)):
            name = node.output[index]
            if name and name not in outputs and used[name] == 0:
                removed.add(name)
                node.output[index] = ""
    if not removed:
        return None
    clean_value_info(candidate, removed)
    return candidate


def transform_manual_noops(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception:  # noqa: BLE001
        return None
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    inits = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    graph_outputs = {value.name for value in candidate.graph.output}
    remove: list[int] = []
    removed_outputs: set[str] = set()
    for index, node in enumerate(candidate.graph.node):
        if len(node.output) != 1 or not node.output[0] or node.output[0] in graph_outputs:
            continue
        replacement: str | None = None
        if node.op_type == "Identity" and node.input:
            replacement = node.input[0]
        elif node.op_type == "Cast" and node.input:
            source = typed.get(node.input[0])
            target = typed.get(node.output[0])
            if source is not None and target is not None and source.type.tensor_type.elem_type == target.type.tensor_type.elem_type:
                replacement = node.input[0]
        elif node.op_type == "CastLike" and len(node.input) >= 2:
            source = typed.get(node.input[0])
            like = typed.get(node.input[1])
            if source is not None and like is not None and source.type.tensor_type.elem_type == like.type.tensor_type.elem_type:
                replacement = node.input[0]
        elif node.op_type == "Reshape" and len(node.input) >= 2 and node.input[1] in inits:
            source = typed.get(node.input[0])
            target = typed.get(node.output[0])
            literal = [int(value) for value in inits[node.input[1]].reshape(-1)]
            if source is not None and target is not None and dims(source) == dims(target) == literal:
                replacement = node.input[0]
        elif node.op_type == "Transpose" and node.input:
            source = typed.get(node.input[0])
            perm = attr(node, "perm")
            if source is not None and perm is not None and list(perm) == list(range(len(dims(source)))):
                replacement = node.input[0]
        elif node.op_type == "Concat":
            actual = [name for name in node.input if name]
            if len(actual) == 1:
                replacement = actual[0]
        elif node.op_type in {"Add", "Sub", "Mul", "Div"} and len(node.input) == 2:
            left, right = node.input
            left_const = inits.get(left)
            right_const = inits.get(right)
            if node.op_type == "Add":
                if left_const is not None and np.all(left_const == 0):
                    replacement = right
                elif right_const is not None and np.all(right_const == 0):
                    replacement = left
            elif node.op_type == "Sub" and right_const is not None and np.all(right_const == 0):
                replacement = left
            elif node.op_type == "Mul":
                if left_const is not None and np.all(left_const == 1):
                    replacement = right
                elif right_const is not None and np.all(right_const == 1):
                    replacement = left
            elif node.op_type == "Div" and right_const is not None and np.all(right_const == 1):
                replacement = left
        if replacement is None:
            continue
        source = typed.get(replacement)
        target = typed.get(node.output[0])
        if source is None or target is None:
            continue
        if dims(source) != dims(target) or source.type.tensor_type.elem_type != target.type.tensor_type.elem_type:
            continue
        replace_uses(candidate, node.output[0], replacement)
        removed_outputs.add(node.output[0])
        remove.append(index)
    if not remove:
        return None
    nodes = [node for index, node in enumerate(candidate.graph.node) if index not in set(remove)]
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    prune_unused_initializers(candidate)
    clean_value_info(candidate, removed_outputs)
    return candidate


def transform_scalar_broadcast(model: onnx.ModelProto) -> onnx.ModelProto | None:
    candidate = copy.deepcopy(model)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in candidate.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
    safe = {"Add", "Sub", "Mul", "Div", "Pow", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Where", "And", "Or", "Xor"}
    changed = False
    for item in candidate.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        if array.size <= 1 or not consumers[item.name] or not all(node.op_type in safe for node in consumers[item.name]):
            continue
        if not np.all(array == array.reshape(-1)[0]):
            continue
        scalar = np.asarray(array.reshape(-1)[0], dtype=array.dtype)
        item.CopyFrom(numpy_helper.from_array(scalar, item.name))
        changed = True
    return candidate if changed else None


def generate_variants(model: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto]]:
    variants: list[tuple[str, onnx.ModelProto]] = []
    manual: tuple[tuple[str, Callable[[onnx.ModelProto], onnx.ModelProto | None]], ...] = (
        ("alias_initializers", transform_alias_initializers),
        ("dead_unused", transform_dead),
        ("common_subexpression", transform_cse),
        ("optional_outputs", transform_optional),
        ("manual_noops", transform_manual_noops),
        ("scalar_broadcast", transform_scalar_broadcast),
    )
    for label, function in manual:
        try:
            value = function(model)
            if value is not None:
                variants.append((label, value))
        except Exception:  # noqa: BLE001
            continue
    for name in OPT_PASSES:
        try:
            variants.append((f"opt_{name}", onnxoptimizer.optimize(copy.deepcopy(model), [name], fixed_point=True)))
        except Exception:  # noqa: BLE001
            continue
    try:
        variants.append(("opt_conservative_all", onnxoptimizer.optimize(copy.deepcopy(model), list(OPT_PASSES), fixed_point=True)))
    except Exception:  # noqa: BLE001
        pass
    try:
        simplified, check = onnxsim.simplify(copy.deepcopy(model), check_n=0, perform_optimization=True)
        if check:
            variants.append(("onnxsim", simplified))
    except Exception:  # noqa: BLE001
        pass
    return variants


def main() -> int:
    authority_data = AUTHORITY.read_bytes()
    if digest(authority_data) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority hash changed")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
    }
    candidates_report: dict[str, Any] = {"tasks": {}, "winner_candidates": []}
    with zipfile.ZipFile(AUTHORITY) as archive:
        baseline_data = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}

    for task in TASKS:
        data = baseline_data[task]
        path = BASELINE_DIR / f"task{task:03d}.onnx"
        path.write_bytes(data)
        model = onnx.load_model_from_string(data)
        base_cost = official_cost(data, f"task{task:03d}_base")
        base_static = structural(model)
        inventory["tasks"][str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "official_cost": base_cost,
            "structural": base_static,
            "graph": graph_inventory(model),
        }
        rows: list[dict[str, Any]] = []
        seen = {digest(data)}
        for label, candidate in generate_variants(model):
            candidate_data = candidate.SerializeToString()
            candidate_sha = digest(candidate_data)
            if candidate_sha in seen:
                continue
            seen.add(candidate_sha)
            static = structural(candidate)
            row: dict[str, Any] = {
                "label": label,
                "sha256": candidate_sha,
                "serialized_bytes": len(candidate_data),
                "structural": static,
                "status": "structural_reject",
            }
            if not static.get("pass", False):
                rows.append(row)
                continue
            profile = official_cost(candidate_data, f"task{task:03d}_{label}")
            row["official_cost"] = profile
            if profile["cost"] >= base_cost["cost"]:
                row["status"] = "not_strict_lower"
                rows.append(row)
                continue
            out = CANDIDATE_DIR / f"task{task:03d}_{label}_{candidate_sha[:12]}.onnx"
            out.write_bytes(candidate_data)
            row.update(
                {
                    "status": "strict_lower_needs_runtime",
                    "path": str(out.relative_to(ROOT)),
                    "cost_reduction": base_cost["cost"] - profile["cost"],
                }
            )
            candidates_report["winner_candidates"].append(
                {"task": task, "label": label, "path": row["path"], "sha256": candidate_sha, "official_cost": profile}
            )
            rows.append(row)
        candidates_report["tasks"][str(task)] = {
            "baseline_sha256": digest(data),
            "baseline_cost": base_cost,
            "variants_unique": len(rows),
            "strict_lower": [row for row in rows if row["status"] == "strict_lower_needs_runtime"],
            "rows": rows,
        }
        print(
            f"task{task:03d} sha={digest(data)[:12]} cost={base_cost['cost']} "
            f"nodes={len(model.graph.node)} variants={len(rows)} lower={len(candidates_report['tasks'][str(task)]['strict_lower'])}",
            flush=True,
        )
    inventory["summary"] = {"tasks": len(TASKS), "authority_hash_ok": True}
    candidates_report["summary"] = {
        "tasks": len(TASKS),
        "unique_variants": sum(row["variants_unique"] for row in candidates_report["tasks"].values()),
        "strict_lower_candidates": len(candidates_report["winner_candidates"]),
    }
    (HERE / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (HERE / "candidate_scan.json").write_text(json.dumps(candidates_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(candidates_report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
