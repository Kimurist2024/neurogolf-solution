#!/usr/bin/env python3
"""Build and inventory conservative exact rewrites for the 8003.40 baseline.

This lane is deliberately isolated.  It never updates a submission archive or
any root score file.  The only generated files live below agent_exact_resume.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = ROOT / "submission_base_8003.40.zip"
CANDIDATES = HERE / "candidates"

# Explicitly prohibited by the task instruction.  No candidate is emitted for
# these tasks even when a local algebraic simplification exists.
PRIVATE_ZERO_EXCLUDE = {70}

BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

EXISTING = {
    48: ROOT
    / "scripts/golf/loop_8003_40/agent_exact_scanners/fuse_unique/task048_r01.onnx",
    333: ROOT
    / "scripts/golf/loop_8003_40/agent_exact_scanners/shared_sign_absorption/task333_r01.onnx",
    233: ROOT
    / "scripts/golf/loop_8003_40/duplicate_initializer_candidates/task233.onnx",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parameter_count(model: onnx.ModelProto) -> int:
    return sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer)


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def equation(node: onnx.NodeProto) -> str | None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            return attribute.s.decode()
    return None


def set_equation(node: onnx.NodeProto, value: str) -> None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            attribute.s = value.encode()
            return
    raise ValueError("Einsum equation attribute is missing")


def all_value_uses(model: onnx.ModelProto) -> Counter[str]:
    return Counter(value for node in model.graph.node for value in node.input if value)


def strict_gate(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {
        "checker": "FAIL",
        "strict_shape_inference": "FAIL",
        "static_shapes": False,
        "banned_ops": [],
        "conv_bias_ub": [],
        "errors": [],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["checker"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"checker: {type(exc).__name__}: {exc}")
        return row

    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        row["strict_shape_inference"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"shape: {type(exc).__name__}: {exc}")
        return row

    bad_shapes: list[str] = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(
        inferred.graph.output
    ):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            bad_shapes.append(value.name)
            continue
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.HasField("dim_param") or dim.dim_value <= 0:
                bad_shapes.append(value.name)
                break
    row["static_shapes"] = not bad_shapes
    if bad_shapes:
        row["errors"].append(f"dynamic_or_missing_shapes: {bad_shapes[:20]}")

    bad_ops = []
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            bad_ops.append(node.op_type)
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                bad_ops.append(f"nested:{node.op_type}")
    row["banned_ops"] = sorted(set(bad_ops))

    spec = importlib.util.spec_from_file_location(
        "check_conv_bias", ROOT / "scripts/golf/check_conv_bias.py"
    )
    if spec is None or spec.loader is None:
        row["errors"].append("unable to import check_conv_bias")
    else:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row["conv_bias_ub"] = [list(item) for item in module.check_model(model)]

    row["pass"] = bool(
        row["checker"] == "PASS"
        and row["strict_shape_inference"] == "PASS"
        and row["static_shapes"]
        and not row["banned_ops"]
        and not row["conv_bias_ub"]
    )
    return row


def save_candidate(
    task: int,
    label: str,
    baseline_model: onnx.ModelProto,
    candidate: onnx.ModelProto,
    rewrite: dict[str, object],
) -> dict[str, object]:
    gate = strict_gate(candidate)
    before = parameter_count(baseline_model)
    after = parameter_count(candidate)
    destination = CANDIDATES / f"task{task:03d}_{label}.onnx"
    if gate.get("pass") and after < before and task not in PRIVATE_ZERO_EXCLUDE:
        onnx.save(candidate, destination)
        path: str | None = str(destination.relative_to(ROOT))
        digest: str | None = sha256_path(destination)
    else:
        path = None
        digest = None
    return {
        "task": task,
        "label": label,
        "path": path,
        "sha256": digest,
        "baseline_params": before,
        "candidate_params": after,
        "parameter_reduction": before - after,
        "excluded_private_zero": task in PRIVATE_ZERO_EXCLUDE,
        "rewrite": rewrite,
        "structural_gate": gate,
    }


def initializer_dedup(
    task: int, model: onnx.ModelProto
) -> tuple[onnx.ModelProto | None, dict[str, object]]:
    canonical: dict[bytes, str] = {}
    replacements: dict[str, str] = {}
    kept = []
    for initializer in model.graph.initializer:
        key = tensor_key(initializer)
        if key in canonical:
            replacements[initializer.name] = canonical[key]
        else:
            canonical[key] = initializer.name
            kept.append(initializer)
    if not replacements:
        return None, {"task": task, "replacements": {}}
    candidate = copy.deepcopy(model)
    for node in candidate.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(
        item for item in copy.deepcopy(model).graph.initializer if item.name not in replacements
    )
    return candidate, {"task": task, "replacements": replacements}


def outer_fusions(task: int, model: onnx.ModelProto) -> list[tuple[onnx.ModelProto, dict[str, object]]]:
    initializers = {item.name: item for item in model.graph.initializer}
    uses = all_value_uses(model)
    results: list[tuple[onnx.ModelProto, dict[str, object]]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        eq = equation(node)
        if eq is None or "..." in eq:
            continue
        left, output_term = eq.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input):
            continue
        for first in range(len(node.input)):
            first_name = node.input[first]
            if first_name not in initializers or uses[first_name] != 1:
                continue
            first_term = terms[first]
            if len(set(first_term)) != len(first_term):
                continue
            first_array = numpy_helper.to_array(initializers[first_name])
            if first_array.ndim != len(first_term):
                continue
            for second in range(first + 1, len(node.input)):
                second_name = node.input[second]
                if second_name not in initializers or uses[second_name] != 1:
                    continue
                second_term = terms[second]
                if set(first_term) & set(second_term):
                    continue
                if len(set(second_term)) != len(second_term):
                    continue
                second_array = numpy_helper.to_array(initializers[second_name])
                if second_array.ndim != len(second_term):
                    continue
                fused_size = first_array.size * second_array.size
                if fused_size >= first_array.size + second_array.size:
                    continue
                fused = np.multiply.outer(first_array, second_array)
                candidate = copy.deepcopy(model)
                candidate_node = candidate.graph.node[node_index]
                fused_name = f"__exact_outer_{first_name}_{second_name}"
                candidate_node.input[first] = fused_name
                del candidate_node.input[second]
                new_terms = list(terms)
                new_terms[first] = first_term + second_term
                del new_terms[second]
                set_equation(candidate_node, ",".join(new_terms) + "->" + output_term)
                kept = [
                    item
                    for item in candidate.graph.initializer
                    if item.name not in {first_name, second_name}
                ]
                kept.append(numpy_helper.from_array(fused, fused_name))
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept)
                results.append(
                    (
                        candidate,
                        {
                            "node_index": node_index,
                            "first": first_name,
                            "first_term": first_term,
                            "second": second_name,
                            "second_term": second_term,
                            "fused_term": first_term + second_term,
                        },
                    )
                )
    return results


def sign_absorptions(
    task: int, model: onnx.ModelProto
) -> list[tuple[onnx.ModelProto, dict[str, object]]]:
    """Find exact +/-1 gauge absorption opportunities in Einsum nodes.

    A one-dimensional sign initializer is absorbed into a tensor axis.  If the
    target tensor is shared by other terms, each affected use is compensated
    with a single-use initializer containing the corresponding term label.
    Multiplying both factors by the same sign is exact because sign**2 == 1.
    """

    initializers = {item.name: item for item in model.graph.initializer}
    uses = all_value_uses(model)
    results: list[tuple[onnx.ModelProto, dict[str, object]]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        eq = equation(node)
        if eq is None or "..." in eq:
            continue
        left, output_term = eq.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input):
            continue
        for source_index, source_name in enumerate(node.input):
            if source_name not in initializers or uses[source_name] != 1:
                continue
            source = numpy_helper.to_array(initializers[source_name])
            source_term = terms[source_index]
            if source.ndim != 1 or len(source_term) != 1 or source.size <= 1:
                continue
            if not np.all(np.isin(source, (-1, 1))):
                continue
            source_label = source_term[0]
            for target_index, target_name in enumerate(node.input):
                if target_index == source_index or target_name not in initializers:
                    continue
                target_term = terms[target_index]
                if target_term.count(source_label) != 1:
                    continue
                target = numpy_helper.to_array(initializers[target_name])
                target_axis = target_term.index(source_label)
                if target.ndim != len(target_term) or target.shape[target_axis] != source.size:
                    continue

                # Locate all uses of the target initializer in this node.  The
                # physical axis is the same, but its symbolic label may differ.
                target_uses = [
                    index for index, name in enumerate(node.input) if name == target_name
                ]
                compensations: list[tuple[int, str, int, str]] = []
                ok = True
                for other_index in target_uses:
                    if other_index == target_index:
                        continue
                    other_term = terms[other_index]
                    if len(other_term) != target.ndim:
                        ok = False
                        break
                    affected_label = other_term[target_axis]
                    selected = None
                    for comp_index, comp_name in enumerate(node.input):
                        if comp_index in {source_index, target_index, other_index}:
                            continue
                        if comp_name not in initializers or uses[comp_name] != 1:
                            continue
                        comp_term = terms[comp_index]
                        if comp_term.count(affected_label) != 1:
                            continue
                        comp = numpy_helper.to_array(initializers[comp_name])
                        comp_axis = comp_term.index(affected_label)
                        if comp.ndim == len(comp_term) and comp.shape[comp_axis] == source.size:
                            selected = (comp_index, comp_name, comp_axis, affected_label)
                            break
                    if selected is None:
                        ok = False
                        break
                    compensations.append(selected)
                if not ok:
                    continue

                candidate = copy.deepcopy(model)
                cnode = candidate.graph.node[node_index]
                cterms = list(terms)
                del cnode.input[source_index]
                del cterms[source_index]
                set_equation(cnode, ",".join(cterms) + "->" + output_term)

                arrays: dict[str, np.ndarray] = {target_name: target.copy()}
                reshape = [1] * target.ndim
                reshape[target_axis] = source.size
                arrays[target_name] = arrays[target_name] * source.reshape(reshape)
                for _, comp_name, comp_axis, _ in compensations:
                    comp = numpy_helper.to_array(initializers[comp_name]).copy()
                    comp_reshape = [1] * comp.ndim
                    comp_reshape[comp_axis] = source.size
                    arrays[comp_name] = comp * source.reshape(comp_reshape)

                kept = []
                for initializer in candidate.graph.initializer:
                    if initializer.name == source_name:
                        continue
                    if initializer.name in arrays:
                        kept.append(numpy_helper.from_array(arrays[initializer.name], initializer.name))
                    else:
                        kept.append(initializer)
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept)
                results.append(
                    (
                        candidate,
                        {
                            "node_index": node_index,
                            "source": source_name,
                            "source_term": source_term,
                            "target": target_name,
                            "target_term": target_term,
                            "target_axis": target_axis,
                            "compensations": [
                                {
                                    "input_index": index,
                                    "name": name,
                                    "axis": axis,
                                    "label": label,
                                }
                                for index, name, axis, label in compensations
                            ],
                        },
                    )
                )
    return results


def static_shape(value: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]] | None:
    if not value.type.HasField("tensor_type"):
        return None
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape") or tensor_type.elem_type == 0:
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return int(tensor_type.elem_type), tuple(dims)


def dtype_bytes(elem_type: int) -> int | None:
    mapping = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
        onnx.TensorProto.BFLOAT16: 2,
    }
    return mapping.get(elem_type)


def metadata_opportunity(task: int, model: onnx.ModelProto) -> dict[str, object]:
    clone = copy.deepcopy(model)
    original = {value.name: copy.deepcopy(value) for value in clone.graph.value_info}
    del clone.graph.value_info[:]
    try:
        inferred = onnx.shape_inference.infer_shapes(clone, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        return {"task": task, "status": "inference_failed", "error": str(exc)}
    rebuilt = {value.name: value for value in inferred.graph.value_info}
    overdeclared = []
    for name, old_value in original.items():
        if name not in rebuilt:
            continue
        old = static_shape(old_value)
        new = static_shape(rebuilt[name])
        if old is None or new is None or old[0] != new[0]:
            continue
        width = dtype_bytes(old[0])
        if width is None:
            continue
        old_bytes = math.prod(old[1]) * width
        new_bytes = math.prod(new[1]) * width
        if old_bytes > new_bytes:
            overdeclared.append(
                {
                    "name": name,
                    "old_shape": list(old[1]),
                    "inferred_shape": list(new[1]),
                    "old_bytes": old_bytes,
                    "inferred_bytes": new_bytes,
                    "potential_reduction": old_bytes - new_bytes,
                }
            )
    return {"task": task, "status": "scanned", "overdeclared": overdeclared}


def main() -> None:
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": sha256_path(BASELINE),
        "task_count": 0,
        "private_zero_excluded": sorted(PRIVATE_ZERO_EXCLUDE),
        "existing_candidate_copies": [],
        "initializer_dedup": [],
        "outer_fusion": [],
        "sign_absorption": [],
        "metadata_scan": [],
        "errors": [],
    }

    # Preserve byte identity of the three explicitly requested candidates.
    for task, source in EXISTING.items():
        data = source.read_bytes()
        destination = CANDIDATES / f"task{task:03d}_requested.onnx"
        destination.write_bytes(data)
        reports["existing_candidate_copies"].append(
            {
                "task": task,
                "source": str(source.relative_to(ROOT)),
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256_bytes(data),
                "structural_gate": strict_gate(onnx.load_model_from_string(data)),
            }
        )

    with zipfile.ZipFile(BASELINE) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        reports["task_count"] = len(members)
        for member in members:
            task = int(Path(member).stem[-3:])
            try:
                model = onnx.load_model(io.BytesIO(archive.read(member)))
                candidate, detail = initializer_dedup(task, model)
                if candidate is not None:
                    reports["initializer_dedup"].append(
                        save_candidate(task, "dedup", model, candidate, detail)
                    )

                for index, (candidate, detail) in enumerate(outer_fusions(task, model), 1):
                    reports["outer_fusion"].append(
                        save_candidate(task, f"outer_{index:02d}", model, candidate, detail)
                    )

                for index, (candidate, detail) in enumerate(sign_absorptions(task, model), 1):
                    reports["sign_absorption"].append(
                        save_candidate(task, f"sign_{index:02d}", model, candidate, detail)
                    )

                reports["metadata_scan"].append(metadata_opportunity(task, model))
            except Exception as exc:  # noqa: BLE001
                reports["errors"].append(
                    {"task": task, "type": type(exc).__name__, "error": str(exc)}
                )

    # Compact the no-op metadata rows while still proving all 400 were scanned.
    metadata_rows = reports["metadata_scan"]
    reports["metadata_scan_summary"] = {
        "scanned": len(metadata_rows),
        "inference_failures": sum(row.get("status") == "inference_failed" for row in metadata_rows),
        "tasks_with_overdeclared_value_info": [
            row["task"] for row in metadata_rows if row.get("overdeclared")
        ],
    }
    reports["metadata_scan"] = [
        row for row in metadata_rows if row.get("status") != "scanned" or row.get("overdeclared")
    ]

    output = HERE / "scan_report.json"
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports["metadata_scan_summary"], indent=2))
    print(
        json.dumps(
            {
                "initializer_dedup": len(reports["initializer_dedup"]),
                "outer_fusion": len(reports["outer_fusion"]),
                "sign_absorption": len(reports["sign_absorption"]),
                "errors": len(reports["errors"]),
                "report": str(output.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
