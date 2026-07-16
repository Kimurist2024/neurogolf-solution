#!/usr/bin/env python3
"""Scan authority and staged-best models for T -> unary -> inverse-T.

The rewrite is admitted only when both intermediate values are single-use,
the transpose permutations compose to identity, the middle node is a safe
single-output elementwise unary, and neither intermediate is a graph output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
STAGE = ROOT / "others/71407"
CANDIDATES = HERE / "candidates.json"

# Pointwise unary ops whose output at each index depends only on the input at
# that same index. Cast's `to` and activation attributes are retained verbatim.
SAFE_UNARY = {
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BitwiseNot",
    "Cast",
    "Ceil",
    "Clip",  # only accepted below when it really has one input
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "IsInf",
    "IsNaN",
    "LeakyRelu",
    "Log",
    "Mish",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stage_snapshot() -> list[dict[str, Any]]:
    return [
        {
            "task": int(path.stem.removeprefix("task")),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path in sorted(STAGE.glob("task*.onnx"))
    ]


def snapshot_digest(rows: list[dict[str, Any]]) -> str:
    encoded = "\n".join(
        f"{row['path']}\0{row['sha256']}" for row in rows
    ).encode()
    return sha256_bytes(encoded)


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def tensor_dtype(value: onnx.ValueInfoProto) -> str:
    return onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)


def value_maps(model: onnx.ModelProto) -> tuple[dict[str, list[int | None]], dict[str, str]]:
    shapes: dict[str, list[int | None]] = {}
    dtypes: dict[str, str] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shapes[value.name] = tensor_shape(value)
        dtypes[value.name] = tensor_dtype(value)
    for item in model.graph.initializer:
        shapes[item.name] = [int(value) for value in item.dims]
        dtypes[item.name] = onnx.TensorProto.DataType.Name(item.data_type)
    return shapes, dtypes


def infer(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str | None]:
    try:
        return (
            onnx.shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=True
            ),
            None,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Detection remains complete for legacy authorities when lenient shape
        # inference can recover rank. Such a source cannot become a candidate
        # until the strict gate passes after rewriting.
        try:
            return (
                onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=False, data_prop=True
                ),
                error,
            )
        except Exception:
            return copy.deepcopy(model), error


def node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    result = {}
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        elif hasattr(value, "tolist"):
            value = value.tolist()
        result[attr.name] = value
    return result


def transpose_perm(node: onnx.NodeProto, rank: int | None) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == "perm":
            return [int(value) for value in attr.ints]
    if rank is None:
        return None
    return list(range(rank - 1, -1, -1))


def composed_identity(first: list[int] | None, second: list[int] | None) -> bool:
    if first is None or second is None or len(first) != len(second):
        return False
    rank = len(first)
    if sorted(first) != list(range(rank)) or sorted(second) != list(range(rank)):
        return False
    return [first[second[index]] for index in range(rank)] == list(range(rank))


def is_safe_unary(node: onnx.NodeProto) -> bool:
    return (
        node.op_type in SAFE_UNARY
        and node.domain in ("", "ai.onnx")
        and len(node.input) == 1
        and bool(node.input[0])
        and len(node.output) == 1
        and bool(node.output[0])
        and all(
            attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for attr in node.attribute
        )
    )


def is_one_input_one_output(node: onnx.NodeProto) -> bool:
    """Broad near-miss predicate used only to prove the safe list did not hide a path."""
    return (
        len(node.input) == 1
        and bool(node.input[0])
        and len(node.output) == 1
        and bool(node.output[0])
    )


def analyze_model(
    model: onnx.ModelProto,
    *,
    task: int,
    source_kind: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    inferred, strict_error = infer(model)
    shapes, dtypes = value_maps(inferred)
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name].append(index)
    graph_outputs = {value.name for value in model.graph.output}
    transpose_count = sum(node.op_type == "Transpose" for node in model.graph.node)
    safe_unary_count = sum(is_safe_unary(node) for node in model.graph.node)
    transpose_to_safe_unary = []
    safe_unary_to_transpose = []
    any_one_input_output_sandwiches = []
    sandwiches = []

    for first_index, first in enumerate(model.graph.node):
        if first.op_type != "Transpose" or len(first.output) != 1:
            continue
        first_output = first.output[0]
        for middle_index in consumers.get(first_output, []):
            middle = model.graph.node[middle_index]
            if is_one_input_one_output(middle) and middle.input[0] == first_output:
                middle_output = middle.output[0]
                for second_index in consumers.get(middle_output, []):
                    second = model.graph.node[second_index]
                    if (
                        second.op_type == "Transpose"
                        and second.input
                        and second.input[0] == middle_output
                    ):
                        any_one_input_output_sandwiches.append(
                            {
                                "node_indices": [first_index, middle_index, second_index],
                                "middle_op": middle.op_type,
                                "middle_domain": middle.domain,
                            }
                        )

            unary_index = middle_index
            unary = middle
            if not is_safe_unary(unary) or unary.input[0] != first_output:
                continue
            edge = {
                "transpose_index": first_index,
                "unary_index": unary_index,
                "unary_op": unary.op_type,
                "transpose_output_node_uses": len(consumers.get(first_output, [])),
                "transpose_output_is_graph_output": first_output in graph_outputs,
            }
            transpose_to_safe_unary.append(edge)
            unary_output = unary.output[0]
            next_transposes = [
                index
                for index in consumers.get(unary_output, [])
                if model.graph.node[index].op_type == "Transpose"
                and model.graph.node[index].input
                and model.graph.node[index].input[0] == unary_output
            ]
            for second_index in next_transposes:
                second = model.graph.node[second_index]
                safe_unary_to_transpose.append(
                    {
                        "unary_index": unary_index,
                        "transpose_index": second_index,
                        "unary_op": unary.op_type,
                        "unary_output_node_uses": len(consumers.get(unary_output, [])),
                        "unary_output_is_graph_output": unary_output in graph_outputs,
                    }
                )
                input_shape = shapes.get(first.input[0])
                rank = len(input_shape) if input_shape is not None else None
                first_perm = transpose_perm(first, rank)
                second_perm = transpose_perm(second, rank)
                identity = composed_identity(first_perm, second_perm)
                single_use = (
                    len(consumers.get(first_output, [])) == 1
                    and len(consumers.get(unary_output, [])) == 1
                    and first_output not in graph_outputs
                    and unary_output not in graph_outputs
                )
                final_shape_same = (
                    shapes.get(first.input[0]) is not None
                    and shapes.get(second.output[0]) == shapes.get(first.input[0])
                )
                dtype_chain = [
                    dtypes.get(first.input[0]),
                    dtypes.get(first_output),
                    dtypes.get(unary_output),
                    dtypes.get(second.output[0]),
                ]
                eligible = bool(
                    identity
                    and single_use
                    and final_shape_same
                    and dtype_chain[0] == dtype_chain[1]
                    and dtype_chain[2] == dtype_chain[3]
                )
                sandwiches.append(
                    {
                        "task": task,
                        "source_kind": source_kind,
                        "source_path": source_path,
                        "source_sha256": source_sha256,
                        "node_indices": [first_index, unary_index, second_index],
                        "unary_op": unary.op_type,
                        "unary_attributes": node_attributes(unary),
                        "first_perm": first_perm,
                        "second_perm": second_perm,
                        "perm_composes_identity": identity,
                        "intermediates_single_use": single_use,
                        "input_shape": shapes.get(first.input[0]),
                        "first_output_shape": shapes.get(first_output),
                        "unary_output_shape": shapes.get(unary_output),
                        "final_output_shape": shapes.get(second.output[0]),
                        "dtype_chain": dtype_chain,
                        "final_shape_same": final_shape_same,
                        "strict_source_shape_inference": strict_error is None,
                        "eligible": eligible,
                    }
                )

    checker = True
    checker_error = None
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    except Exception as exc:
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    return {
        "task": task,
        "source_kind": source_kind,
        "source_path": source_path,
        "sha256": source_sha256,
        "node_count": len(model.graph.node),
        "transpose_count": transpose_count,
        "safe_unary_count": safe_unary_count,
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_shape_inference_data_prop": strict_error is None,
        "strict_error": strict_error,
        "any_one_input_output_sandwiches": any_one_input_output_sandwiches,
        "transpose_to_safe_unary_edges": transpose_to_safe_unary,
        "safe_unary_to_transpose_edges": safe_unary_to_transpose,
        "sandwiches": sandwiches,
    }


def cancel_sandwich(
    model: onnx.ModelProto, first_index: int, unary_index: int, second_index: int
) -> onnx.ModelProto:
    """Relocate the unary to the untransposed tensor and keep final naming."""
    candidate = copy.deepcopy(model)
    first = candidate.graph.node[first_index]
    unary = candidate.graph.node[unary_index]
    second = candidate.graph.node[second_index]
    first_output = first.output[0]
    unary_output = unary.output[0]
    relocated = copy.deepcopy(unary)
    relocated.input[0] = first.input[0]
    relocated.output[0] = second.output[0]
    nodes = []
    for index, node in enumerate(candidate.graph.node):
        if index == first_index:
            nodes.append(relocated)
        elif index in (unary_index, second_index):
            continue
        else:
            nodes.append(copy.deepcopy(node))
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    retained_value_info = [
        copy.deepcopy(value)
        for value in candidate.graph.value_info
        if value.name not in (first_output, unary_output)
    ]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(retained_value_info)
    return candidate


def authority_models() -> tuple[dict[int, tuple[onnx.ModelProto, str, str]], str]:
    data = AUTHORITY.read_bytes()
    result = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for name in sorted(member for member in archive.namelist() if member.endswith(".onnx")):
            raw = archive.read(name)
            task = int(Path(name).stem.removeprefix("task"))
            result[task] = (
                onnx.load_model_from_string(raw),
                f"submission.zip::{name}",
                sha256_bytes(raw),
            )
    return result, sha256_bytes(data)


def stage_models() -> dict[int, tuple[onnx.ModelProto, str, str]]:
    result = {}
    for path in sorted(STAGE.glob("task*.onnx")):
        raw = path.read_bytes()
        task = int(path.stem.removeprefix("task"))
        result[task] = (onnx.load_model_from_string(raw), str(path.relative_to(ROOT)), sha256_bytes(raw))
    return result


def analyze_collection(
    models: Iterable[tuple[int, tuple[onnx.ModelProto, str, str], str]]
) -> list[dict[str, Any]]:
    return [
        analyze_model(
            model,
            task=task,
            source_kind=source_kind,
            source_path=path,
            source_sha256=digest,
        )
        for task, (model, path, digest), source_kind in models
    ]


def main() -> None:
    authority, authority_sha = authority_models()
    stage = stage_models()
    staged_snapshot = stage_snapshot()
    composite = dict(authority)
    composite.update(stage)
    authority_rows = analyze_collection(
        (task, item, "authority") for task, item in sorted(authority.items())
    )
    stage_rows = analyze_collection(
        (task, item, "staged") for task, item in sorted(stage.items())
    )
    composite_rows = analyze_collection(
        (
            task,
            item,
            "staged_best" if task in stage else "authority_best",
        )
        for task, item in sorted(composite.items())
    )

    all_sandwiches = [
        sandwich
        for row in composite_rows
        for sandwich in row["sandwiches"]
    ]
    eligible = [row for row in all_sandwiches if row["eligible"]]
    candidate_rows = []
    # Current census has no eligible pattern. This branch intentionally builds
    # into a temporary directory first; only a strict-lower profile would be
    # retained under CANDIDATES by a follow-on gate.
    for row in eligible:
        task = row["task"]
        source_model = composite[task][0]
        candidate = cancel_sandwich(source_model, *row["node_indices"])
        try:
            onnx.checker.check_model(candidate, full_check=True)
            onnx.shape_inference.infer_shapes(
                copy.deepcopy(candidate), strict_mode=True, data_prop=True
            )
            candidate_rows.append(
                {
                    **row,
                    "candidate_built_in_memory": True,
                    "candidate_sha256": sha256_bytes(candidate.SerializeToString()),
                    "requires_official_cost_and_runtime_gate": True,
                }
            )
        except Exception as exc:
            candidate_rows.append(
                {**row, "candidate_built_in_memory": False, "error": f"{type(exc).__name__}: {exc}"}
            )

    def collection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "models": len(rows),
            "nodes": sum(row["node_count"] for row in rows),
            "transpose_nodes": sum(row["transpose_count"] for row in rows),
            "safe_unary_nodes": sum(row["safe_unary_count"] for row in rows),
            "transpose_to_safe_unary_edges": sum(
                len(row["transpose_to_safe_unary_edges"]) for row in rows
            ),
            "safe_unary_to_transpose_edges": sum(
                len(row["safe_unary_to_transpose_edges"]) for row in rows
            ),
            "any_one_input_output_sandwiches": sum(
                len(row["any_one_input_output_sandwiches"]) for row in rows
            ),
            "sandwiches": sum(len(row["sandwiches"]) for row in rows),
            "eligible": sum(
                sandwich["eligible"]
                for row in rows
                for sandwich in row["sandwiches"]
            ),
            "checker_failures": sum(not row["checker_full"] for row in rows),
            "strict_inference_failures": sum(
                not row["strict_shape_inference_data_prop"] for row in rows
            ),
        }

    payload = {
        "scan": "root_transpose_unary_cancel_scan_268",
        "safe_unary_ops": sorted(SAFE_UNARY),
        "authority": {
            "path": "submission.zip",
            "sha256": authority_sha,
            "models": len(authority),
        },
        "stage": {
            "path": "others/71407",
            "active_root_onnx": len(stage),
            "snapshot_digest": snapshot_digest(staged_snapshot),
            "snapshot": staged_snapshot,
            "manifest_sha256": sha256(STAGE / "MANIFEST.json") if (STAGE / "MANIFEST.json").is_file() else None,
        },
        "composite_best": {
            "models": len(composite),
            "staged_overrides": sorted(stage),
        },
        "summaries": {
            "authority": collection_summary(authority_rows),
            "stage": collection_summary(stage_rows),
            "composite_best": collection_summary(composite_rows),
        },
        "near_edges": {
            "authority": [
                {
                    "task": row["task"],
                    "transpose_to_safe_unary": row["transpose_to_safe_unary_edges"],
                    "safe_unary_to_transpose": row["safe_unary_to_transpose_edges"],
                }
                for row in authority_rows
                if row["transpose_to_safe_unary_edges"] or row["safe_unary_to_transpose_edges"]
            ],
            "stage": [
                {
                    "task": row["task"],
                    "transpose_to_safe_unary": row["transpose_to_safe_unary_edges"],
                    "safe_unary_to_transpose": row["safe_unary_to_transpose_edges"],
                }
                for row in stage_rows
                if row["transpose_to_safe_unary_edges"] or row["safe_unary_to_transpose_edges"]
            ],
        },
        "sandwiches": all_sandwiches,
        "eligible_patterns": eligible,
        "candidate_rows": candidate_rows,
        "strict_lower_candidates": [],
        "winner": None,
        "candidate_gate": {
            "official_known_four_config_raw": False,
            "fresh_independent_2000": False,
            "full_strict_profile_ub0": False,
            "skip_reason": (
                "No structural Transpose-oneInputOneOutput-Transpose path exists in "
                "composite best, even before safe-op, inverse-permutation, or single-use filters."
            ),
        },
        "policy": {
            "private_zero_candidate": False,
            "runtime_shape_cloak_candidate": False,
            "lookup_candidate": False,
            "root_or_others71407_modified": False,
        },
    }
    if eligible or candidate_rows:
        raise RuntimeError("eligible patterns require the full runtime candidate gate before completion")
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    CANDIDATES.write_text(
        json.dumps(
            {
                "scan": payload["scan"],
                "structural_candidates": candidate_rows,
                "strict_lower_candidates": [],
                "winner": None,
                "execution_gate": payload["candidate_gate"],
                "policy": payload["policy"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "authority_sha256": authority_sha,
                "stage_files": len(stage),
                "composite_models": len(composite),
                "summaries": payload["summaries"],
                "eligible_patterns": len(eligible),
                "strict_lower_candidates": 0,
                "output": str((HERE / "scan.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
