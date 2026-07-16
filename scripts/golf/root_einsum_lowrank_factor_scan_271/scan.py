#!/usr/bin/env python3
"""Exact-rational all-bipartition scan of constant Einsum operands.

For each numeric initializer used by an Einsum, every unordered nonempty axis
bipartition is flattened to a matrix over Q and ranked exactly.  A rank-R
factorization is considered removable only if every graph use of the original
initializer can be replaced in its Einsum and every repeated occurrence in a
single equation has its own unused ASCII latent index.
"""

from __future__ import annotations

import hashlib
import json
import string
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import sympy as sp
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
STAGE = ROOT / "others/71407"
LETTERS = tuple(string.ascii_letters)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            value = onnx.helper.get_attribute_value(attr)
            return value.decode("ascii") if isinstance(value, bytes) else str(value)
    return None


def scalar_rational(value: Any) -> sp.Rational:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (int, np.integer, bool, np.bool_)):
        return sp.Rational(int(scalar))
    numerator, denominator = float(scalar).as_integer_ratio()
    return sp.Rational(numerator, denominator)


def exact_matrix(array: np.ndarray, left: tuple[int, ...], right: tuple[int, ...]) -> sp.Matrix:
    rows = int(np.prod([array.shape[axis] for axis in left], dtype=np.int64))
    columns = int(np.prod([array.shape[axis] for axis in right], dtype=np.int64))
    flattened = np.transpose(array, left + right).reshape(rows, columns)
    return sp.Matrix(
        rows,
        columns,
        [scalar_rational(value) for value in flattened.reshape(-1)],
    )


def rational_to_dtype(value: sp.Rational, dtype: np.dtype[Any]) -> tuple[bool, Any | None]:
    if np.issubdtype(dtype, np.integer):
        if value.q != 1:
            return False, None
        integer = int(value.p)
        info = np.iinfo(dtype)
        if integer < info.min or integer > info.max:
            return False, None
        scalar = np.asarray(integer, dtype=dtype).item()
    elif np.issubdtype(dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            scalar = np.asarray(float(value), dtype=dtype).item()
        if not np.isfinite(scalar):
            return False, None
    else:
        return False, None
    return scalar_rational(scalar) == value, scalar


def serialized_exact_factorization(
    matrix: sp.Matrix, dtype: np.dtype[Any]
) -> dict[str, Any]:
    left_q, right_q = matrix.rank_decomposition()
    left_values: list[Any] = []
    right_values: list[Any] = []
    representable = True
    for value in left_q:
        exact, scalar = rational_to_dtype(value, dtype)
        representable &= exact
        left_values.append(scalar)
    for value in right_q:
        exact, scalar = rational_to_dtype(value, dtype)
        representable &= exact
        right_values.append(scalar)

    reconstructed = left_q * right_q == matrix
    serialized_reconstructed = False
    left_array = None
    right_array = None
    if representable:
        left_array = np.asarray(left_values, dtype=dtype).reshape(left_q.rows, left_q.cols)
        right_array = np.asarray(right_values, dtype=dtype).reshape(right_q.rows, right_q.cols)
        left_serialized_q = sp.Matrix(
            left_q.rows,
            left_q.cols,
            [scalar_rational(value) for value in left_array.reshape(-1)],
        )
        right_serialized_q = sp.Matrix(
            right_q.rows,
            right_q.cols,
            [scalar_rational(value) for value in right_array.reshape(-1)],
        )
        serialized_reconstructed = left_serialized_q * right_serialized_q == matrix

    def rational_rows(value: sp.Matrix) -> list[list[str]]:
        return [
            [str(value[row, column]) for column in range(value.cols)]
            for row in range(value.rows)
        ]

    return {
        "rational_reconstruction_exact": bool(reconstructed),
        "factor_dtype_fully_representable": bool(representable),
        "serialized_coefficient_reconstruction_exact": bool(serialized_reconstructed),
        "left_rationals": rational_rows(left_q),
        "right_rationals": rational_rows(right_q),
        "left_serialized_sha256": (
            sha256_bytes(left_array.tobytes(order="C")) if left_array is not None else None
        ),
        "right_serialized_sha256": (
            sha256_bytes(right_array.tobytes(order="C")) if right_array is not None else None
        ),
    }


def bipartitions(rank: int) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate unordered nonempty bipartitions once by fixing axis zero left."""
    for mask in range(1, 1 << rank):
        if not mask & 1 or mask == (1 << rank) - 1:
            continue
        left = tuple(axis for axis in range(rank) if mask & (1 << axis))
        right = tuple(axis for axis in range(rank) if not mask & (1 << axis))
        yield left, right


def use_inventory(
    model: onnx.ModelProto, initializer_names: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    all_uses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    einsum_uses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        eq = equation(node) if node.op_type == "Einsum" else None
        terms = eq.split("->", 1)[0].split(",") if eq and "->" in eq else []
        used_labels = sorted(set("".join(terms) + (eq.split("->", 1)[1] if eq and "->" in eq else "")))
        free_labels = [label for label in LETTERS if label not in used_labels]
        for input_index, name in enumerate(node.input):
            if name not in initializer_names:
                continue
            common = {
                "node_index": node_index,
                "input_index": input_index,
                "op_type": node.op_type,
            }
            all_uses[name].append(common)
            if node.op_type == "Einsum":
                einsum_uses[name].append(
                    {
                        **common,
                        "equation": eq,
                        "term": terms[input_index] if input_index < len(terms) else None,
                        "used_label_count": len(used_labels),
                        "free_labels": free_labels,
                    }
                )
    return all_uses, einsum_uses


def label_budget(
    uses: list[dict[str, Any]], initializer_rank: int
) -> tuple[list[dict[str, Any]], bool]:
    per_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for use in uses:
        per_node[int(use["node_index"])].append(use)
    rows = []
    all_sufficient = True
    for node_index, node_uses in sorted(per_node.items()):
        representative = node_uses[0]
        supported_terms = all(
            isinstance(use["term"], str)
            and "..." not in use["term"]
            and len(use["term"]) == initializer_rank
            for use in node_uses
        )
        required = len(node_uses)
        available = len(representative["free_labels"])
        sufficient = supported_terms and available >= required
        all_sufficient &= sufficient
        rows.append(
            {
                "node_index": node_index,
                "equation": representative["equation"],
                "occurrences_requiring_independent_latents": required,
                "available_unused_labels": available,
                "unused_labels": representative["free_labels"],
                "terms_supported": supported_terms,
                "sufficient": sufficient,
            }
        )
    return rows, all_sufficient


def scan_model(
    model: onnx.ModelProto,
    *,
    task: int,
    source_kind: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    arrays = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer))
        for initializer in model.graph.initializer
    }
    all_uses, einsum_uses = use_inventory(model, set(arrays))
    partition_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name, uses in sorted(einsum_uses.items()):
        array = arrays[name]
        if array.ndim < 2:
            skipped.append({"initializer": name, "reason": "rank_below_two", "rank": array.ndim})
            continue
        if array.ndim > 8:
            skipped.append({"initializer": name, "reason": "rank_above_eight", "rank": array.ndim})
            continue
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            skipped.append(
                {"initializer": name, "reason": "nonfinite_or_nonnumeric", "dtype": str(array.dtype)}
            )
            continue

        every_graph_use_is_einsum = len(all_uses[name]) == len(uses)
        budgets, labels_sufficient = label_budget(uses, array.ndim)
        terms_supported = all(row["terms_supported"] for row in budgets)
        for left, right in bipartitions(array.ndim):
            left_size = int(np.prod([array.shape[axis] for axis in left], dtype=np.int64))
            right_size = int(np.prod([array.shape[axis] for axis in right], dtype=np.int64))
            matrix = exact_matrix(array, left, right)
            exact_rank = int(matrix.rank())
            original_params = int(array.size)
            unique_new_params = exact_rank * (left_size + right_size)
            nominal_saving = original_params - unique_new_params
            rank1_overlap = exact_rank == 1 and nominal_saving > 0
            rge2_parameter_saving = exact_rank >= 2 and nominal_saving > 0
            factor_details: dict[str, Any] | None = None
            if rge2_parameter_saving:
                factor_details = serialized_exact_factorization(matrix, array.dtype)
            dtype_exact = bool(
                factor_details
                and factor_details["factor_dtype_fully_representable"]
                and factor_details["rational_reconstruction_exact"]
                and factor_details["serialized_coefficient_reconstruction_exact"]
            )
            removable_params = original_params if every_graph_use_is_einsum and terms_supported else 0
            strict_parameter_lower = removable_params > unique_new_params
            structural_candidate = bool(
                rge2_parameter_saving
                and strict_parameter_lower
                and dtype_exact
                and labels_sufficient
            )
            row = {
                "task": task,
                "source_kind": source_kind,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "initializer": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "einsum_use_occurrences": len(uses),
                "all_graph_use_occurrences": len(all_uses[name]),
                "every_graph_use_is_supported_einsum": bool(
                    every_graph_use_is_einsum and terms_supported
                ),
                "left_axes": list(left),
                "right_axes": list(right),
                "matrix_shape": [left_size, right_size],
                "exact_rational_rank": exact_rank,
                "original_params": original_params,
                "unique_new_params": unique_new_params,
                "nominal_parameter_saving": nominal_saving,
                "removable_params": removable_params,
                "rank1_overlap_lane270": rank1_overlap,
                "rge2_parameter_saving": rge2_parameter_saving,
                "label_budget": budgets if rge2_parameter_saving else None,
                "label_budget_sufficient": labels_sufficient if rge2_parameter_saving else None,
                "factorization": factor_details,
                "strict_parameter_lower": strict_parameter_lower if rge2_parameter_saving else False,
                "structural_candidate": structural_candidate,
            }
            partition_rows.append(row)

    return {
        "task": task,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "einsum_nodes": sum(node.op_type == "Einsum" for node in model.graph.node),
        "einsum_constant_initializers": len(einsum_uses),
        "partition_rows": partition_rows,
        "skipped": skipped,
    }


def authority_models() -> tuple[dict[int, tuple[onnx.ModelProto, str, str]], str]:
    payload = AUTHORITY.read_bytes()
    models = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            name = f"task{task:03d}.onnx"
            raw = archive.read(name)
            models[task] = (
                onnx.load_model_from_string(raw),
                f"submission.zip::{name}",
                sha256_bytes(raw),
            )
    return models, sha256_bytes(payload)


def active_stage_models() -> tuple[dict[int, tuple[onnx.ModelProto, str, str]], dict[str, Any]]:
    manifest = json.loads((STAGE / "MANIFEST.json").read_text())
    models = {}
    snapshot = []
    for row in manifest["active_candidates"]:
        path = STAGE / row["file"]
        raw = path.read_bytes()
        task = int(row["task"])
        digest = sha256_bytes(raw)
        if digest != row["sha256"]:
            raise RuntimeError(f"stage manifest SHA mismatch: {path}")
        models[task] = (onnx.load_model_from_string(raw), str(path.relative_to(ROOT)), digest)
        snapshot.append({"task": task, "path": str(path.relative_to(ROOT)), "sha256": digest})
    if len(models) != manifest["active_root_onnx_count"]:
        raise RuntimeError("active stage count does not match manifest")
    return models, {
        "manifest_sha256": sha256(STAGE / "MANIFEST.json"),
        "snapshot": snapshot,
        "snapshot_digest": sha256_bytes(
            "\n".join(f"{row['path']}\0{row['sha256']}" for row in snapshot).encode()
        ),
    }


def scan_collection(
    models: dict[int, tuple[onnx.ModelProto, str, str]], source_kind: str
) -> list[dict[str, Any]]:
    return [
        scan_model(
            model,
            task=task,
            source_kind=source_kind,
            source_path=path,
            source_sha256=digest,
        )
        for task, (model, path, digest) in sorted(models.items())
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = [partition for row in rows for partition in row["partition_rows"]]
    return {
        "models": len(rows),
        "einsum_nodes": sum(row["einsum_nodes"] for row in rows),
        "einsum_constant_initializers": sum(row["einsum_constant_initializers"] for row in rows),
        "exact_axis_bipartitions": len(partitions),
        "rank1_parameter_saving_lane270_overlap": sum(
            partition["rank1_overlap_lane270"] for partition in partitions
        ),
        "rge2_parameter_saving": sum(partition["rge2_parameter_saving"] for partition in partitions),
        "rge2_dtype_exact": sum(
            partition["rge2_parameter_saving"]
            and bool(partition["factorization"])
            and partition["factorization"]["serialized_coefficient_reconstruction_exact"]
            for partition in partitions
        ),
        "rge2_label_budget_sufficient": sum(
            partition["rge2_parameter_saving"]
            and bool(partition["label_budget_sufficient"])
            for partition in partitions
        ),
        "structural_candidates": sum(partition["structural_candidate"] for partition in partitions),
    }


def main() -> None:
    authority, authority_sha = authority_models()
    stage, stage_snapshot = active_stage_models()
    authority_rows = scan_collection(authority, "authority")
    stage_rows = scan_collection(stage, "active_descendant")
    authority_by_task = {row["task"]: row for row in authority_rows}
    stage_by_task = {row["task"]: row for row in stage_rows}
    composite_rows = [
        stage_by_task.get(task, authority_by_task[task])
        for task in range(1, 401)
    ]
    composite_partitions = [
        partition for row in composite_rows for partition in row["partition_rows"]
    ]
    rge2 = [row for row in composite_partitions if row["rge2_parameter_saving"]]
    structural = [row for row in rge2 if row["structural_candidate"]]
    if structural:
        raise RuntimeError(
            "structural candidate exists; build and full runtime gate are required before completion"
        )

    payload = {
        "scan": "root_einsum_lowrank_factor_scan_271",
        "method": {
            "coefficient_domain": "exact rationals derived from serialized numeric values",
            "bipartitions": "all unordered nonempty axis bipartitions; axis0 fixes orientation",
            "rank": "SymPy exact rational matrix rank",
            "factorization": "exact rank decomposition M=C*F with serialized-dtype reconstruction check",
            "replacement": "one independent unused ASCII latent label per initializer occurrence in each Einsum",
            "latent_label_alphabet": (
                "a-zA-Z; audit micrographs confirm digits, punctuation, and Unicode are rejected by "
                "ONNX full shape inference and ORT 1.24"
            ),
            "rank1_policy": "record only; lane270 overlap excluded from this lane",
        },
        "authority": {"path": "submission.zip", "sha256": authority_sha, "models": 400},
        "active_stage": {
            "path": "others/71407",
            "models": len(stage),
            "tasks": sorted(stage),
            **stage_snapshot,
        },
        "composite_best": {"models": 400, "active_descendant_overrides": sorted(stage)},
        "summaries": {
            "authority": summarize(authority_rows),
            "active_stage": summarize(stage_rows),
            "composite_best": summarize(composite_rows),
        },
        "composite_rge2_parameter_saving": rge2,
        "structural_candidates": structural,
        "strict_lower_candidates": [],
        "winner": None,
        "candidate_gate": {
            "full_checker_strict_data_prop": False,
            "actual_profile_truthful_shape": False,
            "known_four_raw": False,
            "fresh_two_by_2000": False,
            "runtime_errors_zero": False,
            "nonfinite_zero": False,
            "conv_bias_ub0": False,
            "skip_reason": (
                "All exact rank-R>=2 parameter-saving decompositions fail the independent-latent "
                "label budget before a candidate graph can be formed."
            ),
        },
        "policy": {
            "approximate_factorization_admitted": False,
            "rank1_candidate_admitted": False,
            "shared_initializer_partially_replaced": False,
            "runtime_shape_cloak_admitted": False,
            "private_zero_candidate_admitted": False,
            "root_or_others71407_modified": False,
        },
        "partition_rows": {
            "authority": [partition for row in authority_rows for partition in row["partition_rows"]],
            "active_stage": [partition for row in stage_rows for partition in row["partition_rows"]],
        },
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    candidates = {
        "scan": payload["scan"],
        "rge2_parameter_saving": rge2,
        "structural_candidates": [],
        "strict_lower_candidates": [],
        "winner": None,
        "candidate_gate": payload["candidate_gate"],
        "policy": payload["policy"],
    }
    (HERE / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(
        json.dumps(
            {
                "authority_sha256": authority_sha,
                "active_stage_models": len(stage),
                "summaries": payload["summaries"],
                "rge2_parameter_saving": len(rge2),
                "structural_candidates": len(structural),
                "output": str((HERE / "scan.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
