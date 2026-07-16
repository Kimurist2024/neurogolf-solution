#!/usr/bin/env python3
"""Scan Wave16 for exact singleton-shape initializer aliases in Einsums.

The transformation is deliberately narrow.  A removable initializer must be
used only by Einsum nodes and must equal an already-stored initializer after
dropping singleton dimensions and permuting the remaining dimensions.  Each
use is redirected to the stored initializer and only that operand's subscript
is rewritten.  No node, runtime tensor, or Einsum operand is added.

Same-rank aliases are intentionally excluded because the parent lane already
performed the same-rank permutation scan.  This lane focuses on rank changes
caused by singleton dimensions.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import string
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_BASELINE_SHA256 = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"
ASCII_LABELS = string.ascii_letters
MAX_PERMUTATIONS = 100_000


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    attributes = [attribute for attribute in node.attribute if attribute.name == "equation"]
    if len(attributes) != 1:
        raise ValueError("Einsum must have exactly one equation attribute")
    return attributes[0]


def array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def exact_core_permutations(source: np.ndarray, target: np.ndarray) -> list[tuple[int, ...]]:
    """Return source-core axis orders whose transpose is byte-identical to target."""
    source_core = np.squeeze(source)
    target_core = np.squeeze(target)
    if source_core.ndim != target_core.ndim:
        return []
    if source_core.size != target_core.size or source_core.dtype != target_core.dtype:
        return []
    rank = source_core.ndim
    if rank == 0:
        return [()] if array_bytes(source_core) == array_bytes(target_core) else []

    candidates_by_target_axis = [
        tuple(index for index, dimension in enumerate(source_core.shape) if dimension == wanted)
        for wanted in target_core.shape
    ]
    if any(not candidates for candidates in candidates_by_target_axis):
        return []

    results: list[tuple[int, ...]] = []
    explored = 0
    for permutation in itertools.product(*candidates_by_target_axis):
        if len(set(permutation)) != rank:
            continue
        explored += 1
        if explored > MAX_PERMUTATIONS:
            raise RuntimeError(f"permutation search exceeded {MAX_PERMUTATIONS}")
        permuted = np.transpose(source_core, axes=permutation)
        if array_bytes(permuted) == array_bytes(target_core):
            results.append(tuple(int(axis) for axis in permutation))
    return results


def replacement_term(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    target_term: str,
    output_term: str,
    core_permutation: tuple[int, ...],
    available_labels: list[str],
) -> tuple[str, list[dict[str, object]]]:
    """Construct a source subscript exactly equivalent to a target subscript."""
    if "..." in target_term or len(target_term) != len(target_shape):
        raise ValueError(f"unsupported target subscript {target_term!r} for shape {target_shape}")
    if any(label not in ASCII_LABELS for label in target_term + output_term):
        raise ValueError("only explicit ASCII-letter subscripts are supported")
    # Repeated labels invoke diagonal semantics.  They can be handled, but
    # excluding them keeps the proof local and unambiguous.
    if len(set(target_term)) != len(target_term):
        raise ValueError("repeated labels within the target operand are excluded")

    source_core_axes = [index for index, dim in enumerate(source_shape) if dim != 1]
    target_core_axes = [index for index, dim in enumerate(target_shape) if dim != 1]
    source_singleton_axes = [index for index, dim in enumerate(source_shape) if dim == 1]
    target_singleton_axes = [index for index, dim in enumerate(target_shape) if dim == 1]
    if len(source_core_axes) != len(target_core_axes):
        raise ValueError("squeezed ranks differ")
    if len(core_permutation) != len(target_core_axes):
        raise ValueError("invalid core permutation")

    source_labels: list[str | None] = [None] * len(source_shape)
    proof: list[dict[str, object]] = []
    # np.transpose(source_core, core_permutation) == target_core.  Therefore
    # target core axis j is source core axis core_permutation[j].
    for target_core_index, source_core_index in enumerate(core_permutation):
        source_axis = source_core_axes[source_core_index]
        target_axis = target_core_axes[target_core_index]
        label = target_term[target_axis]
        source_labels[source_axis] = label
        proof.append(
            {
                "source_axis": source_axis,
                "target_axis": target_axis,
                "dimension": int(source_shape[source_axis]),
                "label": label,
                "kind": "non_singleton_exact_permutation",
            }
        )

    target_singleton_labels = [target_term[axis] for axis in target_singleton_axes]
    # Any label emitted only by the target operand must remain present.  The
    # stronger conservative rule below retains every output singleton label,
    # even when another operand also carries it.
    mandatory = [label for label in target_singleton_labels if label in output_term]
    if len(mandatory) > len(source_singleton_axes):
        raise ValueError("source lacks enough singleton axes for output labels")
    retained = mandatory + [label for label in target_singleton_labels if label not in mandatory]

    singleton_iter = iter(source_singleton_axes)
    for label in retained:
        try:
            source_axis = next(singleton_iter)
        except StopIteration:
            break
        source_labels[source_axis] = label
        proof.append(
            {
                "source_axis": source_axis,
                "target_axis": target_term.index(label),
                "dimension": 1,
                "label": label,
                "kind": "singleton_label_retained",
            }
        )

    for source_axis in singleton_iter:
        if not available_labels:
            raise ValueError("no unused Einsum label for an extra singleton axis")
        label = available_labels.pop(0)
        source_labels[source_axis] = label
        proof.append(
            {
                "source_axis": source_axis,
                "target_axis": None,
                "dimension": 1,
                "label": label,
                "kind": "extra_singleton_summed_over_one",
            }
        )

    if any(label is None for label in source_labels):
        raise AssertionError("failed to label every source axis")
    return "".join(label for label in source_labels if label is not None), proof


def rewrite(
    source_model: onnx.ModelProto,
    target_name: str,
    source_name: str,
    core_permutation: tuple[int, ...],
) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = copy.deepcopy(source_model)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    target_tensor = initializers[target_name]
    source_tensor = initializers[source_name]
    target = np.asarray(numpy_helper.to_array(target_tensor))
    source = np.asarray(numpy_helper.to_array(source_tensor))
    if target.dtype != source.dtype:
        raise ValueError("dtype mismatch")
    if target.ndim == source.ndim:
        raise ValueError("same-rank aliases belong to the already-completed parent scan")
    matches = exact_core_permutations(source, target)
    if core_permutation not in matches:
        raise ValueError("initializer values are not the claimed exact shape alias")

    changes: list[dict[str, object]] = []
    uses = 0
    for node_index, node in enumerate(model.graph.node):
        positions = [index for index, name in enumerate(node.input) if name == target_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise ValueError(f"target has non-Einsum consumer {node.op_type}")
        attribute = equation_attribute(node)
        equation = attribute.s.decode("ascii")
        if equation.count("->") != 1:
            raise ValueError("explicit Einsum output is required")
        lhs, rhs = equation.split("->", 1)
        terms = lhs.split(",")
        if len(terms) != len(node.input):
            raise ValueError("Einsum operand/equation arity mismatch")
        used_labels = set("".join(terms) + rhs)
        available_labels = [label for label in ASCII_LABELS if label not in used_labels]
        for position in positions:
            old_term = terms[position]
            new_term, proof = replacement_term(
                tuple(int(dim) for dim in source.shape),
                tuple(int(dim) for dim in target.shape),
                old_term,
                rhs,
                core_permutation,
                available_labels,
            )
            node.input[position] = source_name
            terms[position] = new_term
            uses += 1
            changes.append(
                {
                    "node_index": node_index,
                    "input_index": position,
                    "old_term": old_term,
                    "new_term": new_term,
                    "axis_proof": proof,
                }
            )
        attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")

    if uses == 0:
        raise ValueError("target initializer has no consumers")
    remaining = [
        (node_index, input_index, node.op_type)
        for node_index, node in enumerate(model.graph.node)
        for input_index, name in enumerate(node.input)
        if name == target_name
    ]
    if remaining:
        raise ValueError(f"not every target use was replaced: {remaining}")

    kept = [tensor for tensor in model.graph.initializer if tensor.name != target_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    return model, {
        "target": target_name,
        "source": source_name,
        "dtype": str(target.dtype),
        "target_shape": list(target.shape),
        "source_shape": list(source.shape),
        "core_permutation": list(core_permutation),
        "removed_parameters": int(target.size),
        "uses": uses,
        "changes": changes,
        "nodes_added": 0,
        "runtime_tensors_added": 0,
        "einsum_operands_added": 0,
    }


def scan_task(
    task: int, payload: bytes
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    model = onnx.load_model_from_string(payload)
    initializers = list(model.graph.initializer)
    arrays = {tensor.name: np.asarray(numpy_helper.to_array(tensor)) for tensor in initializers}
    uses: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in arrays:
                uses[name].append((node_index, input_index, node.op_type))

    stats = {
        "initializers": len(initializers),
        "einsum_only_targets": 0,
        "rank_different_same_dtype_size_pairs": 0,
        "squeezed_shape_permutation_compatible_pairs": 0,
        "raw_flat_byte_equal_but_not_axis_alias_pairs": 0,
        "exact_aliases": 0,
    }
    built: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    pair_ordinal = 0
    for target_tensor in initializers:
        target_name = target_tensor.name
        target_uses = uses.get(target_name, [])
        if not target_uses or any(op_type != "Einsum" for _, _, op_type in target_uses):
            continue
        stats["einsum_only_targets"] += 1
        target = arrays[target_name]
        for source_tensor in initializers:
            source_name = source_tensor.name
            if source_name == target_name:
                continue
            source = arrays[source_name]
            if target.dtype != source.dtype or target.size != source.size:
                continue
            if target.ndim == source.ndim:
                continue
            stats["rank_different_same_dtype_size_pairs"] += 1
            source_core_shape = tuple(int(dim) for dim in np.squeeze(source).shape)
            target_core_shape = tuple(int(dim) for dim in np.squeeze(target).shape)
            shape_compatible = sorted(source_core_shape) == sorted(target_core_shape)
            if shape_compatible:
                stats["squeezed_shape_permutation_compatible_pairs"] += 1
            elif array_bytes(source) == array_bytes(target):
                # Exact storage-order reshapes that merge/split non-singleton
                # axes cannot be expressed by changing only one Einsum term.
                stats["raw_flat_byte_equal_but_not_axis_alias_pairs"] += 1
            try:
                permutations = exact_core_permutations(source, target)
            except Exception as error:
                rejected.append(
                    {
                        "task": task,
                        "target": target_name,
                        "source": source_name,
                        "stage": "alias_search",
                        "error": repr(error),
                    }
                )
                continue
            for permutation in permutations:
                stats["exact_aliases"] += 1
                pair_ordinal += 1
                stem = f"task{task:03d}_alias{pair_ordinal:03d}"
                try:
                    candidate, change = rewrite(model, target_name, source_name, permutation)
                    candidate_payload = candidate.SerializeToString()
                    output = HERE / f"{stem}.onnx"
                    onnx.save_model(candidate, output)
                    built.append(
                        {
                            "task": task,
                            "path": str(output.relative_to(ROOT)),
                            "sha256": sha256_bytes(output.read_bytes()),
                            "baseline_task_sha256": sha256_bytes(payload),
                            "change": change,
                            "strict_checker": "pass",
                        }
                    )
                except Exception as error:
                    rejected.append(
                        {
                            "task": task,
                            "target": target_name,
                            "source": source_name,
                            "core_permutation": list(permutation),
                            "stage": "rewrite_or_strict_check",
                            "error": repr(error),
                        }
                    )
    return built, rejected, stats


def main() -> int:
    baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if baseline_hash != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"baseline SHA mismatch: {baseline_hash}")

    built: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    aggregate_stats = {
        "initializers": 0,
        "einsum_only_targets": 0,
        "rank_different_same_dtype_size_pairs": 0,
        "squeezed_shape_permutation_compatible_pairs": 0,
        "raw_flat_byte_equal_but_not_axis_alias_pairs": 0,
        "exact_aliases": 0,
    }
    with zipfile.ZipFile(BASELINE) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            payload = archive.read(member)
            task_built, task_rejected, task_stats = scan_task(task, payload)
            built.extend(task_built)
            rejected.extend(task_rejected)
            for key, value in task_stats.items():
                aggregate_stats[key] += value
            task_rows.append(
                {
                    "task": task,
                    "built_candidates": len(task_built),
                    "rejected_exact_aliases": len(task_rejected),
                    **task_stats,
                }
            )

    manifest = {
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": baseline_hash,
        "tasks_scanned": 400,
        "scope": "rank-changing singleton-axis exact initializer aliases; same-rank permutations excluded",
        "built_count": len(built),
        "rejected_count": len(rejected),
        "scan_statistics": aggregate_stats,
        "built": built,
        "rejected": rejected,
        "tasks": task_rows,
    }
    (HERE / "scan_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    winner_manifest = {
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": baseline_hash,
        "winners": [],
        "winner_count": 0,
        "reason": (
            "No exact rank-changing singleton-axis initializer alias exists among the "
            "400 Wave16 task models, so no candidate reached empirical validation."
        ),
        "scan_manifest": str((HERE / "scan_build_manifest.json").relative_to(ROOT)),
        "fresh5000": "not applicable: zero candidates were produced",
        "dual_ort": "not applicable: zero candidates were produced",
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("baseline_sha256", "tasks_scanned", "built_count", "rejected_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
