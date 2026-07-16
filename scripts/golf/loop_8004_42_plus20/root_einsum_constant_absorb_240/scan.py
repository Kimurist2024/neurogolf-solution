#!/usr/bin/env python3
"""Global exact constant absorption scan for multi-operand Einsum graphs.

The scan is deliberately authority-preserving: it only combines serialized
constant operands by exact integer or copy/sign/power-of-two arithmetic, and
it prices removals over every graph use of each initializer.  No root artifact
is modified.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"
MAX_SUBSET = 4
EXCLUDED_TASKS = {310}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            value = helper.get_attribute_value(attr)
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def labels(term: str) -> list[str]:
    return [char for char in term if char.isalpha()]


def load_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open() as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def static_profile(model: onnx.ModelProto) -> dict[str, int]:
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    free = {value.name for value in [*inferred.graph.input, *inferred.graph.output]}
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in free or name in seen:
                continue
            seen.add(name)
            tensor = typed[name].type.tensor_type
            dims = [int(dim.dim_value) for dim in tensor.shape.dim]
            if any(dim <= 0 for dim in dims):
                raise ValueError(f"nonstatic output {name}")
            dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
            memory += math.prod(dims) * dtype.itemsize
    params = sum(math.prod(item.dims) for item in model.graph.initializer)
    for sparse in model.graph.sparse_initializer:
        params += math.prod(sparse.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def occurrence_map(model: onnx.ModelProto) -> tuple[dict[str, list[tuple[int, int]]], dict[tuple[int, int], str]]:
    by_name: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    by_position: dict[tuple[int, int], str] = {}
    for node_index, node in enumerate(model.graph.node):
        for position, name in enumerate(node.input):
            if name:
                key = (node_index, position)
                by_name[name].append(key)
                by_position[key] = name
    return dict(by_name), by_position


def is_power_of_two_value(value: float) -> bool:
    if value == 0 or not math.isfinite(value):
        return value == 0
    mantissa, _ = math.frexp(abs(value))
    return mantissa == 0.5


def exact_combine(
    arrays: list[np.ndarray], terms: list[str], result_labels: str, contracted: list[str]
) -> tuple[np.ndarray, str] | None:
    if not arrays or len({array.dtype for array in arrays}) != 1:
        return None
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return None
    expression = ",".join(terms) + "->" + result_labels
    dtype = arrays[0].dtype

    integer_valued = all(
        array.dtype.kind in "biu" or np.equal(array, np.trunc(array)).all()
        for array in arrays
    )
    if integer_valued:
        maxima = [int(np.max(np.abs(array.astype(np.float64)), initial=0)) for array in arrays]
        product_bound = math.prod(maximum for maximum in maxima)
        dims: dict[str, int] = {}
        for term, array in zip(terms, arrays):
            for label, dim in zip(labels(term), array.shape):
                dims.setdefault(label, int(dim))
        sum_bound = math.prod(dims[label] for label in contracted)
        if product_bound * sum_bound <= 2**52:
            high = np.einsum(
                expression, *[array.astype(np.float64) for array in arrays], optimize=False
            )
            cast = np.asarray(high, dtype=dtype)
            if np.array_equal(cast.astype(np.float64), high):
                return cast, "exact_integer"

    # Copy/sign/permutation and dyadic scale absorption.  With at most one
    # non-dyadic operand and at most one contributing product per result cell,
    # the precomputed tensor is an exact copy/sign/power-of-two rescaling.
    dyadic = [
        bool(all(is_power_of_two_value(float(value)) for value in array.reshape(-1)))
        for array in arrays
    ]
    if sum(not flag for flag in dyadic) <= 1:
        support = np.einsum(
            expression,
            *[(array != 0).astype(np.int64) for array in arrays],
            optimize=False,
        )
        if int(np.max(support, initial=0)) <= 1:
            high = np.einsum(
                expression, *[array.astype(np.float64) for array in arrays], optimize=False
            )
            cast = np.asarray(high, dtype=dtype)
            if np.all(np.isfinite(cast)) and np.array_equal(cast.astype(np.float64), high):
                return cast, "exact_copy_sign_dyadic"
    return None


def subset_spec(
    terms: list[str], positions: tuple[int, ...], arrays: list[np.ndarray], output_labels: str
) -> tuple[str, list[str], int] | None:
    selected_counts = Counter(
        label for position in positions for label in labels(terms[position])
    )
    outside_counts = Counter(
        label
        for position, term in enumerate(terms)
        if position not in positions
        for label in labels(term)
    )
    output_set = set(labels(output_labels))
    retained = {
        label
        for label in selected_counts
        if outside_counts[label] > 0 or label in output_set
    }
    ordered = "".join(
        dict.fromkeys(
            label
            for position in positions
            for label in labels(terms[position])
            if label in retained
        )
    )
    contracted = sorted(set(selected_counts) - retained)
    dims: dict[str, int] = {}
    for position, array in zip(positions, arrays):
        term_labels = labels(terms[position])
        if len(term_labels) != array.ndim:
            return None
        for label, dim in zip(term_labels, array.shape):
            if label in dims and dims[label] != int(dim):
                return None
            dims[label] = int(dim)
    after = math.prod(dims[label] for label in ordered)
    return ordered, contracted, int(after)


def connected_subsets(positions: list[int], terms: list[str]) -> list[tuple[int, ...]]:
    label_sets = {position: set(labels(terms[position])) for position in positions}
    result: set[tuple[int, ...]] = set()
    frontier: set[frozenset[int]] = set()
    for left, right in itertools.combinations(positions, 2):
        if label_sets[left] & label_sets[right] or not label_sets[left] or not label_sets[right]:
            item = frozenset((left, right))
            frontier.add(item)
            result.add(tuple(sorted(item)))
    for _size in range(3, MAX_SUBSET + 1):
        new_frontier: set[frozenset[int]] = set()
        for current in frontier:
            current_labels = set().union(*(label_sets[position] for position in current))
            for position in positions:
                if position in current:
                    continue
                if current_labels & label_sets[position] or not label_sets[position]:
                    item = frozenset((*current, position))
                    if len(item) == _size:
                        new_frontier.add(item)
                        result.add(tuple(sorted(item)))
        frontier = new_frontier
    return sorted(result, key=lambda item: (len(item), item))


def enumerate_actions(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: np.asarray(numpy_helper.to_array(item)) for name, item in initializers.items()}
    actions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum" or len(node.input) < 3:
            continue
        eq = equation(node)
        if not eq or "->" not in eq:
            counters["unsupported_equation"] += 1
            continue
        left, output_labels = eq.split("->", 1)
        terms = left.split(",")
        if len(terms) != len(node.input):
            counters["term_arity_mismatch"] += 1
            continue
        positions = [
            position
            for position, name in enumerate(node.input)
            if name in initializers and "..." not in terms[position]
        ]
        for selected in connected_subsets(positions, terms):
            counters["subsets_considered"] += 1
            selected_arrays = [arrays[node.input[position]] for position in selected]
            if len({array.dtype for array in selected_arrays}) != 1:
                counters["mixed_dtype"] += 1
                continue
            spec = subset_spec(terms, selected, selected_arrays, output_labels)
            if spec is None:
                counters["bad_spec"] += 1
                continue
            result_labels, contracted, estimated_after = spec
            unique_names = sorted({node.input[position] for position in selected})
            available = sum(arrays[name].size for name in unique_names)
            if estimated_after > max(available * 4, 256):
                counters["size_pruned"] += 1
                continue
            exact = exact_combine(
                selected_arrays,
                [terms[position] for position in selected],
                result_labels,
                contracted,
            )
            if exact is None:
                counters["not_exact_class"] += 1
                continue
            combined, exact_class = exact
            if int(combined.size) != estimated_after:
                counters["shape_mismatch"] += 1
                continue
            action = {
                "id": len(actions),
                "node_index": node_index,
                "positions": selected,
                "occurrences": tuple((node_index, position) for position in selected),
                "input_names": tuple(node.input[position] for position in selected),
                "input_terms": tuple(terms[position] for position in selected),
                "result_labels": result_labels,
                "contracted_labels": contracted,
                "after_elements": int(combined.size),
                "exact_class": exact_class,
                "combined": np.ascontiguousarray(combined),
            }
            actions.append(action)
            counters[f"exact_{exact_class}"] += 1
    return actions, dict(counters)


def group_actions(actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    groups_by_key: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    groups: list[dict[str, Any]] = []
    action_group: dict[int, int] = {}
    for action in actions:
        array = action["combined"]
        key = (array.dtype.str, tuple(array.shape), array.tobytes())
        group_id = groups_by_key.get(key)
        if group_id is None:
            group_id = len(groups)
            groups_by_key[key] = group_id
            groups.append(
                {
                    "id": group_id,
                    "array": array,
                    "elements": int(array.size),
                    "action_ids": [],
                }
            )
        groups[group_id]["action_ids"].append(action["id"])
        action_group[action["id"]] = group_id
    return groups, action_group


def optimize_actions(
    model: onnx.ModelProto,
    actions: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    action_group: dict[int, int],
) -> dict[str, Any] | None:
    if not actions:
        return None
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    uses, _ = occurrence_map(model)
    relevant_names = sorted(
        {
            name
            for action in actions
            for name in action["input_names"]
            if name in arrays
        }
    )
    name_index = {name: index for index, name in enumerate(relevant_names)}
    action_count = len(actions)
    init_offset = action_count
    group_offset = init_offset + len(relevant_names)
    variable_count = group_offset + len(groups)
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:action_count] = 1e-6
    for name, index in name_index.items():
        objective[init_offset + index] = -float(arrays[name].size)
    for group in groups:
        objective[group_offset + group["id"]] = float(group["elements"])

    action_by_occurrence: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    coverage: defaultdict[tuple[str, int], int] = defaultdict(int)
    for action in actions:
        for occurrence, name in zip(action["occurrences"], action["input_names"]):
            action_by_occurrence[occurrence].append(action["id"])
            coverage[(name, action["id"])] += 1

    row_count = len(action_by_occurrence) + len(relevant_names) + len(actions)
    matrix = lil_matrix((row_count, variable_count), dtype=np.float64)
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.zeros(row_count, dtype=np.float64)
    row = 0
    for action_ids in action_by_occurrence.values():
        for action_id in action_ids:
            matrix[row, action_id] = 1
        upper[row] = 1
        row += 1
    for name, index in name_index.items():
        matrix[row, init_offset + index] = len(uses.get(name, []))
        for action in actions:
            count = coverage.get((name, action["id"]), 0)
            if count:
                matrix[row, action["id"]] = -count
        upper[row] = 0
        row += 1
    for action in actions:
        matrix[row, action["id"]] = 1
        matrix[row, group_offset + action_group[action["id"]]] = -1
        upper[row] = 0
        row += 1

    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 20.0},
    )
    if result.x is None:
        return None
    selected = [action["id"] for action in actions if result.x[action["id"]] > 0.5]
    removed = [
        name
        for name, index in name_index.items()
        if result.x[init_offset + index] > 0.5
    ]
    selected_groups = sorted({action_group[action_id] for action_id in selected})
    old_removed = sum(arrays[name].size for name in removed)
    new_elements = sum(groups[group_id]["elements"] for group_id in selected_groups)
    saving = int(old_removed - new_elements)
    if saving <= 0:
        return None
    return {
        "selected_action_ids": selected,
        "removed_initializers": removed,
        "selected_group_ids": selected_groups,
        "removed_elements": int(old_removed),
        "new_elements": int(new_elements),
        "parameter_saving": saving,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }


def build_candidate(
    source: onnx.ModelProto,
    actions: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    action_group: dict[int, int],
    solution: dict[str, Any],
) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    selected = {action_id: actions[action_id] for action_id in solution["selected_action_ids"]}
    group_names = {
        group_id: f"absorb240_g{group_id}"
        for group_id in solution["selected_group_ids"]
    }
    actions_by_node: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in selected.values():
        actions_by_node[action["node_index"]].append(action)

    for node_index, node_actions in actions_by_node.items():
        node = model.graph.node[node_index]
        eq = equation(node)
        if eq is None:
            raise RuntimeError("selected action lost equation")
        left, output_labels = eq.split("->", 1)
        terms = left.split(",")
        action_at_start: dict[int, dict[str, Any]] = {}
        covered: set[int] = set()
        for action in node_actions:
            start = min(action["positions"])
            action_at_start[start] = action
            covered.update(action["positions"])
        new_inputs: list[str] = []
        new_terms: list[str] = []
        for position, (name, term) in enumerate(zip(node.input, terms)):
            action = action_at_start.get(position)
            if action is not None:
                group_id = action_group[action["id"]]
                new_inputs.append(group_names[group_id])
                new_terms.append(action["result_labels"])
            elif position not in covered:
                new_inputs.append(name)
                new_terms.append(term)
        del node.input[:]
        node.input.extend(new_inputs)
        for attr in node.attribute:
            if attr.name == "equation":
                attr.s = (",".join(new_terms) + "->" + output_labels).encode()

    removed = set(solution["removed_initializers"])
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for group_id in solution["selected_group_ids"]:
        model.graph.initializer.append(
            numpy_helper.from_array(groups[group_id]["array"], name=group_names[group_id])
        )
    return model


def public_action(action: dict[str, Any], action_group: dict[int, int]) -> dict[str, Any]:
    return {
        "id": action["id"],
        "node_index": action["node_index"],
        "positions": list(action["positions"]),
        "input_names": list(action["input_names"]),
        "input_terms": list(action["input_terms"]),
        "result_labels": action["result_labels"],
        "contracted_labels": action["contracted_labels"],
        "after_elements": action["after_elements"],
        "exact_class": action["exact_class"],
        "shared_group": action_group[action["id"]],
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    costs = load_costs()
    inventory_rows: list[dict[str, Any]] = []
    leads: list[dict[str, Any]] = []
    authority_sha = sha256(AUTHORITY.read_bytes())
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            if task in EXCLUDED_TASKS or costs[task] <= 100:
                continue
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            model = onnx.load_model_from_string(data)
            einsums = [
                (index, node)
                for index, node in enumerate(model.graph.node)
                if node.op_type == "Einsum" and len(node.input) >= 3
            ]
            if not einsums:
                continue
            try:
                profile = static_profile(model)
                authority_strict_error = None
            except Exception as exc:  # noqa: BLE001
                profile = None
                authority_strict_error = f"{type(exc).__name__}: {exc}"
            actions, counters = enumerate_actions(model)
            groups, action_group = group_actions(actions)
            solution = optimize_actions(model, actions, groups, action_group)
            row: dict[str, Any] = {
                "task": task,
                "member_sha256": sha256(data),
                "authority_cost": costs[task],
                "static_profile": profile,
                "authority_strict_error": authority_strict_error,
                "node_count": len(model.graph.node),
                "einsum_count": len(einsums),
                "multi_operand_einsum_count": len(einsums),
                "max_einsum_inputs": max(len(node.input) for _, node in einsums),
                "single_node_output_only": bool(
                    len(model.graph.node) == 1
                    and len(einsums) == 1
                    and einsums[0][1].output[0] == model.graph.output[0].name
                ),
                "exact_action_count": len(actions),
                "shared_combined_group_count": len(groups),
                "enumeration": counters,
                "strict_lower_lead": solution is not None and authority_strict_error is None,
            }
            if solution is not None and authority_strict_error is None:
                candidate = build_candidate(model, actions, groups, action_group, solution)
                onnx.checker.check_model(candidate, full_check=True)
                shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                candidate_profile = static_profile(candidate)
                candidate_path = CANDIDATES / f"task{task:03d}_constant_absorb.onnx"
                onnx.save(candidate, candidate_path)
                candidate_data = candidate_path.read_bytes()
                solution.update(
                    {
                        "candidate": str(candidate_path.relative_to(ROOT)),
                        "candidate_sha256": sha256(candidate_data),
                        "candidate_serialized_bytes": len(candidate_data),
                        "candidate_static_profile": candidate_profile,
                        "memory_nonincrease": candidate_profile["memory"] <= profile["memory"],
                        "static_strict_lower": candidate_profile["cost"] < costs[task],
                        "selected_actions": [
                            public_action(actions[action_id], action_group)
                            for action_id in solution["selected_action_ids"]
                        ],
                    }
                )
                row["solution"] = solution
                if solution["memory_nonincrease"] and solution["static_strict_lower"]:
                    leads.append(
                        {
                            "task": task,
                            "authority_data": data,
                            "candidate_data": candidate_data,
                            "row": row,
                        }
                    )
            elif solution is not None:
                row["solution_rejected"] = {
                    "parameter_saving": solution["parameter_saving"],
                    "reason": "authority graph does not pass strict inference; no transform emitted",
                }
            inventory_rows.append(row)
            print(
                f"task{task:03d} einsum={len(einsums)} actions={len(actions)} "
                f"lead={None if solution is None else solution['parameter_saving']}",
                flush=True,
            )

    result = {
        "authority": AUTHORITY.name,
        "authority_sha256": authority_sha,
        "excluded_tasks": sorted(EXCLUDED_TASKS),
        "max_subset": MAX_SUBSET,
        "policy": {
            "approximate_factorization": False,
            "exact_classes": ["exact_integer", "exact_copy_sign_dyadic"],
            "global_use_accounting": True,
            "shared_combined_initializer_accounting": True,
            "memory_increase_allowed": False,
        },
        "summary": {
            "tasks_scanned": len(inventory_rows),
            "einsum_nodes_scanned": sum(row["einsum_count"] for row in inventory_rows),
            "single_node_output_only_tasks": sum(row["single_node_output_only"] for row in inventory_rows),
            "exact_actions": sum(row["exact_action_count"] for row in inventory_rows),
            "strict_lower_structural_leads": len(leads),
        },
        "rows": inventory_rows,
    }
    # Runtime raw-equivalence is performed by audit_survivors.py after this
    # structural scan, so authority/candidate bytes are intentionally omitted.
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
