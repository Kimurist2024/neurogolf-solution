#!/usr/bin/env python3
"""All-authority + active-71407 exact Einsum outer-factor scan."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
ACTIVE = ROOT / "others/71407"
ACTIVE_MANIFEST = ACTIVE / "MANIFEST.json"
RESULT = HERE / "scan_result.json"
CANDIDATES = HERE / "candidates"
FRESH_COUNT = 2000
FRESH_SEEDS = (270_000_001, 270_000_002)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "einsum_outer_screen_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "einsum_outer_trace_helpers",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_key(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    # np.ascontiguousarray promotes a rank-0 ndarray to shape (1,), so capture
    # the serialized ONNX rank before making the byte view contiguous.
    shape = tuple(int(value) for value in array.shape)
    contiguous = np.ascontiguousarray(array)
    return array.dtype.str, shape, contiguous.tobytes()


def key_id(key: tuple[str, tuple[int, ...], bytes]) -> str:
    dtype, shape, payload = key
    h = hashlib.sha256()
    h.update(dtype.encode())
    h.update(repr(shape).encode())
    h.update(payload)
    return h.hexdigest()


def key_size(key: tuple[str, tuple[int, ...], bytes]) -> int:
    return math.prod(key[1]) if key[1] else 1


def direction_key(array: np.ndarray):
    """Scale-free exact-rational tensor direction (an optimistic dedupe key)."""
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ) or not np.all(np.isfinite(array)):
        return None
    fractions = [exact_fraction(value, array.dtype) for value in array.reshape(-1)]
    pivot = next((value for value in fractions if value), None)
    if pivot is None:
        normalized = tuple((0, 1) for _ in fractions)
    else:
        normalized = tuple(
            ((value / pivot).numerator, (value / pivot).denominator)
            for value in fractions
        )
    return array.dtype.str, tuple(int(value) for value in array.shape), normalized


def exact_fraction(value: Any, dtype: np.dtype) -> Fraction:
    if np.issubdtype(dtype, np.integer):
        return Fraction(int(value))
    return Fraction.from_float(float(value))


def exact_cast(value: Fraction, dtype: np.dtype):
    try:
        if np.issubdtype(dtype, np.integer):
            if value.denominator != 1:
                return None
            info = np.iinfo(dtype)
            integer = value.numerator
            return dtype.type(integer) if info.min <= integer <= info.max else None
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = dtype.type(float(value))
        if not np.isfinite(converted):
            return None
        return converted if exact_fraction(converted, dtype) == value else None
    except (OverflowError, ValueError, TypeError):
        return None


def candidate_scales(
    pivot: Fraction,
    row: list[Fraction],
    column: list[Fraction],
    dtype: np.dtype,
) -> set[Fraction]:
    scales = {Fraction(1), Fraction(-1), pivot, -pivot}
    scales.update(value for value in row + column if value)
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        for exponent in range(info.minexp - 1, info.maxexp):
            power = Fraction(2) ** exponent
            scales.update((power, -power))
    return scales


def exact_outer_options(
    array: np.ndarray, axes_a: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Return every exact serialized factor pair found for one bipartition.

    Mathematical rank one is checked over the exact rational values encoded by
    the initializer.  A pair survives only if both factors are exactly
    representable in the original dtype and dtype multiplication reconstructs
    the original contiguous bytes exactly (including signed zero bits).
    """

    dtype = array.dtype
    rank = array.ndim
    axes_b = tuple(axis for axis in range(rank) if axis not in axes_a)
    permutation = axes_a + axes_b
    shape_a = tuple(int(array.shape[axis]) for axis in axes_a)
    shape_b = tuple(int(array.shape[axis]) for axis in axes_b)
    size_a, size_b = math.prod(shape_a), math.prod(shape_b)
    matrix = np.transpose(array, permutation).reshape((size_a, size_b))
    nonzero = np.argwhere(matrix != 0)
    raw_target = np.ascontiguousarray(array).tobytes()
    pairs: dict[tuple[Any, Any], dict[str, Any]] = {}

    if nonzero.size == 0:
        candidates = [(np.zeros(shape_a, dtype=dtype), np.ones(shape_b, dtype=dtype))]
    else:
        pivot_i, pivot_j = (int(value) for value in nonzero[0])
        pivot = exact_fraction(matrix[pivot_i, pivot_j], dtype)
        row = [exact_fraction(value, dtype) for value in matrix[pivot_i]]
        column = [exact_fraction(value, dtype) for value in matrix[:, pivot_j]]
        for i in range(size_a):
            for j in range(size_b):
                if exact_fraction(matrix[i, j], dtype) * pivot != column[i] * row[j]:
                    return []
        candidates = []
        for scale in candidate_scales(pivot, row, column, dtype):
            factor_a_values = [exact_cast(value * scale / pivot, dtype) for value in column]
            factor_b_values = [exact_cast(value / scale, dtype) for value in row]
            if any(value is None for value in factor_a_values + factor_b_values):
                continue
            candidates.append(
                (
                    np.asarray(factor_a_values, dtype=dtype).reshape(shape_a),
                    np.asarray(factor_b_values, dtype=dtype).reshape(shape_b),
                )
            )

    inverse = np.argsort(permutation)
    for factor_a, factor_b in candidates:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            reconstructed = np.multiply(
                factor_a.reshape((size_a, 1)),
                factor_b.reshape((1, size_b)),
                dtype=dtype,
            ).reshape(shape_a + shape_b).transpose(inverse)
        if np.ascontiguousarray(reconstructed).tobytes() != raw_target:
            continue
        key_a, key_b = tensor_key(factor_a), tensor_key(factor_b)
        pairs.setdefault(
            (key_a, key_b),
            {
                "axes_a": axes_a,
                "axes_b": axes_b,
                "factor_a": factor_a,
                "factor_b": factor_b,
                "key_a": key_a,
                "key_b": key_b,
                "direction_a": direction_key(factor_a),
                "direction_b": direction_key(factor_b),
                "factor_params_raw": key_size(key_a) + (
                    0 if key_b == key_a else key_size(key_b)
                ),
            },
        )
    return list(pairs.values())


def synthetic_self_tests() -> list[dict[str, Any]]:
    cases = [
        (
            "float32_rank1",
            np.asarray([[2, 4, 6], [3, 6, 9]], dtype=np.float32),
            (0,),
            True,
        ),
        (
            "float16_rank1_3d",
            np.multiply(
                np.asarray([1, -2], dtype=np.float16).reshape(2, 1, 1),
                np.asarray([[1, 2], [3, 4]], dtype=np.float16).reshape(1, 2, 2),
                dtype=np.float16,
            ),
            (0,),
            True,
        ),
        (
            "int32_rank1",
            np.asarray([[2, 4], [6, 12]], dtype=np.int32),
            (0,),
            True,
        ),
        (
            "non_rank1_reject",
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            (0,),
            False,
        ),
    ]
    rows = []
    for name, array, axes, expected in cases:
        options = exact_outer_options(array, axes)
        actual = bool(options)
        rows.append(
            {
                "name": name,
                "expected_factorable": expected,
                "actual_factorable": actual,
                "option_count": len(options),
                "passed": actual == expected,
            }
        )
    if not all(row["passed"] for row in rows):
        raise AssertionError(rows)
    return rows


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            try:
                return attr.s.decode("ascii")
            except UnicodeDecodeError:
                return None
    return None


def equation_terms(value: str) -> tuple[list[str], str | None] | None:
    if value.count("->") > 1:
        return None
    if "->" in value:
        lhs, rhs = value.split("->")
    else:
        lhs, rhs = value, None
    terms = lhs.split(",")
    if any("..." in term or not all(char.isalpha() for char in term) for term in terms):
        return None
    return terms, rhs


def public_option(option: dict[str, Any]) -> dict[str, Any]:
    return {
        "axes_a": list(option["axes_a"]),
        "axes_b": list(option["axes_b"]),
        "factor_a": {
            "shape": list(option["factor_a"].shape),
            "dtype": str(option["factor_a"].dtype),
            "params": int(option["factor_a"].size),
            "key_sha256": key_id(option["key_a"]),
        },
        "factor_b": {
            "shape": list(option["factor_b"].shape),
            "dtype": str(option["factor_b"].dtype),
            "params": int(option["factor_b"].size),
            "key_sha256": key_id(option["key_b"]),
        },
        "factor_params_raw": int(option["factor_params_raw"]),
    }


def discover_internal(task: int, lineage: str, data: bytes):
    model = onnx.load_model_from_string(data)
    initializer_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    initializer_protos = {item.name: item for item in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    uses: dict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    einsum_count = 0
    for node_index, node in enumerate(model.graph.node):
        einsum_count += int(node.op_type == "Einsum")
        for slot, name in enumerate(node.input):
            if name:
                uses[name].append((node_index, slot, node))

    eligible = factorable = partition_count = option_count = 0
    reason_counts = Counter()
    groups = []
    for name, array in initializer_arrays.items():
        user_slots = uses.get(name, [])
        if not user_slots:
            continue
        if name in graph_inputs:
            reason_counts["initializer_is_graph_input"] += 1
            continue
        if not all(node.op_type == "Einsum" for _, _, node in user_slots):
            reason_counts["shared_non_einsum_use"] += 1
            continue
        if array.ndim < 2:
            reason_counts["rank_below_two"] += 1
            continue
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            reason_counts["unsupported_nonreal_dtype"] += 1
            continue
        if not np.all(np.isfinite(array)):
            reason_counts["nonfinite_initializer"] += 1
            continue
        compatible_uses = []
        compatible = True
        for node_index, slot, node in user_slots:
            parsed = equation_terms(equation(node) or "")
            if parsed is None:
                compatible = False
                reason_counts["unsupported_equation_syntax"] += 1
                break
            terms, _ = parsed
            if len(terms) != len(node.input) or slot >= len(terms) or len(terms[slot]) != array.ndim:
                compatible = False
                reason_counts["operand_term_rank_mismatch"] += 1
                break
            compatible_uses.append(
                {"node_index": node_index, "slot": slot, "term": terms[slot]}
            )
        if not compatible:
            continue
        eligible += 1
        options = []
        # Axis bipartitions are unordered; fixing axis zero in A visits every
        # nontrivial bipartition exactly once.
        for mask in range(1 << (array.ndim - 1)):
            axes_a = (0,) + tuple(
                axis
                for axis in range(1, array.ndim)
                if mask & (1 << (axis - 1))
            )
            if len(axes_a) == array.ndim:
                continue
            partition_count += 1
            options.extend(exact_outer_options(array, axes_a))
        if not options:
            reason_counts["no_exact_rank1_axis_bipartition"] += 1
            continue
        factorable += 1
        option_count += len(options)
        groups.append(
            {
                "initializer": name,
                "initializer_key": tensor_key(array),
                "initializer_params": int(array.size),
                "initializer_shape": list(array.shape),
                "initializer_dtype": str(array.dtype),
                "uses": compatible_uses,
                "options": options,
            }
        )

    public = {
        "task": task,
        "lineage": lineage,
        "source_sha256": sha256(data),
        "einsum_count": einsum_count,
        "eligible_all_use_initializers": eligible,
        "factorable_initializers": factorable,
        "axis_bipartitions_checked": partition_count,
        "exact_factor_options": option_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "groups": [
            {
                key: value
                for key, value in group.items()
                if key not in ("initializer_key", "options")
            }
            | {
                "option_count": len(group["options"]),
                "best_raw_factor_params": min(
                    option["factor_params_raw"] for option in group["options"]
                ),
                "best_raw_param_delta": group["initializer_params"]
                - min(option["factor_params_raw"] for option in group["options"]),
            }
            for group in groups
        ],
    }
    return model, initializer_arrays, groups, public


def prune_dominated_states(states, weights):
    """Exact future-cost dominance over already materialized shared factors."""
    result = {}
    for keys, payload in sorted(states.items(), key=lambda item: (len(item[0]), item[1][0])):
        cost, choices = payload
        key_list = tuple(keys)
        dominated = False
        for subset_size in range(len(key_list)):
            for subset_tuple in itertools.combinations(key_list, subset_size):
                subset = frozenset(subset_tuple)
                prior = result.get(subset)
                if prior is None:
                    continue
                # The subset state can materialize every extra factor now (or
                # later) and still cost no more, so the superset is redundant.
                if prior[0] + sum(weights[key] for key in keys - subset) <= cost:
                    dominated = True
                    break
            if dominated:
                break
        if not dominated:
            current = result.get(keys)
            if current is None or cost < current[0]:
                result[keys] = payload
    return result


def optimize_groups(initializer_arrays, groups, *, optimistic_directions: bool = False):
    if not groups:
        return None, {"subsets_checked": 0, "dp_states_peak": 0}
    all_existing: dict[Any, list[str]] = defaultdict(list)
    for name, array in initializer_arrays.items():
        key = direction_key(array) if optimistic_directions else tensor_key(array)
        if key is not None:
            all_existing[key].append(name)
    best = None
    subsets_checked = states_peak = 0
    for count in range(1, len(groups) + 1):
        for indices in itertools.combinations(range(len(groups)), count):
            subsets_checked += 1
            selected = [groups[index] for index in indices]
            removed = {group["initializer"] for group in selected}
            existing_keys = {
                key
                for key, names in all_existing.items()
                if any(name not in removed for name in names)
            }
            presence = Counter()
            for group in selected:
                group_keys = {
                    key
                    for option in group["options"]
                    for key in (
                        (option["direction_a"], option["direction_b"])
                        if optimistic_directions
                        else (option["key_a"], option["key_b"])
                    )
                }
                presence.update(group_keys)
            shared = {
                key for key, frequency in presence.items() if frequency >= 2 and key not in existing_keys
            }
            weights = {key: key_size(key) for key in shared}
            compressed_options = []
            for group in selected:
                compressed: dict[frozenset, tuple[int, dict[str, Any]]] = {}
                for option in group["options"]:
                    distinct = set(
                        (option["direction_a"], option["direction_b"])
                        if optimistic_directions
                        else (option["key_a"], option["key_b"])
                    )
                    shared_set = frozenset(key for key in distinct if key in shared)
                    local_cost = sum(
                        key_size(key)
                        for key in distinct
                        if key not in shared and key not in existing_keys
                    )
                    current = compressed.get(shared_set)
                    if current is None or local_cost < current[0]:
                        compressed[shared_set] = (local_cost, option)
                compressed_options.append(compressed)

            states = {frozenset(): (0, [])}
            for options in compressed_options:
                next_states = {}
                for materialized, (cost_so_far, choices) in states.items():
                    for option_keys, (local_cost, option) in options.items():
                        union = materialized | option_keys
                        cost = cost_so_far + local_cost + sum(
                            weights[key] for key in option_keys - materialized
                        )
                        current = next_states.get(union)
                        if current is None or cost < current[0]:
                            next_states[union] = (cost, choices + [option])
                states_peak = max(states_peak, len(next_states))
                states = prune_dominated_states(next_states, weights)
            factor_cost, choices = min(states.values(), key=lambda value: value[0])
            removed_params = sum(group["initializer_params"] for group in selected)
            saving = removed_params - factor_cost
            candidate = {
                "group_indices": list(indices),
                "groups": selected,
                "choices": choices,
                "removed_params": int(removed_params),
                "added_factor_params_after_dedupe": int(factor_cost),
                "projected_param_saving": int(saving),
            }
            if best is None or saving > best["projected_param_saving"] or (
                saving == best["projected_param_saving"] and len(selected) > len(best["groups"])
            ):
                best = candidate
    return best, {"subsets_checked": subsets_checked, "dp_states_peak": states_peak}


def public_optimization(best, stats):
    if best is None:
        return {**stats, "selected_group_count": 0, "projected_param_saving": 0}
    return {
        **stats,
        "selected_group_count": len(best["groups"]),
        "selected_initializers": [group["initializer"] for group in best["groups"]],
        "removed_params": best["removed_params"],
        "added_factor_params_after_dedupe": best["added_factor_params_after_dedupe"],
        "projected_param_saving": best["projected_param_saving"],
        "selected_options": [public_option(option) for option in best["choices"]],
    }


def unique_name(existing: set[str], base: str) -> str:
    name, suffix = base, 0
    while name in existing:
        suffix += 1
        name = f"{base}_{suffix}"
    existing.add(name)
    return name


def build_candidate(data: bytes, best) -> tuple[bytes, dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    initializer_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    removed = {group["initializer"] for group in best["groups"]}
    existing_names = {
        name
        for node in model.graph.node
        for name in list(node.input) + list(node.output)
        if name
    } | set(initializer_arrays)
    key_names: dict[Any, str] = {}
    for name, array in initializer_arrays.items():
        if name not in removed:
            key_names.setdefault(tensor_key(array), name)
    added = []
    factor_names = []
    for group, option in zip(best["groups"], best["choices"]):
        names = []
        for label, factor, key in (
            ("a", option["factor_a"], option["key_a"]),
            ("b", option["factor_b"], option["key_b"]),
        ):
            if key not in key_names:
                name = unique_name(existing_names, f"outer_{group['initializer']}_{label}")
                model.graph.initializer.append(numpy_helper.from_array(factor, name=name))
                key_names[key] = name
                added.append(
                    {
                        "name": name,
                        "shape": list(factor.shape),
                        "dtype": str(factor.dtype),
                        "params": int(factor.size),
                        "key_sha256": key_id(key),
                    }
                )
            names.append(key_names[key])
        factor_names.append(tuple(names))
    selected = {
        group["initializer"]: (option, names)
        for group, option, names in zip(best["groups"], best["choices"], factor_names)
    }

    rewritten_nodes = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum" or not any(name in selected for name in node.input):
            continue
        raw_equation = equation(node)
        assert raw_equation is not None
        parsed = equation_terms(raw_equation)
        assert parsed is not None
        terms, rhs = parsed
        new_inputs, new_terms = [], []
        for term, input_name in zip(terms, node.input):
            if input_name not in selected:
                new_inputs.append(input_name)
                new_terms.append(term)
                continue
            option, names = selected[input_name]
            term_a = "".join(term[axis] for axis in option["axes_a"])
            term_b = "".join(term[axis] for axis in option["axes_b"])
            new_inputs.extend(names)
            new_terms.extend((term_a, term_b))
        rewritten_equation = ",".join(new_terms) + (("->" + rhs) if rhs is not None else "")
        del node.input[:]
        node.input.extend(new_inputs)
        for attr in node.attribute:
            if attr.name == "equation":
                attr.s = rewritten_equation.encode("ascii")
        node.doc_string = "exact serialized outer-product constant factorization"
        rewritten_nodes.append(
            {"node_index": node_index, "old_equation": raw_equation, "new_equation": rewritten_equation}
        )

    kept = [copy.deepcopy(item) for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model.SerializeToString(), {
        "removed_initializers": sorted(removed),
        "added_factor_initializers": added,
        "factor_key_reuse_count": 2 * len(best["groups"]) - len(added),
        "rewritten_nodes": rewritten_nodes,
        "params_after_build": scoring.calculate_params(model),
    }


def full_validation(data: bytes) -> dict[str, Any]:
    row = {"checker_full": False, "shape_inference_strict_data_prop": False}
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        row["checker_error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["shape_inference_strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["shape_inference_error"] = f"{type(exc).__name__}: {exc}"
    return row


def score(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"einsum_outer_{task:03d}_{label}_") as wd:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, wd, label=label, require_correct=False
        )


def known_raw_four(task: int, baseline: bytes, candidate: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    result: dict[str, Any] = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            stats = {
                "total": 0,
                "candidate_right": 0,
                "raw_equal": 0,
                "threshold_equal": 0,
                "baseline_errors": 0,
                "candidate_errors": 0,
            }
            try:
                base_session = SCREEN.make_session(baseline, disabled, threads)
                cand_session = SCREEN.make_session(candidate, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["session_error"] = f"{type(exc).__name__}: {exc}"
                result[key] = stats
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    stats["total"] += 1
                    try:
                        base = base_session.run(
                            None, {base_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["baseline_errors"] += 1
                        continue
                    try:
                        cand = cand_session.run(
                            None, {cand_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["candidate_errors"] += 1
                        continue
                    stats["candidate_right"] += int(
                        np.array_equal(cand > 0, benchmark["output"] > 0)
                    )
                    stats["raw_equal"] += int(np.array_equal(cand, base))
                    stats["threshold_equal"] += int(np.array_equal(cand > 0, base > 0))
            result[key] = stats
    return result


def known_pass(report: dict[str, Any]) -> bool:
    return len(report) == 4 and all(
        row.get("total", 0) > 0
        and row.get("candidate_right") == row.get("total")
        and row.get("raw_equal") == row.get("total")
        and row.get("threshold_equal") == row.get("total")
        and row.get("baseline_errors") == 0
        and row.get("candidate_errors") == 0
        and not row.get("session_error")
        for row in report.values()
    )


def fresh_four(task: int, baseline: bytes, candidate: bytes) -> dict[str, Any]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    configs = (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    )
    sessions = {
        name: (
            SCREEN.make_session(baseline, disabled, threads),
            SCREEN.make_session(candidate, disabled, threads),
        )
        for disabled, threads, name in configs
    }
    runs = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {
            name: {"right": 0, "wrong": 0, "raw_equal": 0, "errors": 0}
            for _, _, name in configs
        }
        valid = attempts = generation_errors = conversion_skips = 0
        while valid < FRESH_COUNT:
            attempts += 1
            try:
                benchmark = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if benchmark is None:
                conversion_skips += 1
                continue
            valid += 1
            want = benchmark["output"] > 0
            for _, _, name in configs:
                try:
                    base_session, cand_session = sessions[name]
                    feeds = {base_session.get_inputs()[0].name: benchmark["input"]}
                    base = base_session.run(None, feeds)[0]
                    cand = cand_session.run(
                        None, {cand_session.get_inputs()[0].name: benchmark["input"]}
                    )[0]
                    stats[name]["right" if np.array_equal(cand > 0, want) else "wrong"] += 1
                    stats[name]["raw_equal"] += int(np.array_equal(cand, base))
                except Exception:  # noqa: BLE001
                    stats[name]["errors"] += 1
        runs.append(
            {
                "seed": seed,
                "valid": valid,
                "attempts": attempts,
                "generation_errors": generation_errors,
                "conversion_skips": conversion_skips,
                "configs": stats,
            }
        )
    return {"count_per_seed": FRESH_COUNT, "seeds": list(FRESH_SEEDS), "runs": runs}


def main() -> None:
    ort.set_default_logger_severity(4)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority archive changed")
    self_tests = synthetic_self_tests()
    manifest = json.loads(ACTIVE_MANIFEST.read_text())
    active_rows = {int(row["task"]): row for row in manifest["active_candidates"]}
    active_payloads = {
        task: (ACTIVE / row["file"]).read_bytes() for task, row in active_rows.items()
    }
    if any(sha256(active_payloads[task]) != row["sha256"] for task, row in active_rows.items()):
        raise RuntimeError("active 71407 manifest/payload SHA mismatch")
    CANDIDATES.mkdir(parents=True, exist_ok=True)

    authority_payloads = {}
    source_records = []
    internal_records = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            authority_payloads[task] = archive.read(f"task{task:03d}.onnx")
    for lineage, payload_map in (("authority", authority_payloads), ("active71407", active_payloads)):
        for task, data in sorted(payload_map.items()):
            model, init_arrays, groups, public = discover_internal(task, lineage, data)
            best, optimization_stats = optimize_groups(init_arrays, groups)
            optimistic_best, optimistic_stats = optimize_groups(
                init_arrays, groups, optimistic_directions=True
            )
            public["optimization"] = public_optimization(best, optimization_stats)
            public["optimistic_scale_free_dedupe_upper_bound"] = public_optimization(
                optimistic_best, optimistic_stats
            )
            public["candidate_possible"] = bool(
                best is not None and best["projected_param_saving"] > 0
            )
            source_records.append(public)
            internal_records.append(
                {
                    "task": task,
                    "lineage": lineage,
                    "data": data,
                    "best": best,
                    "public": public,
                }
            )

    # Build every source-lineage parameter winner, then compare official cost
    # against the active descendant for that task when one exists.
    candidate_rows = []
    for record in internal_records:
        best = record["best"]
        if best is None or best["projected_param_saving"] <= 0:
            continue
        task, lineage, source = record["task"], record["lineage"], record["data"]
        baseline_lineage = "active71407" if task in active_payloads else "authority"
        baseline = active_payloads.get(task, authority_payloads[task])
        candidate, build = build_candidate(source, best)
        path = CANDIDATES / f"task{task:03d}_{lineage}_einsum_outer.onnx"
        onnx.save_model(onnx.load_model_from_string(candidate), path)
        candidate = path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "source_lineage": lineage,
            "baseline_lineage": baseline_lineage,
            "path": str(path.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "baseline_sha256": sha256(baseline),
            "candidate_sha256": sha256(candidate),
            "optimization": record["public"]["optimization"],
            "build_metadata": build,
            "baseline_score": score(task, baseline, "baseline"),
            "candidate_score": score(task, candidate, "candidate"),
            "full_validation": full_validation(candidate),
            "strict": SCREEN.structural_audit(candidate),
        }
        try:
            trace = TRACE.runtime_shape_trace(task, onnx.load_model_from_string(candidate))
            row["runtime_shape"] = trace
            row["truthful"] = not trace["declared_actual_mismatches"]
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_error"] = f"{type(exc).__name__}: {exc}"
            row["truthful"] = False
        base_score, cand_score = row["baseline_score"], row["candidate_score"]
        row["strictly_lower_than_active_baseline"] = bool(
            base_score and cand_score and cand_score["cost"] < base_score["cost"]
        )
        full_ok = row["full_validation"]["checker_full"] and row["full_validation"][
            "shape_inference_strict_data_prop"
        ]
        if not row["strictly_lower_than_active_baseline"]:
            row["decision"] = "REJECT_NOT_LOWER_THAN_ACTIVE_BASELINE"
        elif not full_ok or not row["strict"]["pass"] or not row["strict"].get("conv_bias_ub0"):
            row["decision"] = "REJECT_FULL_STRICT_SCHEMA_OR_UB"
        else:
            known = known_raw_four(task, baseline, candidate)
            row["known_raw_four"] = known
            row["known_raw_four_pass"] = known_pass(known)
            if not cand_score.get("correct"):
                row["decision"] = "REJECT_OFFICIAL_KNOWN"
            elif not row["truthful"]:
                row["decision"] = "REJECT_RUNTIME_SHAPE"
            elif not row["known_raw_four_pass"]:
                row["decision"] = "REJECT_KNOWN_RAW_OR_RUNTIME"
            else:
                fresh = fresh_four(task, baseline, candidate)
                row["fresh"] = fresh
                row["fresh_pass"] = all(
                    config["right"] == FRESH_COUNT
                    and config["wrong"] == 0
                    and config["raw_equal"] == FRESH_COUNT
                    and config["errors"] == 0
                    for run in fresh["runs"]
                    for config in run["configs"].values()
                )
                row["decision"] = "ACCEPT" if row["fresh_pass"] else "REJECT_FRESH"
        candidate_rows.append(row)
        print(
            f"task{task:03d} source={lineage} projected=-{best['projected_param_saving']} "
            f"cost={base_score}->{cand_score} decision={row['decision']}",
            flush=True,
        )

    accepted = [row for row in candidate_rows if row["decision"] == "ACCEPT"]
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "active_stage": str(ACTIVE.relative_to(ROOT)),
        "active_manifest_sha256": sha256(ACTIVE_MANIFEST.read_bytes()),
        "active_descendant_count": len(active_payloads),
        "authority_tasks_scanned": 400,
        "synthetic_self_tests": self_tests,
        "source_models_scanned": len(source_records),
        "einsum_nodes": sum(row["einsum_count"] for row in source_records),
        "eligible_all_use_initializers": sum(
            row["eligible_all_use_initializers"] for row in source_records
        ),
        "factorable_initializers": sum(row["factorable_initializers"] for row in source_records),
        "axis_bipartitions_checked": sum(
            row["axis_bipartitions_checked"] for row in source_records
        ),
        "exact_factor_options": sum(row["exact_factor_options"] for row in source_records),
        "source_records": source_records,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "accepted": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["candidate_sha256"],
                "baseline_lineage": row["baseline_lineage"],
                "baseline_cost": row["baseline_score"]["cost"],
                "candidate_cost": row["candidate_score"]["cost"],
            }
            for row in accepted
        ],
        "decision": "ACCEPT" if accepted else "NO_SAFE_EINSUM_OUTER_FACTOR_WINNER",
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_models_scanned": report["source_models_scanned"],
                "einsum_nodes": report["einsum_nodes"],
                "eligible_all_use_initializers": report["eligible_all_use_initializers"],
                "factorable_initializers": report["factorable_initializers"],
                "axis_bipartitions_checked": report["axis_bipartitions_checked"],
                "exact_factor_options": report["exact_factor_options"],
                "candidate_count": report["candidate_count"],
                "accepted": len(accepted),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
