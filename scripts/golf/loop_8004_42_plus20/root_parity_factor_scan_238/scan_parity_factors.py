#!/usr/bin/env python3
"""Exact small-tensor factor scan over the LB8009.46 all400 authority."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import string
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"
BUILD_RESULT = HERE / "scan.json"
SKIP_TASKS = {310}
EXCLUDED_STAGED_SHA256 = {
    310: "6ccf625a0dca41d5c9cb39ddb41c3756313f2a01ac95f38d70c880c677ccf858"
}
LETTERS = string.ascii_lowercase + string.ascii_uppercase

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def array_key(value: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    value = np.ascontiguousarray(value)
    return value.dtype.str, tuple(value.shape), value.tobytes()


def small_exact_array(value: np.ndarray) -> bool:
    if value.size < 8 or value.size > 256 or value.ndim < 2:
        return False
    if value.dtype.kind not in "fiu b".replace(" ", ""):
        return False
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        return False
    as_float = value.astype(np.float64)
    return bool(
        np.all(as_float == np.rint(as_float))
        and np.max(np.abs(as_float), initial=0.0) <= 16.0
    )


@dataclass
class Component:
    factor: np.ndarray
    axes: tuple[int, ...]
    latent: bool


@dataclass
class Factorization:
    kind: str
    components: list[Component]
    proof: dict[str, Any]

    @property
    def needs_latent(self) -> bool:
        return any(component.latent for component in self.components)

    @property
    def raw_factor_elements(self) -> int:
        seen: set[tuple[str, tuple[int, ...], bytes]] = set()
        total = 0
        for component in self.components:
            key = array_key(component.factor)
            if key not in seen:
                seen.add(key)
                total += int(component.factor.size)
        return total


def exact_rank1(value: np.ndarray) -> Factorization | None:
    unique = set(np.unique(value).astype(np.float64).tolist())
    if not unique.issubset({-1.0, 0.0, 1.0}) or not np.any(value):
        return None
    pivot = tuple(int(x) for x in np.argwhere(value != 0)[0])
    p = float(value[pivot])
    vectors: list[np.ndarray] = []
    for axis, size in enumerate(value.shape):
        vector = np.empty(size, dtype=value.dtype)
        for index in range(size):
            point = list(pivot)
            point[axis] = index
            vector[index] = value[tuple(point)] / p
        vectors.append(vector)
    vectors[0] = np.asarray(vectors[0] * p, dtype=value.dtype)
    rebuilt = vectors[0].astype(np.float64)
    for vector in vectors[1:]:
        rebuilt = np.multiply.outer(rebuilt, vector.astype(np.float64))
    rebuilt = np.asarray(rebuilt, dtype=value.dtype).reshape(value.shape)
    if not np.array_equal(rebuilt, value):
        return None
    return Factorization(
        "exact_rank1",
        [Component(vector, (axis,), False) for axis, vector in enumerate(vectors)],
        {"pivot": list(pivot), "pivot_value": p},
    )


def partition_rank1(value: np.ndarray) -> list[Factorization]:
    unique = set(np.unique(value).astype(np.float64).tolist())
    if not unique.issubset({-1.0, 0.0, 1.0}) or not np.any(value):
        return []
    ndim = value.ndim
    result: list[Factorization] = []
    axes = tuple(range(ndim))
    for count in range(1, ndim // 2 + 1):
        for left in itertools.combinations(axes, count):
            right = tuple(axis for axis in axes if axis not in left)
            if len(left) == len(right) and left > right:
                continue
            order = left + right
            matrix = np.transpose(value, order).reshape(
                math.prod(value.shape[axis] for axis in left), -1
            )
            positions = np.argwhere(matrix != 0)
            if not len(positions):
                continue
            row, col = (int(x) for x in positions[0])
            pivot = float(matrix[row, col])
            lhs = np.asarray(matrix[:, col], dtype=value.dtype)
            rhs = np.asarray(matrix[row, :] / pivot, dtype=value.dtype)
            if not np.array_equal(np.outer(lhs, rhs), matrix):
                continue
            lhs = lhs.reshape(tuple(value.shape[axis] for axis in left))
            rhs = rhs.reshape(tuple(value.shape[axis] for axis in right))
            result.append(
                Factorization(
                    "partition_rank1",
                    [Component(lhs, left, False), Component(rhs, right, False)],
                    {"left_axes": list(left), "right_axes": list(right)},
                )
            )
    return result


def equality_subsets(value: np.ndarray) -> list[Factorization]:
    if any(size != 2 for size in value.shape):
        return []
    if not np.array_equal(value, value.astype(bool).astype(value.dtype)):
        return []
    result: list[Factorization] = []
    axes = tuple(range(value.ndim))
    identity = np.eye(2, dtype=value.dtype)
    ones = np.ones(2, dtype=value.dtype)
    for count in range(3, value.ndim + 1):
        for selected in itertools.combinations(axes, count):
            expected = np.empty(value.shape, dtype=value.dtype)
            selected_set = set(selected)
            for index in np.ndindex(value.shape):
                bits = [index[axis] for axis in selected]
                expected[index] = int(all(bit == bits[0] for bit in bits))
            if not np.array_equal(value, expected):
                continue
            components = [Component(identity, (axis,), True) for axis in selected]
            components.extend(
                Component(ones, (axis,), False)
                for axis in axes
                if axis not in selected_set
            )
            result.append(
                Factorization(
                    "all_equality",
                    components,
                    {"selected_axes": list(selected)},
                )
            )
    return result


def walsh_low_term(value: np.ndarray) -> Factorization | None:
    if value.dtype.kind != "f" or any(size != 2 for size in value.shape):
        return None
    integers = value.astype(np.int64)
    if not np.array_equal(value, integers.astype(value.dtype)):
        return None
    ndim = value.ndim
    masks = list(np.ndindex((2,) * ndim))
    indices = list(np.ndindex(value.shape))
    coefficients: list[tuple[tuple[int, ...], int]] = []
    for mask in masks:
        coefficient = 0
        for index in indices:
            sign = -1 if sum(a * b for a, b in zip(mask, index)) % 2 else 1
            coefficient += int(integers[index]) * sign
        if coefficient:
            coefficients.append((mask, coefficient))
    if not 1 <= len(coefficients) <= 4:
        return None
    k = len(coefficients)
    factors: list[np.ndarray] = []
    components: list[Component] = []
    for axis in range(ndim):
        factor = np.asarray(
            [[1.0, -1.0 if mask[axis] else 1.0] for mask, _ in coefficients],
            dtype=value.dtype,
        )
        if np.all(factor == factor[0]):
            vector = np.asarray(factor[0], dtype=value.dtype)
            factors.append(vector)
            components.append(Component(vector, (axis,), False))
        else:
            factors.append(factor)
            components.append(Component(factor, (axis,), True))
    weights = np.asarray(
        [coefficient / value.size for _, coefficient in coefficients],
        dtype=value.dtype,
    )
    components.append(Component(weights, (), True))
    rebuilt = np.empty(value.shape, dtype=value.dtype)
    for index in indices:
        total = 0.0
        for term, (_, coefficient) in enumerate(coefficients):
            product = coefficient / value.size
            for axis in range(ndim):
                product *= float(factors[axis][term, index[axis]] if factors[axis].ndim == 2 else factors[axis][index[axis]])
            total += product
        rebuilt[index] = total
    if not np.array_equal(rebuilt, value):
        return None
    kind = "walsh_low_term"
    if k == 2 and set(np.unique(value).tolist()).issubset({0.0, 1.0}):
        kind = "parity_indicator"
    elif k == 1:
        kind = "signed_parity_rank1"
    return Factorization(
        kind,
        components,
        {
            "walsh_terms": k,
            "masks": [list(mask) for mask, _ in coefficients],
            "integer_coefficients": [coefficient for _, coefficient in coefficients],
            "normalizer": int(value.size),
        },
    )


def factorizations(value: np.ndarray) -> list[Factorization]:
    result: list[Factorization] = []
    rank1 = exact_rank1(value)
    if rank1 is not None:
        result.append(rank1)
    result.extend(partition_rank1(value))
    result.extend(equality_subsets(value))
    walsh = walsh_low_term(value)
    if walsh is not None:
        result.append(walsh)
    return result


def einsum_terms(node: onnx.NodeProto) -> tuple[list[str], str] | None:
    if node.op_type != "Einsum":
        return None
    attr = next((item for item in node.attribute if item.name == "equation"), None)
    if attr is None:
        return None
    equation = helper.get_attribute_value(attr).decode("ascii")
    lhs, output = equation.split("->")
    return lhs.split(","), output


def factor_cost(
    factorization: Factorization,
    existing: dict[tuple[str, tuple[int, ...], bytes], str],
) -> tuple[int, dict[tuple[str, tuple[int, ...], bytes], np.ndarray]]:
    new: dict[tuple[str, tuple[int, ...], bytes], np.ndarray] = {}
    for component in factorization.components:
        key = array_key(component.factor)
        if key not in existing:
            new[key] = np.ascontiguousarray(component.factor)
    return sum(int(value.size) for value in new.values()), new


def build_candidate(
    source: onnx.ModelProto,
    initializer_name: str,
    factorization: Factorization,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    target = initializers[initializer_name]
    existing = {
        array_key(value): name
        for name, value in initializers.items()
        if name != initializer_name
    }
    _, new_factors = factor_cost(factorization, existing)
    factor_names = dict(existing)
    for index, (key, value) in enumerate(new_factors.items()):
        name = f"pf238_{initializer_name}_{index}"
        while name in initializers or name in factor_names.values():
            name += "_x"
        model.graph.initializer.append(numpy_helper.from_array(value, name=name))
        factor_names[key] = name

    occurrences = 0
    for node in model.graph.node:
        parsed = einsum_terms(node)
        if parsed is None or initializer_name not in node.input:
            continue
        terms, output = parsed
        used = set("".join(terms) + output)
        free = [letter for letter in LETTERS if letter not in used]
        new_inputs: list[str] = []
        new_terms: list[str] = []
        for name, term in zip(node.input, terms):
            if name != initializer_name:
                new_inputs.append(name)
                new_terms.append(term)
                continue
            if len(term) != target.ndim or len(set(term)) != len(term):
                raise ValueError("unsupported repeated/mismatched Einsum term")
            latent = ""
            if factorization.needs_latent:
                if not free:
                    raise ValueError("no free Einsum label")
                latent = free.pop(0)
                used.add(latent)
            for component in factorization.components:
                key = array_key(component.factor)
                new_inputs.append(factor_names[key])
                component_term = "".join(term[axis] for axis in component.axes)
                new_terms.append((latent if component.latent else "") + component_term)
            occurrences += 1
        del node.input[:]
        node.input.extend(new_inputs)
        attr = next(item for item in node.attribute if item.name == "equation")
        attr.s = (",".join(new_terms) + "->" + output).encode("ascii")

    if occurrences == 0:
        raise ValueError("no replaced occurrence")
    kept = [item for item in model.graph.initializer if item.name != initializer_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, {
        "occurrences": occurrences,
        "new_factor_elements": sum(int(value.size) for value in new_factors.values()),
        "new_factor_shapes": [list(value.shape) for value in new_factors.values()],
        "reused_existing_factors": len(
            {
                array_key(component.factor)
                for component in factorization.components
                if array_key(component.factor) in existing
            }
        ),
    }


def profile(model: onnx.ModelProto, prefix: str) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as directory:
        path = Path(directory) / "model.onnx"
        onnx.save(model, path)
        return cost_of(str(path))


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    census = {
        "models": 0,
        "einsum_initializers": 0,
        "small_exact_initializers": 0,
        "factorizable_initializers": 0,
        "raw_factor_saving_opportunities": 0,
    }
    opportunity_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    initializer_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            source_sha = hashlib.sha256(data).hexdigest()
            model = onnx.load_model_from_string(data)
            census["models"] += 1
            if task in SKIP_TASKS:
                excluded_rows.append({
                    "task": task,
                    "authority_sha256": source_sha,
                    "excluded_staged_candidate_sha256": EXCLUDED_STAGED_SHA256[task],
                    "reason": "already staged exact parity-factor candidate",
                })
                continue
            init_arrays = {
                item.name: numpy_helper.to_array(item)
                for item in model.graph.initializer
            }
            all_uses: dict[str, list[tuple[int, int, str, str]]] = {
                name: [] for name in init_arrays
            }
            for node_index, node in enumerate(model.graph.node):
                parsed = einsum_terms(node)
                terms = parsed[0] if parsed else []
                for input_index, name in enumerate(node.input):
                    if name in all_uses:
                        term = terms[input_index] if parsed and input_index < len(terms) else ""
                        all_uses[name].append((node_index, input_index, node.op_type, term))
            for name, value in init_arrays.items():
                uses = all_uses[name]
                if not uses or not any(op == "Einsum" for _, _, op, _ in uses):
                    continue
                census["einsum_initializers"] += 1
                if not small_exact_array(value):
                    continue
                census["small_exact_initializers"] += 1
                census_row: dict[str, Any] = {
                    "task": task,
                    "authority_sha256": source_sha,
                    "initializer": name,
                    "shape": list(value.shape),
                    "elements": int(value.size),
                    "dtype": str(value.dtype),
                    "uses": len(uses),
                    "all_uses_are_einsum": all(op == "Einsum" for _, _, op, _ in uses),
                    "einsum_terms": sorted({term for _, _, _, term in uses}),
                    "unique_values": np.unique(value).astype(np.float64).tolist(),
                }
                initializer_rows.append(census_row)
                if any(op != "Einsum" for _, _, op, _ in uses):
                    census_row["factorization_status"] = "mixed_non_einsum_uses"
                    continue
                if any(len(term) != value.ndim or len(set(term)) != len(term) for _, _, _, term in uses):
                    census_row["factorization_status"] = "unsupported_einsum_term"
                    continue
                options = factorizations(value)
                if not options:
                    census_row["factorization_status"] = "no_exact_pattern_detected"
                    continue
                census["factorizable_initializers"] += 1
                existing = {
                    array_key(array): init_name
                    for init_name, array in init_arrays.items()
                    if init_name != name
                }
                ranked = []
                for factorization in options:
                    added, _ = factor_cost(factorization, existing)
                    saving = int(value.size) - added
                    ranked.append((saving, -added, factorization))
                    opportunity_rows.append({
                        "task": task,
                        "authority_sha256": source_sha,
                        "initializer": name,
                        "shape": list(value.shape),
                        "elements": int(value.size),
                        "dtype": str(value.dtype),
                        "uses": len(uses),
                        "kind": factorization.kind,
                        "raw_factor_elements": factorization.raw_factor_elements,
                        "actual_new_factor_elements": added,
                        "parameter_saving": saving,
                        "proof": factorization.proof,
                    })
                census_row["factorization_status"] = "exact_factorization_detected"
                census_row["factorizations"] = [
                    {
                        "kind": factorization.kind,
                        "raw_factor_elements": factorization.raw_factor_elements,
                        "proof": factorization.proof,
                    }
                    for factorization in options
                ]
                saving, _, best = max(ranked, key=lambda item: (item[0], item[1]))
                if saving <= 0:
                    continue
                census["raw_factor_saving_opportunities"] += 1
                try:
                    candidate, detail = build_candidate(model, name, best)
                    baseline_profile = profile(model, f"pf238_base_{task:03d}_")
                    candidate_profile = profile(candidate, f"pf238_cand_{task:03d}_")
                except Exception as exc:  # noqa: BLE001
                    candidate_rows.append({
                        "task": task,
                        "authority_sha256": source_sha,
                        "initializer": name,
                        "kind": best.kind,
                        "build_error": f"{type(exc).__name__}: {exc}",
                        "accepted_for_validation": False,
                    })
                    continue
                row: dict[str, Any] = {
                    "task": task,
                    "authority_sha256": source_sha,
                    "initializer": name,
                    "initializer_shape": list(value.shape),
                    "kind": best.kind,
                    "proof": best.proof,
                    "baseline": {"memory": baseline_profile[0], "params": baseline_profile[1], "cost": baseline_profile[2]},
                    "candidate": {"memory": candidate_profile[0], "params": candidate_profile[1], "cost": candidate_profile[2]},
                    "detail": detail,
                    "strict_lower": candidate_profile[2] < baseline_profile[2],
                }
                if row["strict_lower"]:
                    safe_name = "".join(character if character.isalnum() else "_" for character in name)
                    path = CANDIDATES / f"task{task:03d}_{safe_name}_{best.kind}.onnx"
                    onnx.save(candidate, path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    row.update({
                        "path": str(path.relative_to(ROOT)),
                        "sha256": digest,
                        "gain": baseline_profile[2] - candidate_profile[2],
                        "score_gain": math.log(baseline_profile[2] / candidate_profile[2]),
                        "accepted_for_validation": True,
                    })
                else:
                    row["accepted_for_validation"] = False
                candidate_rows.append(row)
            print(f"scan task{task:03d}", flush=True)
    candidate_rows.sort(key=lambda row: (-int(row.get("gain", 0)), row["task"], row["initializer"]))
    output = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "skip_tasks": sorted(SKIP_TASKS),
        "excluded_staged_candidates": excluded_rows,
        "census": census,
        "initializer_census": initializer_rows,
        "opportunities": opportunity_rows,
        "candidates": candidate_rows,
        "strict_lower_candidate_count": sum(bool(row.get("strict_lower")) for row in candidate_rows),
    }
    BUILD_RESULT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {BUILD_RESULT}", flush=True)


if __name__ == "__main__":
    main()
