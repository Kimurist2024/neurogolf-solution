#!/usr/bin/env python3
"""Exhaustively search arbitrary equal-code reuse in Wave15 task226.

Unlike ``search_task226_reuse.py``, this search does not require the shared
initializer value to equal either incumbent value.  It compiles the finite
generator domain into symbolic row/column code records, enumerates all 256
scalar values and all 65,536 two-byte row-code values for each possible pair,
then verifies every surviving model in ORT on the complete finite domain.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import search_task226_reuse as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = {
    "Z": 0,
    "RB": 16,
    "RF0": 17,
    "RF1": 18,
    "RM0": 20,
    "RM1": 80,
    "RL0": 24,
    "RL1": 48,
    "CF": 19,
    "CM": 84,
    "CL": 56,
}
TOKEN_ORDER = tuple(CURRENT)
TOKEN_INDEX = {name: index for index, name in enumerate(TOKEN_ORDER)}


def symbolic_records(cases):
    records: set[tuple[str, str, str, int]] = set()
    for example, _, _ in cases:
        grid = np.asarray(example["input"], dtype=np.int64)
        labels = np.asarray(example["output"], dtype=np.int64)
        r1, r3, r6, r8 = (grid[row, 0] == 0 for row in (1, 3, 6, 8))
        c1, c2, c3, c4, c5, c6, c7, c8 = (grid[0, col] == 0 for col in range(1, 9))

        def pair(prefix: str) -> tuple[str, str]:
            return prefix + "0", prefix + "1"

        rf, rm, rl, rb, zero = pair("RF"), pair("RM"), pair("RL"), ("RB", "RB"), ("Z", "Z")
        rp2_i = zero if r3 else rf
        rp3_i = rm if r8 else rb
        rp6_i = rm if r1 else rb
        rp7_i = zero if r6 else rl
        rows = [
            rf,
            rf if r1 else zero,
            rp2_i if r1 else rb,
            rp3_i if r3 else zero,
            zero if (r3 and not r8) else rm,
            zero if (r6 and not r1) else rm,
            rp6_i if r6 else zero,
            rp7_i if r8 else rb,
            rl if r8 else zero,
            rl,
        ]

        c4_mb = "CM" if c4 else "RB"
        cp2_i = c4_mb if c3 else "RB"
        cp2_j = "CF" if c1 else cp2_i
        cp3_j = cp2_j if c2 else c4_mb
        c8_lb = "CL" if c8 else "RB"
        cp6_i = c8_lb if c7 else "RB"
        cp6_j = "CM" if c5 else cp6_i
        cp7_j = cp6_j if c6 else c8_lb
        cols = [
            "CF",
            "CF" if c1 else "Z",
            cp2_j if c2 else "Z",
            cp3_j if c3 else "Z",
            "CM" if c4 else "Z",
            "CM" if c5 else "Z",
            cp6_j if c6 else "Z",
            cp7_j if c7 else "Z",
            "CL" if c8 else "Z",
            "CL",
        ]
        for row in range(10):
            for col in range(10):
                records.add((rows[row][0], rows[row][1], cols[col], int(labels[row, col])))
    return sorted(records)


def points_from_values(records, values: dict[str, int]):
    mapping: dict[tuple[int, int], set[int]] = {}
    for first_token, second_token, col_token, label in records:
        point = (values[first_token] & values[col_token], values[second_token] & values[col_token])
        mapping.setdefault(point, set()).add(label)
    if any(len(labels) != 1 for labels in mapping.values()):
        return None
    ordered = sorted((point[0], point[1], next(iter(labels))) for point, labels in mapping.items())
    points = np.asarray([[first, second] for first, second, _ in ordered], dtype=np.int32)
    labels = np.asarray([label for _, _, label in ordered], dtype=np.int64)
    return points, labels


def fit_weights(points: np.ndarray, labels: np.ndarray, x_zp: int, w_zp: int, scale: float):
    centered = points - x_zp
    byte = np.arange(256, dtype=np.int32)
    weights = np.stack(np.meshgrid(byte, byte, indexing="ij"), axis=-1).reshape(-1, 2)
    deltas = weights - w_zp
    scores = centered @ deltas.T
    max_off = math.ceil(0.5 / scale) - 1
    min_on = max_off + 1
    fitted = np.full((10, 2, 1, 1), w_zp, dtype=np.uint8)
    details = {}
    for label in (0, 1, 2, 3, 5):
        on = labels == label
        off = ~on
        feasible = np.all(scores[on] >= min_on, axis=0) & np.all(scores[off] <= max_off, axis=0)
        choices = np.flatnonzero(feasible)
        if not len(choices):
            return None
        on_min = scores[on][:, choices].min(axis=0)
        off_max = scores[off][:, choices].max(axis=0)
        choice = int(choices[int(np.argmax(on_min - off_max))])
        fitted[label, :, 0, 0] = weights[choice].astype(np.uint8)
        details[str(label)] = {
            "weights": weights[choice].tolist(),
            "minimum_on": int(scores[on, choice].min()),
            "maximum_off": int(scores[off, choice].max()),
        }
    return fitted, details


def tie_arbitrary(
    original: onnx.ModelProto,
    canonical: str,
    removed: str,
    value: np.ndarray,
) -> onnx.ModelProto:
    model = copy.deepcopy(original)
    initializers = base.initializer_map(model)
    initializers[canonical].CopyFrom(numpy_helper.from_array(value, canonical))
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == removed:
                node.input[index] = canonical
    kept = [item for item in model.graph.initializer if item.name != removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def patch_decoder(model: onnx.ModelProto, weights: np.ndarray) -> None:
    _, weight_name, _, _ = base.qlinear_inputs(model)
    base.initializer_map(model)[weight_name].CopyFrom(numpy_helper.from_array(weights, weight_name))


def semantic_matches_ort(model, cases, records, values):
    expected = points_from_values(records, values)
    actual = base.feature_dataset(model, cases)
    if expected is None:
        return False
    left = sorted((int(x), int(y), int(label)) for (x, y), label in zip(*expected))
    right = sorted((int(x), int(y), int(label)) for (x, y), label in zip(*actual))
    return left == right


def main() -> None:
    cases = base.domain_cases()
    records = symbolic_records(cases)
    original = onnx.load(base.BASELINE)
    scale = float(base.scalar_value(original, "q_scale"))
    baseline_semantic_ok = semantic_matches_ort(original, cases, records, CURRENT)
    searches: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []

    scalar_specs = [
        ("C_F", "C_M", "CF", "CM"),
        ("C_F", "C_L", "CF", "CL"),
        ("C_M", "C_L", "CM", "CL"),
        ("x_zp", "R_B", None, "RB"),
    ]
    for canonical, removed, first_token, second_token in scalar_specs:
        conflict_free = separable = 0
        survivors = []
        for value in range(256):
            values = dict(CURRENT)
            if first_token is not None:
                values[first_token] = value
            values[second_token] = value
            dataset = points_from_values(records, values)
            if dataset is None:
                continue
            conflict_free += 1
            x_zp = value if canonical == "x_zp" else CURRENT.get("XZP", 5)
            w_zp = value if canonical == "x_zp" else CURRENT["RB"]
            fitted = fit_weights(*dataset, x_zp=x_zp, w_zp=w_zp, scale=scale)
            if fitted is None:
                continue
            separable += 1
            survivors.append((value, values, fitted))
        searches.append(
            {
                "kind": "scalar",
                "canonical": canonical,
                "removed": removed,
                "enumerated": 256,
                "conflict_free": conflict_free,
                "decoder_separable": separable,
            }
        )
        for value, values, (weights, details) in survivors:
            dtype = numpy_helper.to_array(base.initializer_map(original)[canonical]).dtype
            shape = numpy_helper.to_array(base.initializer_map(original)[canonical]).shape
            array = np.full(shape, value, dtype=dtype)
            model = tie_arbitrary(original, canonical, removed, array)
            patch_decoder(model, weights)
            if not semantic_matches_ort(model, cases, records, values):
                continue
            disabled = base.exhaustive_runtime(model, cases, True)
            default = base.exhaustive_runtime(model, cases, False)
            path = HERE / f"task226_arbitrary_{canonical}_{removed}_{value}.onnx"
            onnx.save(model, path)
            memory, params, cost = base.cost_of(str(path))
            row = {
                "kind": "scalar",
                "canonical": canonical,
                "removed": removed,
                "value": value,
                "cost": int(cost),
                "memory": int(memory),
                "params": int(params),
                "disabled": disabled,
                "default": default,
                "decoder": details,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if cost < 399 and disabled["right"] == 136 and default["right"] == 136:
                row["path"] = str(path.relative_to(ROOT))
                candidates.append(row)
            else:
                path.unlink(missing_ok=True)

    vector_specs = [
        ("R_F", "R_M", "RF", "RM"),
        ("R_F", "R_L", "RF", "RL"),
        ("R_M", "R_L", "RM", "RL"),
    ]
    for canonical, removed, first_token, second_token in vector_specs:
        conflict_free = separable = 0
        survivors = []
        for first in range(256):
            for second in range(256):
                values = dict(CURRENT)
                values[first_token + "0"] = values[second_token + "0"] = first
                values[first_token + "1"] = values[second_token + "1"] = second
                dataset = points_from_values(records, values)
                if dataset is None:
                    continue
                conflict_free += 1
                fitted = fit_weights(*dataset, x_zp=5, w_zp=16, scale=scale)
                if fitted is None:
                    continue
                separable += 1
                survivors.append((first, second, values, fitted))
        searches.append(
            {
                "kind": "vector",
                "canonical": canonical,
                "removed": removed,
                "enumerated": 65536,
                "conflict_free": conflict_free,
                "decoder_separable": separable,
            }
        )
        for first, second, values, (weights, details) in survivors:
            array = np.asarray([[[[first]], [[second]]]], dtype=np.uint8)
            model = tie_arbitrary(original, canonical, removed, array)
            patch_decoder(model, weights)
            if not semantic_matches_ort(model, cases, records, values):
                continue
            disabled = base.exhaustive_runtime(model, cases, True)
            default = base.exhaustive_runtime(model, cases, False)
            path = HERE / f"task226_arbitrary_{canonical}_{removed}_{first}_{second}.onnx"
            onnx.save(model, path)
            memory, params, cost = base.cost_of(str(path))
            row = {
                "kind": "vector",
                "canonical": canonical,
                "removed": removed,
                "value": [first, second],
                "cost": int(cost),
                "memory": int(memory),
                "params": int(params),
                "disabled": disabled,
                "default": default,
                "decoder": details,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if cost < 399 and disabled["right"] == 136 and default["right"] == 136:
                row["path"] = str(path.relative_to(ROOT))
                candidates.append(row)
            else:
                path.unlink(missing_ok=True)

    payload = {
        "task": 226,
        "baseline_sha256": base.sha256(base.BASELINE),
        "finite_domain_cases": len(cases),
        "symbolic_records": len(records),
        "baseline_semantic_matches_ort": baseline_semantic_ok,
        "searches": searches,
        "candidates": candidates,
    }
    (HERE / "task226_arbitrary_ties.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
