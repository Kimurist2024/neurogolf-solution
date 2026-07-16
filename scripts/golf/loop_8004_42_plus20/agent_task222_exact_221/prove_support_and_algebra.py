#!/usr/bin/env python3
"""Document task222 generator non-identifiability and exact factor ranks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL_PATH = HERE / "authority" / "task222.onnx"
GENERATOR_PATH = ROOT / "inputs/arc-gen-repo/tasks/task_91714a58.py"
COMMON_DIR = GENERATOR_PATH.parent.parent


def exact_fraction(value: np.generic) -> Fraction:
    # The coefficients are serialized float32. Converting the unpacked Python
    # float gives the exact binary rational represented by those 32 bits.
    packed = struct.pack("<f", float(value))
    unpacked = struct.unpack("<f", packed)[0]
    return Fraction.from_float(unpacked)


def exact_rank(rows: Iterable[Iterable[np.generic]]) -> int:
    matrix = [[exact_fraction(value) for value in row] for row in rows]
    if not matrix:
        return 0
    height, width = len(matrix), len(matrix[0])
    rank = 0
    for col in range(width):
        pivot = next((r for r in range(rank, height) if matrix[r][col]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][col]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for r in range(height):
            if r == rank or not matrix[r][col]:
                continue
            factor = matrix[r][col]
            matrix[r] = [
                left - factor * right
                for left, right in zip(matrix[r], matrix[rank], strict=True)
            ]
        rank += 1
        if rank == height:
            break
    return rank


def load_generator():
    sys.path.insert(0, str(COMMON_DIR))
    spec = importlib.util.spec_from_file_location("task222_true_generator", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ambiguity_certificate() -> dict:
    module = load_generator()
    grid = [[0 for _ in range(16)] for _ in range(16)]
    for r in range(2, 5):
        for c in range(2, 5):
            grid[r][c] = 2
    for r in range(10, 13):
        for c in range(10, 13):
            grid[r][c] = 3
    colors = [color for row in grid for color in row]
    planted_a = module.generate(
        width=3, height=3, row=2, col=2, colors=colors, size=16
    )
    planted_b = module.generate(
        width=3, height=3, row=10, col=10, colors=colors, size=16
    )
    assert planted_a["input"] == planted_b["input"] == grid
    output_diff = sum(
        planted_a["output"][r][c] != planted_b["output"][r][c]
        for r in range(16)
        for c in range(16)
    )
    assert output_diff == 18
    # Under no-argument generation, either exact pre-overlay bitmap has
    # positive probability: select exactly the other nine rectangle cells,
    # choose its color for all nine, and select no other cells. Width/height,
    # position, and target color also each have positive discrete probability.
    conditional_bitmap_probability = (0.5**256) * ((1.0 / 9.0) ** 9)
    return {
        "same_input": True,
        "different_outputs": True,
        "different_output_cells": output_diff,
        "parameterization_a": {
            "width": 3,
            "height": 3,
            "row": 2,
            "col": 2,
            "color": 2,
        },
        "parameterization_b": {
            "width": 3,
            "height": 3,
            "row": 10,
            "col": 10,
            "color": 3,
        },
        "input_grid": grid,
        "output_a": planted_a["output"],
        "output_b": planted_b["output"],
        "random_support_argument": {
            "random_pixels_can_select_exact_other_rectangle": True,
            "all_nine_random_colors_can_equal_other_color": True,
            "target_color_has_no_pre_overlay_adjacent_pair": True,
            "all_four_post_overlay_boundary_checks_pass": True,
            "conditional_bitmap_probability_positive": conditional_bitmap_probability > 0.0,
            "conditional_bitmap_probability_log": 256 * math.log(0.5)
            + 9 * math.log(1.0 / 9.0),
            "conclusion": (
                "Both identical-input/different-output parameterizations have "
                "positive probability in the no-argument generator support."
            ),
        },
        "logical_conclusion": (
            "No deterministic ONNX model can be correct on every element of the "
            "full generator parameter support; only an all-input pass-through "
            "rewrite of an admitted authority can close a zero-added-risk proof."
        ),
    }


def known_rule_certificate() -> dict:
    payload = json.loads(
        (ROOT / "inputs/neurogolf-2026/task222.json").read_text()
    )
    split_counts: dict[str, int] = {}
    failures = []
    for split in ("train", "test", "arc-gen"):
        rows = payload.get(split, [])
        split_counts[split] = len(rows)
        for index, example in enumerate(rows):
            source = example["input"]
            target = example["output"]
            nonzero = [
                (r, c, target[r][c])
                for r in range(len(target))
                for c in range(len(target[r]))
                if target[r][c] != 0
            ]
            reasons = []
            colors = {color for _, _, color in nonzero}
            if len(colors) != 1:
                reasons.append("output_nonzero_color_not_unique")
            if not nonzero:
                reasons.append("empty_output")
            else:
                rows_used = [r for r, _, _ in nonzero]
                cols_used = [c for _, c, _ in nonzero]
                r0, r1 = min(rows_used), max(rows_used)
                c0, c1 = min(cols_used), max(cols_used)
                height, width = r1 - r0 + 1, c1 - c0 + 1
                if len(nonzero) != height * width:
                    reasons.append("output_not_filled_rectangle")
                if not (2 <= width <= 8 and 2 <= height <= 8):
                    reasons.append("dimensions_outside_generator_range")
                if not (9 <= width * height <= 16):
                    reasons.append("area_outside_generator_range")
                if not (
                    r0 >= 1
                    and c0 >= 1
                    and r1 <= len(target) - 2
                    and c1 <= len(target[0]) - 2
                ):
                    reasons.append("rectangle_not_in_one_cell_interior")
                if any(source[r][c] != color for r, c, color in nonzero):
                    reasons.append("output_rectangle_does_not_copy_input")
                if any(
                    target[r][c] != 0
                    for r in range(len(target))
                    for c in range(len(target[r]))
                    if not (r0 <= r <= r1 and c0 <= c <= c1)
                ):
                    reasons.append("nonzero_outside_rectangle")
            if reasons:
                failures.append({"split": split, "index": index, "reasons": reasons})
    return {
        "generator_source": str(GENERATOR_PATH.relative_to(ROOT)),
        "generator_source_sha256": hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest(),
        "dataset_source": "inputs/neurogolf-2026/task222.json",
        "split_counts": split_counts,
        "total": sum(split_counts.values()),
        "failures": failures,
        "passed": not failures,
        "decoded_rule": (
            "Return only the latent planted filled monochrome rectangle. Its width "
            "and height are in 2..8, area in 9..16, and it is placed at least one "
            "cell inside the fixed 16x16 border; all other output cells are zero."
        ),
        "classification": "global latent-object selection / non-identifiable Type D",
    }


def algebra_certificate() -> dict:
    model = onnx.load(MODEL_PATH)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    v, u, a, s, p = (arrays[name] for name in ("V", "U", "A", "S", "P"))
    assert np.count_nonzero(v[16:]) == 0
    allowed_pairs = [
        [width, height]
        for width in range(2, 9)
        for height in range(2, 9)
        if 9 <= width * height <= 16
    ]
    dense_params = {name: int(array.size) for name, array in arrays.items()}
    report = {
        "authority_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "equation": next(
            attr.s.decode("ascii")
            for attr in model.graph.node[0].attribute
            if attr.name == "equation"
        ),
        "initializer_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "initializer_dense_params": dense_params,
        "total_dense_params": sum(dense_params.values()),
        "initializer_nonzeros": {
            name: int(np.count_nonzero(array)) for name, array in arrays.items()
        },
        "exact_ranks": {
            "V_full_30x8": exact_rank(v),
            "V_active_rows_0_15": exact_rank(v[:16]),
            "U": exact_rank(u),
            "A": exact_rank(a),
            "S": exact_rank(s),
            "P": exact_rank(p),
        },
        "all_U_entries_nonzero": bool(np.all(u != 0)),
        "all_A_entries_nonzero": bool(np.all(a != 0)),
        "V_zero_tail": {
            "first_zero_row": 16,
            "zero_rows": 14,
            "zero_elements": int(np.count_nonzero(v[16:] == 0)),
            "dense_Einsum_contract": (
                "D/E/F/H are tied to the static input spatial dimension 30, so "
                "a dense V with shape [16,8] is not a legal replacement."
            ),
            "standard_dense_pad_cost": {
                "V16_params": 128,
                "Pad_output_memory_float32_30x8": 960,
                "combined_before_other_initializers": 1088,
                "authority_V_params": 240,
            },
        },
        "rank_conclusion": (
            "V is exact full column rank 8 on the only active rows and every U "
            "coefficient is nonzero. Any rank-7 V/U deletion changes the real "
            "polynomial; the runtime audit supplies explicit counterexamples."
        ),
        "P_projection": {
            "formula": "out0=10*(s0-sum(s1..s9)); out_o=10*s_o for o>0",
            "rank": exact_rank(p),
            "conclusion": (
                "P is full rank 10. Direct deletion is not a pass-through; the "
                "known audit disproves the required background-sign equivalence "
                "on every one of 266 cases."
            ),
        },
        "generator": {
            "hash": "91714a58",
            "size": 16,
            "allowed_dimension_pairs": allowed_pairs,
            "allowed_pair_count": len(allowed_pairs),
            "row_col_interior": True,
        },
        "examined_dense_floor": {
            "V": 240,
            "U": 16,
            "A": 4,
            "S": 20,
            "P": 100,
            "sum": 380,
            "scope": (
                "current one-node dense-Einsum factorization under exact real "
                "pass-through, excluding malformed/sparse/extra-activation tricks"
            ),
        },
    }
    assert report["total_dense_params"] == 380
    assert report["exact_ranks"]["V_active_rows_0_15"] == 8
    assert report["exact_ranks"]["P"] == 10
    return report


def main() -> None:
    evidence = HERE / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "generator_support_ambiguity.json").write_text(
        json.dumps(ambiguity_certificate(), indent=2) + "\n"
    )
    (evidence / "algebra_certificate.json").write_text(
        json.dumps(algebra_certificate(), indent=2) + "\n"
    )
    (evidence / "true_rule_known_certificate.json").write_text(
        json.dumps(known_rule_certificate(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
