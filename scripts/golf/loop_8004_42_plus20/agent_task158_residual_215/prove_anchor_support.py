#!/usr/bin/env python3
"""Exhaustively prove the reachable local anchor-score support for task158."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task158.onnx"


def original_classes(value: int) -> tuple[bool, bool]:
    """Return (valid, high) from the exact incumbent threshold chain."""
    if value >= 62:
        role_threshold = 144
    elif value >= 22:
        role_threshold = 48
    elif value >= 6:
        role_threshold = 16
    else:
        role_threshold = 4
    return value >= 2, value >= role_threshold


def main() -> None:
    model = onnx.load(AUTHORITY)
    stencil = numpy_helper.to_array(
        next(x for x in model.graph.initializer if x.name == "anchor_stencil")
    )[0, 0].astype(np.int64)
    assert stencil.tolist() == [
        [2, 8, -10],
        [24, 72, -108],
        [-26, -108, 0],
    ]

    positive: set[int] = set()
    configurations: list[dict[str, object]] = []
    # Translation modulo the Conv stride (2) is exhaustive.  The two endpoint
    # blocks occupy one of the two opposite diagonals, and either endpoint may
    # carry code 1 (the numerically lower endpoint colour); the other has code 2.
    for magnitude in (1, 2, 3):
        for diagonal in (0, 1):
            corners = (
                ((0, 0), (2, 2))
                if diagonal == 0
                else ((0, 2), (2, 0))
            )
            for low_endpoint in (0, 1):
                for row_residue in (0, 1):
                    for col_residue in (0, 1):
                        carrier = np.zeros((20, 20), dtype=np.int64)
                        for endpoint, (row, col) in enumerate(corners):
                            code = 1 if endpoint == low_endpoint else 2
                            row0 = row_residue + row * magnitude
                            col0 = col_residue + col * magnitude
                            carrier[
                                row0 : row0 + magnitude,
                                col0 : col0 + magnitude,
                            ] = code
                        local_positive: set[int] = set()
                        for out_row in range(9):
                            for out_col in range(9):
                                patch = carrier[
                                    2 * out_row : 2 * out_row + 3,
                                    2 * out_col : 2 * out_col + 3,
                                ]
                                score = int(np.sum(patch * stencil))
                                if score > 0:
                                    local_positive.add(score)
                                    positive.add(score)
                        configurations.append(
                            {
                                "magnitude": magnitude,
                                "diagonal": diagonal,
                                "low_endpoint": low_endpoint,
                                "row_residue_mod_2": row_residue,
                                "col_residue_mod_2": col_residue,
                                "positive_scores": sorted(local_positive),
                            }
                        )

    expected_positive = {
        2, 4, 8, 10, 16, 20, 24, 26, 48, 52, 72, 106, 144, 212
    }
    assert positive == expected_positive
    rows = []
    for value in [0, *sorted(positive)]:
        valid, high = original_classes(value)
        incumbent_low = valid != high
        bit_low = (value & 0b1010) != 0
        incumbent_phase_ge_0 = value >= 6
        uint8_phase_ge_0 = value > 4
        assert incumbent_low == bit_low
        assert high == (valid != bit_low)
        assert incumbent_phase_ge_0 == uint8_phase_ge_0
        rows.append(
            {
                "score": value,
                "valid": valid,
                "incumbent_high": high,
                "incumbent_low": incumbent_low,
                "score_and_0b1010": value & 0b1010,
                "bit_low": bit_low,
                "bit_high_via_valid_xor_low": valid != bit_low,
                "incumbent_phase_ge_0_score_ge_6": incumbent_phase_ge_0,
                "uint8_phase_ge_0_score_gt_4": uint8_phase_ge_0,
            }
        )

    result = {
        "task": 158,
        "task_hash": "6aa20dc0",
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "exhaustive_local_configurations": len(configurations),
        "dimensions": {
            "magnifications": [1, 2, 3],
            "opposite_diagonals": 2,
            "which_endpoint_is_low_code_1": 2,
            "row_translation_mod_stride_2": 2,
            "col_translation_mod_stride_2": 2,
        },
        "separation_proof": (
            "generator boxes have spacing=2, so nearest cells of distinct boxes "
            "are at coordinate distance >=3 on a separating axis; a 3-cell Conv "
            "window cannot see endpoints from two boxes"
        ),
        "fill_independence": (
            "anchor_weight is nonzero only for the two endpoint-colour channels; "
            "background and sprite-fill cells contribute exactly zero"
        ),
        "boundary_proof": (
            "boundary windows are a subset of the enumerated residue-class local "
            "windows and therefore cannot add a score"
        ),
        "topk_zero_proof": (
            "each endpoint block intersects at most 3x3 sampled Conv windows; "
            "two endpoints x four objects affect at most 72 of 169 windows, so "
            "at least 97 exact-zero windows exist and TopK cannot select negatives"
        ),
        "positive_score_support": sorted(positive),
        "topk_score_support": [0, *sorted(positive)],
        "classification_table": rows,
        "all_classifications_equal": True,
        "all_phase0_predicates_equal": True,
        "configurations": configurations,
    }
    output = HERE / "evidence/anchor_support_proof.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "configurations": len(configurations),
                "positive_score_support": sorted(positive),
                "all_classifications_equal": True,
                "all_phase0_predicates_equal": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
