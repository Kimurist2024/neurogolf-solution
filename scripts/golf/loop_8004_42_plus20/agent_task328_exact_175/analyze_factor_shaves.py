#!/usr/bin/env python3
"""Record exact parameter-count bounds for requested task328 factor shaves."""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "controls/exact554.onnx"


def mode_ranks(array: np.ndarray) -> list[int]:
    return [
        int(np.linalg.matrix_rank(np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)))
        for axis in range(array.ndim)
    ]


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    final = model.graph.node[-1]
    equation = next(
        attr.s.decode("ascii") for attr in final.attribute if attr.name == "equation"
    )
    used = {char for char in equation if char in string.ascii_letters}
    free = [char for char in string.ascii_letters if char not in used]

    core = arrays["CoreB"]
    # Six sparse CP terms: three in each independent 2x2 block.
    a = np.zeros((4, 6), dtype=np.float32)
    b = np.zeros((4, 6), dtype=np.float32)
    a[[0, 1, 1, 2, 3, 3], np.arange(6)] = 1
    b[[0, 0, 1, 2, 2, 3], np.arange(6)] = 1
    c = np.asarray(
        [
            [1, 3, 0, 0, 0],
            [0, -3, 3, 0, 0],
            [1, -3, 0, 0, 0],
            [1, 0, 0, -3, 0],
            [0, 0, 0, 3, -3],
            [1, 0, 0, 3, 0],
        ],
        dtype=np.float32,
    ).T
    rebuilt = np.einsum("ar,br,cr->abc", a, b, c)
    if not np.array_equal(rebuilt, core):
        raise RuntimeError("CoreB CP reconstruction failed")

    core0 = core[0]
    left = np.asarray([1, 0, 0, 0], dtype=np.float32)
    right = np.asarray([1, 3, 0, 0, 0], dtype=np.float32)
    if not np.array_equal(np.outer(left, right), core0):
        raise RuntimeError("CoreB[0] rank-one reconstruction failed")

    ssel = arrays["Ssel"]
    frow = arrays["frow"]
    fused_nonzero = np.einsum("qy,q->y", ssel, frow)
    expected_nonzero = np.asarray([0] + [1] * 9, dtype=np.float32)
    if not np.array_equal(fused_nonzero, expected_nonzero):
        raise RuntimeError("Ssel/frow contraction changed")

    tfeat = arrays["TFeat"]
    tfeat_rebuilt = np.stack(
        (tfeat[0], tfeat[1], tfeat[2], tfeat[1] * tfeat[2])
    )
    result = {
        "exact554_equation_letters_used": len(used),
        "exact554_free_letters": free,
        "exact554_free_letter_count": len(free),
        "core_b": {
            "shape": list(core.shape),
            "params": int(core.size),
            "nonzero_values": int(np.count_nonzero(core)),
            "mode_ranks": mode_ranks(core),
            "six_term_cp_exact": True,
            "six_term_cp_params": int(a.size + b.size + c.size),
            "six_term_cp_saving": int(core.size - (a.size + b.size + c.size)),
            "all_eight_uses_need_private_rank_letters": 8,
            "factorization_blocked_in_exact554_by_letter_budget": len(free) < 8,
            "reason_private_letters": (
                "sharing the CP-rank index between two CoreB occurrences couples "
                "their independent sums and changes the polynomial"
            ),
        },
        "e_selector_absorption": {
            "core_b_zero_slice_rank": int(np.linalg.matrix_rank(core0)),
            "core_b_zero_slice_factor_params": int(left.size + right.size),
            "direct_absorption_total_for_core_group": int(core.size + left.size + right.size),
            "incumbent_core_plus_e_params": int(core.size + 4),
            "direct_absorption_delta": int(left.size + right.size - 4),
            "cp_plus_zero_slice_params": int(a.size + b.size + c.size + left.size + right.size),
            "cp_plus_zero_slice_delta": int(a.size + b.size + c.size + left.size + right.size - core.size - 4),
            "decision": "no parameter shave",
        },
        "frow_ssel": {
            "identity": "sum_q Ssel[q,y]*frow[q] = [0,1,1,1,1,1,1,1,1,1]",
            "current_incremental_params": int(frow.size),
            "materialized_fused_vector_params": int(fused_nonzero.size),
            "delta": int(fused_nonzero.size - frow.size),
            "decision": "fusing costs eight more parameters",
        },
        "duplicate_sign_factor": {
            "tfeat_shape": list(tfeat.shape),
            "row3_equals_row1_times_row2_float32": bool(np.array_equal(tfeat_rebuilt, tfeat)),
            "current_params": int(tfeat.size),
            "two_2x30_factors_plus_4x2x2_selector_params": 136,
            "delta": 16,
            "decision": "exact sign/coordinate split increases parameters",
        },
    }
    (HERE / "factor_shave_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
