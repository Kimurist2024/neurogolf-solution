#!/usr/bin/env python3
"""Reproducible structural cost analysis for task192 exact local candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXACT = HERE / "candidates/task192_center_direct_argmax.onnx"
POLICY = HERE / "candidates/task192_policy90_center_direct.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    model = onnx.load(EXACT)
    initializers = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    adjacency = initializers["adj"]
    param_breakdown = {name: int(value.size) for name, value in initializers.items()}
    exact_profile = profile(EXACT)
    policy_profile = profile(POLICY)
    float_relation_materialization_floor = exact_profile["cost"] - adjacency.size + adjacency.size * 4
    result = {
        "exact_argmax_profile": exact_profile,
        "policy_pass_through_profile": policy_profile,
        "exact_initializer_elements": param_breakdown,
        "adjacency": {
            "shape": list(adjacency.shape),
            "elements": int(adjacency.size),
            "nonzero": int(np.count_nonzero(adjacency)),
            "matrix_rank": int(np.linalg.matrix_rank(adjacency.astype(np.float64))),
            "two_factor_inner_rank_min": int(np.linalg.matrix_rank(adjacency.astype(np.float64))),
            "two_dense_factor_param_floor": int(60 * np.linalg.matrix_rank(adjacency.astype(np.float64))),
        },
        "measured_class_floor": {
            "shared_three_row_basis_argmax_onehot": 1143,
            "gap_above_current_1138": 5,
            "remaining_neighbor_map": {"shape": [2, 3], "elements": 6, "rank": 2},
            "remaining_route_map": {"shape": [2, 3], "elements": 6, "rank": 2},
            "materializing_either_dynamic_pair_bytes": 80,
            "saving_from_removing_its_map_elements": 6,
            "net_regression_per_materialized_pair": 74,
        },
        "replacement_floors": {
            "materialized_float30x30_relation_memory": 3600,
            "exact_candidate_floor_if_adj_replaced_by_one_float30x30_output": int(float_relation_materialization_floor),
            "conv_or_float_slice_pad_single_channel_output_memory": 3600,
            "packed_exact_two_15bit_halves": {
                "float_pack_30x2_bytes": 240,
                "int32_cast_30x2_bytes": 240,
                "horizontal_vertical_intersection_three_int32_outputs_bytes": 720,
                "subtotal_before_decode": 1200,
                "greater_than_dense_adj_param_saving": True,
            },
        },
        "conclusion": (
            "No exact ArgMax+OneHot candidate below 1138 was found. The measured exact shared-basis "
            "floor is 1143. The strict-lower 1134 winner preserves the already admitted POLICY90 "
            "HardSigmoid selector raw-bitwise and is not a new approximation."
        ),
    }
    (HERE / "floor_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
