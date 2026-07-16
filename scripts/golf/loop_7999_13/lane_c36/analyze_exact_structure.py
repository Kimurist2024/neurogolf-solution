#!/usr/bin/env python3
"""Record exact factor/sharing obstructions for the task012 incumbent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
MODEL = HERE / "baseline" / "task012.onnx"


def main() -> int:
    model = onnx.load(MODEL)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    weights = arrays["w"]
    bias = arrays["b"]
    spatial = weights[:, 0]
    exact_fg_duplicate = all(np.array_equal(spatial[1], spatial[i]) for i in range(2, 10))
    report = {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "graph": {
            "nodes": len(model.graph.node),
            "ops": [node.op_type for node in model.graph.node],
            "weight_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "parameter_elements": int(sum(a.size for a in arrays.values())),
        },
        "kernels": [
            {
                "channel": i,
                "matrix_rank": int(np.linalg.matrix_rank(spatial[i].astype(np.float64))),
                "zero_elements": int(np.count_nonzero(spatial[i] == 0)),
                "zero_top_border": bool(np.all(spatial[i, 0] == 0)),
                "zero_bottom_border": bool(np.all(spatial[i, -1] == 0)),
                "zero_left_border": bool(np.all(spatial[i, :, 0] == 0)),
                "zero_right_border": bool(np.all(spatial[i, :, -1] == 0)),
            }
            for i in range(10)
        ],
        "sharing": {
            "foreground_kernels_byte_identical": exact_fg_duplicate,
            "foreground_biases_byte_identical": bool(np.all(bias[1:] == bias[1])),
            "background_kernel_scalar_multiple_of_foreground": bool(
                np.linalg.matrix_rank(
                    np.stack([spatial[0].reshape(-1), spatial[1].reshape(-1)]).astype(np.float64)
                )
                == 1
            ),
            "onnx_grouped_conv_contract": (
                "group=10 requires serialized W shape [10,1,kH,kW]; equal values do not share initializer elements"
            ),
            "batched_shared_kernel_blocker": (
                "separate background and foreground convolutions expose at least 900+8100 output elements "
                "before canonical one-output assembly, already above incumbent cost 710"
            ),
        },
        "factorization": {
            "background_spatial_rank": int(np.linalg.matrix_rank(spatial[0].astype(np.float64))),
            "foreground_spatial_rank": int(np.linalg.matrix_rank(spatial[1].astype(np.float64))),
            "exact_rank_one_separable": False,
            "rank7_two_stage_minimum_intermediate_elements": 10 * 7 * 30 * 30,
            "blocker": "any explicit two-stage rank factorization creates an intermediate far above cost 710",
        },
        "grouped_active_column_budgets_below_710": [
            {
                "groups": groups,
                "input_columns_per_output": 10 // groups,
                "maximum_kernel_area_with_bias": (709 - 10) // (100 // groups),
            }
            for groups in (1, 2, 5, 10)
        ],
        "grouped_active_column_lower_bound": (
            "group=1 permits area only 6, already below the radius-2 spatial lower bound 5x5. "
            "For groups=2/5, every foreground output has legal cases where all other active colours lie "
            "outside its input-channel group, reducing its classifier to the depthwise case. Those groups "
            "permit areas only 13/34 below cost 710; the radius bound rejects <=13 and the exhaustive "
            "depthwise sweep rejects every geometry through area34."
        ),
        "gauge": {
            "bias_free_area70_feasible": False,
            "evidence": [
                "homogeneous_search_area70_strict.json",
                "homogeneous_search_weak.json",
            ],
        },
    }
    (HERE / "exact_structure_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
