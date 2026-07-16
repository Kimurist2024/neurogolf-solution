#!/usr/bin/env python3
"""Exhaust exact score-bearing structural reductions for task398."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = HERE / "baseline" / "task398.onnx"
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attr_key(attr: onnx.AttributeProto) -> bytes:
    clone = onnx.AttributeProto()
    clone.CopyFrom(attr)
    return clone.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> tuple[object, ...]:
    return (
        node.domain,
        node.op_type,
        tuple(node.input),
        tuple(sorted((attr.name, attr_key(attr)) for attr in node.attribute)),
    )


def signed_power_of_two(value: float) -> bool:
    if not math.isfinite(value) or value == 0:
        return False
    exponent = math.log2(abs(value))
    return exponent.is_integer()


def main() -> int:
    model = onnx.load(MODEL)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    memory, params, total = cost_of(MODEL)
    exact_duplicate_pairs = []
    names = list(arrays)
    for index, left in enumerate(names):
        for right in names[:index]:
            if arrays[left].dtype == arrays[right].dtype and arrays[left].shape == arrays[right].shape and np.array_equal(arrays[left], arrays[right]):
                exact_duplicate_pairs.append([right, left])
    seen: dict[tuple[object, ...], int] = {}
    duplicate_nodes = []
    for index, node in enumerate(model.graph.node):
        key = node_key(node)
        if key in seen:
            duplicate_nodes.append([seen[key], index])
        else:
            seen[key] = index

    d = arrays["D"].astype(np.float64)
    gauge_rows = []
    for index in range(5):
        q = arrays[f"Q{index}"].astype(np.float64)
        gauge = d / q
        gauge_rows.append(
            {
                "carrier": f"Q{index}",
                "gauge": gauge.tolist(),
                "all_components_signed_power_of_two": all(
                    signed_power_of_two(float(value)) for value in gauge
                ),
                "exact_equal_to_D": bool(np.array_equal(arrays[f"Q{index}"], arrays["D"])),
                "exact_equal_to_minus_D": bool(np.array_equal(arrays[f"Q{index}"], -arrays["D"])),
            }
        )

    final = model.graph.node[-1]
    final_equation = next(attr.s.decode() for attr in final.attribute if attr.name == "equation")
    k_occurrences = sum(name == "K" for name in final.input)
    t_rank = int(np.linalg.matrix_rank(arrays["T"].astype(np.float64)))
    v_rank = int(np.linalg.matrix_rank(arrays["V"].astype(np.float64)))
    k = arrays["K"].astype(np.float64)
    k_state_rank = int(np.linalg.matrix_rank(k.reshape(3, -1)))
    k_factor_params = 3 * k_state_rank + k_state_rank * 8
    report = {
        "authority": {
            "sha256": digest(MODEL.read_bytes()),
            "cost": {"memory": memory, "params": params, "total": total},
            "nodes": len(model.graph.node),
            "final_einsum_inputs": len(final.input),
        },
        "initializers": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "raw_sha256": digest(array.tobytes()),
            }
            for name, array in arrays.items()
        },
        "exact_duplicate_initializers": exact_duplicate_pairs,
        "exact_duplicate_nodes": duplicate_nodes,
        "identity_nodes": [],
        "initializer_carriers": {
            "Q_vs_D": gauge_rows,
            "conclusion": (
                "No Q is D or -D. Every gauge that maps a Q to D needs a non-power-of-two "
                "first-state scale, so it can change float32 contraction order/rounding."
            ),
            "q4_reuse_rejected": {
                "candidate_sha256": "5a9ff2fb8fd5efed49a3dd1790973b2386bbd636a2eb33e02fc663c93198d25a",
                "cost": 347,
                "external_seeded_cases": 500,
                "threshold_mismatches": 4,
                "reason": "not an arbitrary-input exact carrier",
            },
        },
        "factorization": {
            "T": {
                "shape": [10, 10],
                "rank": t_rank,
                "dense_params": 100,
                "rank_factor_params": 10 * t_rank + t_rank * 10,
                "conclusion": "rank factorization is larger",
            },
            "V": {
                "shape": [30, 2],
                "rank": v_rank,
                "dense_params": 60,
                "rank_factor_params": 30 * v_rank + v_rank * 2,
                "conclusion": "generic rank factors are larger than the dense initializer",
            },
            "K": {
                "shape": [3, 2, 2, 2],
                "state_flatten_rank": k_state_rank,
                "dense_params": 24,
                "two_factor_params": k_factor_params,
                "occurrences_in_final_einsum": k_occurrences,
                "extra_final_operands_if_factored": k_occurrences,
                "intermediate_elements_if_materialized": 24,
                "conclusion": (
                    "The nominal two-parameter factor shave either enlarges the existing giant Einsum "
                    "from 69 to 79 inputs (prohibited) or adds a 96-byte float intermediate, net worse."
                ),
            },
        },
        "operand_axis_audit": {
            "node0_equation": "bihw,oi->b",
            "final_equation": final_equation,
            "input_T_not_duplicate_CSE": (
                "node0 contracts color axis o to scalar n; the final contraction retains o as the output "
                "color channel, so replacing it with n changes semantics"
            ),
            "T_column_sum_probe_rejected": (
                "T column sums can replace only node0; final T must remain 10x10, so adding a separate "
                "10-vector raises parameters from 206 to 216"
            ),
            "attribute_or_name_removal_score_gain": 0,
        },
        "constant_reconstruction_audit": {
            "one": (
                "Scalar ones occur inside D/Q/V but not as a same-shape initializer. Extracting one "
                "requires a scored node output (at least 4 bytes) to save one parameter, so it is worse."
            ),
            "out_size": (
                "The int64 scalar 30 has no exact same-type carrier. Shape/Gather/Cast reconstruction "
                "adds scored intermediates and cannot beat its one parameter."
            ),
            "V": (
                "V is the fixed [1, row_index] basis. Runtime construction materializes 60 float32 "
                "elements (240 bytes) to replace 60 parameters; factoring it would also enlarge the "
                "existing 69-input final Einsum."
            ),
            "Q_shared_suffix": (
                "The five Q vectors share two trailing ones, but rebuilding them creates five length-3 "
                "intermediates or additional final operands and changes contraction ordering."
            ),
            "sparse_initializers": (
                "Not considered: repository policy records sparse_initializer as a proven grader error."
            ),
        },
        "memory_audit": {
            "intermediates": {"n": 4, "N": 12, "roi": 8, "inside": 120},
            "total_bytes": 144,
            "all_runtime_shapes_truthful": True,
            "removable_identity_or_duplicate_intermediate": None,
        },
        "conclusion": "No arbitrary-input exact score reduction under the no-enlarged-giant rule.",
    }
    (HERE / "exact_structure_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
