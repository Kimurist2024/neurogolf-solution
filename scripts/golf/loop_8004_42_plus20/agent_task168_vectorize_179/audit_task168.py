#!/usr/bin/env python3
"""Reproduce the task168 exact-vectorization pre-gate.

This lane is deliberately read-only with respect to the submission and score
ledger.  It inventories the 8009.46 authority and proves why the scoped
vectorization/factorization rewrites cannot produce a strict-lower candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline" / "task168.onnx"
AUDIT = HERE / "audit"

EXPECTED = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
    "baseline/task168.onnx": "642cba5c350b35774bf98e427ca858a675cd8dd483ea6d1b2ec7e13287739b92",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value.type.tensor_type
    return [int(dim.dim_value) for dim in tensor_type.shape.dim]


def tensor_bytes(value: onnx.ValueInfoProto) -> int:
    dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
    return math.prod(tensor_shape(value)) * np.dtype(dtype).itemsize


def rank_by_mode(array: np.ndarray) -> list[int]:
    return [
        int(np.linalg.matrix_rank(np.moveaxis(array, mode, 0).reshape(array.shape[mode], -1)))
        for mode in range(array.ndim)
    ]


def initializer_inventory(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        rows.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
            }
        )
    return rows


def node_memory_inventory(inferred: onnx.ModelProto) -> list[dict[str, object]]:
    values = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    rows = []
    for index, node in enumerate(inferred.graph.node):
        for output in node.output:
            if not output or output == "output":
                continue
            item = values[output]
            rows.append(
                {
                    "node_index": index,
                    "op_type": node.op_type,
                    "output": output,
                    "shape": tensor_shape(item),
                    "dtype": onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type),
                    "bytes": tensor_bytes(item),
                }
            )
    return rows


def conv_bias_ub(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    failures = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "ConvTranspose"} or len(node.input) < 3:
            continue
        weight = initializers.get(node.input[1])
        bias = initializers.get(node.input[2])
        if weight is None or bias is None:
            failures.append({"node_index": index, "reason": "dynamic_weight_or_bias"})
            continue
        expected = int(weight.shape[0] if node.op_type == "Conv" else weight.shape[1])
        if bias.ndim != 1 or bias.shape[0] != expected:
            failures.append(
                {
                    "node_index": index,
                    "reason": "bias_length_mismatch",
                    "expected": expected,
                    "actual": list(bias.shape),
                }
            )
    return failures


def main() -> None:
    # Root checkpoint guard: this script never rewrites these paths.
    guards = {
        "submission.zip": sha256(ROOT / "submission.zip"),
        "submission_base_8009.46.zip": sha256(ROOT / "submission_base_8009.46.zip"),
        "all_scores.csv": sha256(ROOT / "all_scores.csv"),
        "baseline/task168.onnx": sha256(BASELINE),
    }
    assert guards == EXPECTED, (guards, EXPECTED)
    with zipfile.ZipFile(ROOT / "submission.zip") as archive:
        member_sha = hashlib.sha256(archive.read("task168.onnx")).hexdigest()
    assert member_sha == EXPECTED["baseline/task168.onnx"]

    model = onnx.load(str(BASELINE))
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)

    sys.path.insert(0, str(ROOT / "scripts"))
    from lib import scoring  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="task168_vectorize_baseline_") as workdir:
        score = scoring.score_and_verify(
            copy.deepcopy(model), 168, workdir, label="authority", require_correct=True
        )
    assert score is not None
    assert (score["memory"], score["params"], score["cost"]) == (237, 178, 415)

    initializers = initializer_inventory(model)
    node_memory = node_memory_inventory(inferred)
    assert sum(int(row["bytes"]) for row in node_memory) == score["memory"]
    assert sum(int(row["elements"]) for row in initializers) == score["params"]

    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    ranks = {
        name: {
            "shape": list(arrays[name].shape),
            "mode_ranks": rank_by_mode(arrays[name]),
        }
        for name in ("Bc", "C1T", "Xrow")
    }
    assert ranks["Bc"]["mode_ranks"] == [2, 2]
    assert ranks["C1T"]["mode_ranks"] == [2, 2, 2]
    assert ranks["Xrow"]["mode_ranks"] == [1, 1, 1, 1]

    # Branch memory after the four seed tensors is 34,33,34,33 bytes.  A
    # batched op preserves the total
    # number and dtype of elements; producing its input from the four existing
    # seed tensors requires 4*7 additional uint8 elements.
    branch_names = {
        "seed0": ["maxv0", "maxv0s", "ridx0", "r8_0", "rt_0", "sf0", "lg0", "l2_0", "b0", "d0i", "d0"],
        "seed1": ["maxv1", "ridx1", "r8_1", "rt_1", "sfbase1", "lg1", "l2_1", "b1", "d1i", "d1"],
        "seed2": ["maxv2", "maxv2s", "ridx2", "r8_2", "rneg2", "sf2", "lg2", "l2_2", "b2", "d2i", "d2"],
        "seed3": ["maxv3", "ridx3", "r8_3", "rneg3", "sfbase3", "lg3", "l2_3", "b3", "d3i", "d3"],
    }
    by_name = {str(row["output"]): int(row["bytes"]) for row in node_memory}
    branch_memory = {
        branch: sum(by_name[name] for name in names) for branch, names in branch_names.items()
    }
    assert branch_memory == {"seed0": 34, "seed1": 33, "seed2": 34, "seed3": 33}

    transformations = [
        {
            "name": "four_branch_vectorize_via_concat",
            "status": "PRE_GATE_REJECT",
            "memory_delta_lower_bound": 28,
            "params_delta_lower_bound": 0,
            "cost_lower_bound": 443,
            "proof": "batched downstream tensors retain 134 bytes; materializing [4,7] uint8 from four seed outputs adds 28 bytes",
        },
        {
            "name": "two_direction_pair_vectorizations",
            "status": "PRE_GATE_REJECT",
            "memory_delta_lower_bound": 28,
            "params_delta_lower_bound": 0,
            "cost_lower_bound": 443,
            "proof": "pairs (0,2) and (1,3) each need a [2,7] uint8 stack; element counts after ReduceMax/ArgMax/Cast/Log/Selu are conserved",
        },
        {
            "name": "Xrow_rank1_factor",
            "status": "PRE_GATE_REJECT",
            "memory_delta_lower_bound": 320,
            "params_delta": -62,
            "cost_delta_lower_bound": 258,
            "cost_lower_bound": 673,
            "proof": "80-element Conv weight becomes 10+8 parameters, but the reconstructed float32 [1,10,1,8] tensor is a scored 320-byte intermediate",
        },
        {
            "name": "Bc_rank2_factor_inside_Einsum",
            "status": "PRE_GATE_REJECT",
            "memory_delta": 0,
            "params_delta_lower_bound": 4,
            "cost_lower_bound": 419,
            "proof": "rank(Bc)=2; a direct rank-2 factorization needs 30*2+2*2=64 parameters versus the dense 60",
        },
        {
            "name": "C1T_low_rank_factor_inside_Einsum",
            "status": "PRE_GATE_REJECT",
            "memory_delta": 0,
            "params_delta_lower_bound": 2,
            "cost_lower_bound": 417,
            "proof": "even the optimistic CP-rank-2 storage is 3*2+2*2+2*2=14 parameters versus dense 12",
        },
        {
            "name": "absorb_sd_sc_into_duplicate_C1T",
            "status": "PRE_GATE_REJECT",
            "memory_delta": 0,
            "params_delta": 7,
            "cost_lower_bound": 422,
            "proof": "removing sd(3)+sc(2) requires at least one additional 12-element branch-specific C1T while retaining the unscaled C1T",
        },
        {
            "name": "compress_C1T_descriptor_axis_3_to_2",
            "status": "PRE_GATE_REJECT",
            "memory_delta": None,
            "params_delta": None,
            "proof": "unscaled use depends on q0=10*b+11*r, but the sd=[0.9,1,0.9] use depends on q1=10*b+9.9*r; (a,q0,q1) has rank 3 in (a,b,r), so one shared 2-vector cannot be raw-equivalent",
            "runtime_control": "changing d/C1T to length 2 alone leaves the sd operand length 3 and ORT rejects the Einsum dimension",
        },
        {
            "name": "derive_sc_or_sn2_by_slicing_existing_initializers",
            "status": "PRE_GATE_REJECT",
            "memory_delta_lower_bound": 8,
            "params_delta_best": -2,
            "cost_delta_lower_bound": 6,
            "proof": "a length-2 float slice is an 8-byte scored intermediate; the directly stored vector costs only 2 parameters",
        },
        {
            "name": "row_coordinate_rebase",
            "status": "PRE_GATE_REJECT",
            "proof": "Bc is shared for row and column indices; rebasing its second column changes both coordinates and the sn2/sc gauges do not commute with the required shear",
        },
    ]

    ub = conv_bias_ub(model)
    assert ub == []
    examples = scoring.load_examples(168)
    result = {
        "task": 168,
        "decision": "NO_STRICT_LOWER_PRE_GATE",
        "authority": {
            "checkpoint": 8009.46,
            "root_guards": guards,
            "zip_member_sha256": member_sha,
            "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "score": score,
            "known_examples": {key: len(examples[key]) for key in ("train", "test", "arc-gen")},
            "known_total": sum(len(examples[key]) for key in ("train", "test", "arc-gen")),
        },
        "structure": {
            "checker_full": "PASS",
            "shape_inference_strict_data_prop": "PASS",
            "truthful_declared_value_info_count": len(model.graph.value_info),
            "truthful_io_shapes": {
                "input": tensor_shape(model.graph.input[0]),
                "output": tensor_shape(model.graph.output[0]),
            },
            "conv_family_bias_ub_count": len(ub),
        },
        "initializer_inventory": initializers,
        "initializer_ranks": ranks,
        "node_output_memory": node_memory,
        "branch_memory": branch_memory,
        "transformations": transformations,
        "strict_lower_candidates": 0,
        "raw_equivalence_runs": 0,
        "known_candidate_runs": 0,
        "fresh": {
            "required_if_strict_lower": {"seeds": [0, 1, 2, 3], "per_seed": 10000},
            "runs": 0,
            "reason": "no transformation passed the strict-lower algebraic/cost pre-gate",
        },
        "root_mutations": [],
    }

    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"], "score": score, "guards": guards}, indent=2))


if __name__ == "__main__":
    main()
