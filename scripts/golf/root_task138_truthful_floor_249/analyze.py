#!/usr/bin/env python3
"""Reproduce the task138 r01 truthful-shape floor audit.

This lane is diagnostic and fail-closed.  The archive model uses three
CenterCropPad values whose declared [1,1,1,1] shapes do not describe their
runtime tensors.  We construct the smallest direct normalization considered in
this lane: remove the three hiding nodes, use the graph input directly, widen
the connected float16 arithmetic to float32, and regenerate every value_info
from an ORT_DISABLE_ALL runtime trace.  The normalized artifact is retained
only as a rejected cost witness.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = (
    ROOT
    / "scripts/golf/loop_7999_13/lane_archive_all400"
    / "task138_r01_static2588.onnx"
)
SOURCE_SHA256 = "b3e0740b0c2186502d0f5d67ad372304caf6bb948ba07cf67d7ba936193f25db"
REJECTED = HERE / "rejected_probes/task138_r01_truthful_direct_f32.onnx"
RESULT = HERE / "analysis.json"
TASK = 138
AUTHORITY_COST = 2705


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def type_map(model: onnx.ModelProto) -> dict[str, int]:
    return {
        value.name: int(value.type.tensor_type.elem_type)
        for value in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
        if value.type.HasField("tensor_type")
    }


def declared_shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    return {
        value.name: dims(value)
        for value in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.HasField("shape")
    }


def one_hot_example() -> np.ndarray:
    task = json.loads((ROOT / "inputs/neurogolf-2026/task138.json").read_text())
    grid = task["train"][0]["input"]
    array = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            array[0, int(color), row, col] = 1.0
    return array


def trace_all_shapes(
    model: onnx.ModelProto,
    output_types: dict[str, int],
    example: np.ndarray,
) -> dict[str, list[int]]:
    """Run every intermediate as a graph output without stale value_info."""

    traced = copy.deepcopy(model)
    del traced.graph.value_info[:]
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name in names:
                continue
            if name not in output_types:
                raise RuntimeError(f"missing type for traced output {name}")
            traced.graph.output.append(
                helper.make_tensor_value_info(name, output_types[name], None)
            )
            names.append(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    arrays = session.run(names, {"input": example})
    return {
        name: [int(x) for x in np.asarray(array).shape]
        for name, array in zip(names, arrays)
    }


def replace_all_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def widen_float16(model: onnx.ModelProto) -> None:
    """Widen the single connected float computation to match FLOAT input."""

    for index, initializer in enumerate(list(model.graph.initializer)):
        if initializer.data_type != TensorProto.FLOAT16:
            continue
        array = numpy_helper.to_array(initializer).astype(np.float32)
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(array, name=initializer.name)
        )
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
    for value in list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT16:
            tensor_type.elem_type = TensorProto.FLOAT


def make_direct_truthful(source: onnx.ModelProto, example: np.ndarray) -> onnx.ModelProto:
    model = copy.deepcopy(source)

    # Shape(qcol) and the three CenterCropPads are only shape-cloak carriers.
    # The CastLike of the full input would itself create a truthful 18,000-byte
    # tensor, so the already-FLOAT graph input is used directly instead.
    remove_outputs = {"q_abs_shape", "input_hid", "input_f16", "qcol_abs_hid", "qrow_abs_hid"}
    kept = [
        node
        for node in model.graph.node
        if not any(output in remove_outputs for output in node.output)
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    replace_all_uses(model, "input_f16", "input")
    replace_all_uses(model, "qcol_abs_hid", "qcol")
    replace_all_uses(model, "qrow_abs_hid", "qrow")
    widen_float16(model)

    types = type_map(model)
    shapes = trace_all_shapes(model, types, example)
    del model.graph.value_info[:]
    graph_outputs = {value.name for value in model.graph.output}
    for node in model.graph.node:
        for name in node.output:
            if not name or name in graph_outputs:
                continue
            model.graph.value_info.append(
                helper.make_tensor_value_info(name, types[name], shapes[name])
            )
    return model


def strict_audit(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checker_full": False,
        "strict_shape": False,
        "strict_shape_data_prop": False,
        "static_positive_intermediates": False,
        "conv_family_count": 0,
    }
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        result["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True)
        result["strict_shape"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        result["strict_shape_data_prop"] = True
        result["static_positive_intermediates"] = all(
            value.type.tensor_type.HasField("shape")
            and all(
                dim.HasField("dim_value") and dim.dim_value > 0
                for dim in value.type.tensor_type.shape.dim
            )
            for value in list(inferred.graph.value_info) + list(inferred.graph.output)
        )
    except Exception as exc:  # noqa: BLE001
        result["strict_shape_data_prop_error"] = f"{type(exc).__name__}: {exc}"
    result["conv_family_count"] = sum(
        node.op_type in {"Conv", "ConvTranspose", "QLinearConv", "ConvInteger"}
        for node in model.graph.node
    )
    result["ub0"] = result["conv_family_count"] == 0
    result["pass"] = all(
        result[key]
        for key in (
            "checker_full",
            "strict_shape",
            "strict_shape_data_prop",
            "static_positive_intermediates",
            "ub0",
        )
    )
    return result


def score(model: onnx.ModelProto, label: str) -> dict[str, Any] | None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from lib import scoring  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix=f"task138_{label}_") as workdir:
        return scoring.score_and_verify(
            copy.deepcopy(model), TASK, workdir, label=label, require_correct=False
        )


def initializer_aliases(model: onnx.ModelProto) -> list[list[str]]:
    groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for initializer in model.graph.initializer:
        array = np.ascontiguousarray(numpy_helper.to_array(initializer))
        key = (array.dtype.str, tuple(int(x) for x in array.shape), array.tobytes())
        groups[key].append(initializer.name)
    return [names for names in groups.values() if len(names) > 1]


def memory_inventory(model: onnx.ModelProto) -> dict[str, Any]:
    by_name: list[dict[str, Any]] = []
    by_dtype: Counter[str] = Counter()
    for value in model.graph.value_info:
        shape = dims(value)
        dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        size = math.prod(int(x) for x in shape if x is not None) * np.dtype(dtype).itemsize
        by_name.append(
            {
                "name": value.name,
                "shape": shape,
                "dtype": np.dtype(dtype).name,
                "bytes": int(size),
            }
        )
        by_dtype[np.dtype(dtype).name] += int(size)
    by_name.sort(key=lambda row: (-row["bytes"], row["name"]))
    return {
        "sum_bytes": sum(row["bytes"] for row in by_name),
        "by_dtype": dict(sorted(by_dtype.items())),
        "top": by_name[:40],
        "all": by_name,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source hash changed")
    source = onnx.load(SOURCE)
    example = one_hot_example()

    source_declared = declared_shapes(source)
    source_runtime = trace_all_shapes(source, type_map(source), example)
    mismatches = [
        {
            "name": name,
            "declared": source_declared[name],
            "runtime": runtime,
        }
        for name, runtime in source_runtime.items()
        if name in source_declared and source_declared[name] != runtime
    ]
    direct_cloak = [
        row
        for row in mismatches
        if row["name"] in {"input_hid", "qcol_abs_hid", "qrow_abs_hid"}
    ]

    truthful = make_direct_truthful(source, example)
    REJECTED.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(truthful, REJECTED)
    truthful_runtime = trace_all_shapes(truthful, type_map(truthful), example)
    truthful_declared = declared_shapes(truthful)
    truthful_mismatches = [
        name
        for name, runtime in truthful_runtime.items()
        if truthful_declared.get(name) != runtime
    ]

    source_score = score(source, "source")
    truthful_score = score(truthful, "truthful")
    source_declared_inventory = memory_inventory(source)
    inventory = memory_inventory(truthful)

    # Exact local accounting.  Equal+Cast can replace each pair of exact-line
    # comparisons and casts (up/down/left/right), and And+Cast can replace each
    # interior comparison pair.  This is an optimistic upper bound: it ignores
    # the new Equal/And boolean output, so it cannot overstate impossibility.
    exact_line_current = 4 * (30 + 30 + 120 + 120)
    exact_line_optimistic = 4 * 120
    interior_current = 2 * (30 + 30 + 120 + 120)
    interior_optimistic = 2 * 120
    mask_saving_upper_bound = (
        exact_line_current
        - exact_line_optimistic
        + interior_current
        - interior_optimistic
    )
    optimistic_cost_after_local_fusions = (
        int(truthful_score["cost"]) - mask_saving_upper_bound
        if truthful_score is not None
        else None
    )
    # A schema-realistic Equal/And replacement also materializes one 30-byte
    # BOOL result per fused pair.  Keep both this estimate and the stronger
    # impossible best case above.
    mask_saving_schema_realistic = mask_saving_upper_bound - (6 * 30)
    realistic_cost_after_local_fusions = (
        int(truthful_score["cost"]) - mask_saving_schema_realistic
        if truthful_score is not None
        else None
    )

    terminal_names = {"Ttable", "Ctable", "Tcolor", "Ccolor", "rolemap"}
    terminal_rows = [row for row in inventory["all"] if row["name"] in terminal_names]
    terminal_bytes = sum(row["bytes"] for row in terminal_rows)

    result = {
        "task": TASK,
        "authority_cost": AUTHORITY_COST,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": SOURCE_SHA256,
            "nodes": len(source.graph.node),
            "initializers": len(source.graph.initializer),
            "score": source_score,
            "declared_static_memory": source_declared_inventory["sum_bytes"],
            "official_profile_runtime_uplift": (
                source_score["memory"] - source_declared_inventory["sum_bytes"]
                if source_score is not None
                else None
            ),
            "strict_static_checker": strict_audit(source),
            "runtime_shape_mismatch_count": len(mismatches),
            "direct_cloak_count": len(direct_cloak),
            "direct_cloak": direct_cloak,
            "all_runtime_shape_mismatches": mismatches,
        },
        "truthful_direct_normalization": {
            "path": str(REJECTED.relative_to(ROOT)),
            "sha256": sha256(REJECTED),
            "actions": [
                "remove Shape(qcol)",
                "remove three CenterCropPad cloak outputs",
                "use FLOAT input directly instead of materializing input_f16",
                "widen connected FLOAT16 arithmetic and initializers to FLOAT",
                "regenerate every intermediate value_info from ORT_DISABLE_ALL trace",
            ],
            "score": truthful_score,
            "strict": strict_audit(truthful),
            "runtime_shape_mismatch_count": len(truthful_mismatches),
            "runtime_shape_mismatches": truthful_mismatches,
            "inventory": inventory,
            "decision": "REJECT_NOT_STRICTLY_LOWER",
        },
        "mechanical_scans": {
            "exact_initializer_alias_groups": initializer_aliases(source),
            "exact_initializer_alias_count": len(initializer_aliases(source)),
            "dtype_narrowing": {
                "finding": "no byte saving",
                "reason": (
                    "coordinate/table carriers are already UINT8; Scatter/Gather indices "
                    "are schema-constrained INT32/INT64; parameter cost counts elements, not bytes"
                ),
            },
            "constant_fold": {
                "only_foldable_shape_carrier": "q_abs_shape=Shape(qcol)[axis 3]=[30]",
                "result": "folding exposes all three declared/runtime shape contradictions",
            },
            "initializer_factoring": [
                {
                    "proposal": "derive qrow from qcol by Transpose",
                    "parameter_delta": -30,
                    "activation_delta_bytes": 30,
                    "net_cost_best_case": 0,
                },
                {
                    "proposal": "derive ones30_f16 from pow_table",
                    "parameter_delta": -30,
                    "activation_delta_bytes_at_least": 60,
                    "net_cost_best_case": 30,
                },
            ],
        },
        "cost_accounting": {
            "terminal_rolemap_table_bytes": terminal_bytes,
            "terminal_breakdown": terminal_rows,
            "terminal_note": (
                "The current rolemap strategy necessarily materializes two 7x30 tables, "
                "two 10x30 gathered colors, and one length-10 INT32 rolemap."
            ),
            "mask_fusion_optimistic_saving_upper_bound": mask_saving_upper_bound,
            "optimistic_cost_after_all_local_mask_fusions": optimistic_cost_after_local_fusions,
            "mask_fusion_schema_realistic_saving": mask_saving_schema_realistic,
            "schema_realistic_cost_after_all_local_mask_fusions": realistic_cost_after_local_fusions,
            "gap_vs_authority_after_optimistic_local_fusions": (
                optimistic_cost_after_local_fusions - AUTHORITY_COST
                if optimistic_cost_after_local_fusions is not None
                else None
            ),
            "optimistic_cost_if_terminal_rolemap_table_chain_were_free": (
                optimistic_cost_after_local_fusions - terminal_bytes
                if optimistic_cost_after_local_fusions is not None
                else None
            ),
            "gap_vs_authority_even_if_terminal_chain_were_free": (
                optimistic_cost_after_local_fusions - terminal_bytes - AUTHORITY_COST
                if optimistic_cost_after_local_fusions is not None
                else None
            ),
        },
        "policy90": {
            "threshold": 0.90,
            "allowed_runtime_errors": 0,
            "eligible_survivors": [],
            "fresh_tests_run": 0,
            "reason": (
                "No truthful candidate is strictly lower than authority; fresh testing is "
                "therefore forbidden by the lane gate."
            ),
        },
        "decision": "EARLY_STOP_NO_TRUTHFUL_STRICT_LOWER",
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_score": source_score,
        "source_runtime_shape_mismatches": len(mismatches),
        "source_direct_cloaks": len(direct_cloak),
        "truthful_score": truthful_score,
        "truthful_runtime_shape_mismatches": len(truthful_mismatches),
        "optimistic_local_fusion_cost": optimistic_cost_after_local_fusions,
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
