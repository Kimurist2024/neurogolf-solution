#!/usr/bin/env python3
"""Independent fail-closed audit for task349 residual candidate 209."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task349.onnx"
CANDIDATE = ROOT / (
    "scripts/golf/loop_8004_42_plus20/agent_task349_residual_205/"
    "candidates/task349_residual_patch_final.onnx"
)
RESULT = HERE / "audit.json"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_db93a21d.py"
COMMON = ROOT / "inputs/arc-gen-repo/tasks/common.py"
EXPECTED_AUTHORITY_SHA = "f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2"
EXPECTED_CANDIDATE_SHA = "8ab46bc1217c80c1d15c6064ea12a502c15274e12f79d9546f3d4620b76b72a3"
ROOT_GUARDS = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((209_349_73, 2500), (209_349_91, 2500))
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    out: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            out.append(dim.dim_param)
        else:
            out.append(None)
    return out


def nested_graph_count(model: onnx.ModelProto) -> int:
    result = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                result += 1
                pending.extend(attr.g.node)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                result += len(attr.graphs)
                for graph in attr.graphs:
                    pending.extend(graph.node)
    return result


def static_structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    init_names = {x.name for x in model.graph.initializer}
    used_init_names = {name for node in model.graph.node for name in node.input if name in init_names}
    conv_ub = []
    init_by_name = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose", "QLinearConv"}:
            continue
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3:
            weight = init_by_name.get(node.input[1])
            bias = init_by_name.get(node.input[2])
            if weight is not None and bias is not None:
                out_channels = int(weight.shape[1] if node.op_type == "ConvTranspose" else weight.shape[0])
                if bias.size < out_channels:
                    conv_ub.append({"node": node.name, "bias": int(bias.size), "out_channels": out_channels})
    row: dict[str, Any] = {
        "sha256": sha256(path),
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": int(sum(np.asarray(numpy_helper.to_array(x)).size for x in model.graph.initializer)),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested_graph_count(model),
        "standard_domain_only": all(item.domain in {"", "ai.onnx"} for item in model.opset_import),
        "banned_nodes": sorted({
            node.op_type for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }),
        "unused_initializers": sorted(init_names - used_init_names),
        "conv_short_bias_ub": conv_ub,
        "nonfinite_initializers": int(sum(
            np.asarray(numpy_helper.to_array(x)).size
            - np.count_nonzero(np.isfinite(np.asarray(numpy_helper.to_array(x))))
            for x in model.graph.initializer
            if np.asarray(numpy_helper.to_array(x)).dtype.kind in "fc"
        )),
    }
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["full_check"] = False
        row["full_check_error"] = f"{type(exc).__name__}: {exc}"
    for data_prop in (False, True):
        key = "strict_data_prop" if data_prop else "strict"
        try:
            shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=data_prop)
            row[key] = True
        except Exception as exc:  # noqa: BLE001
            row[key] = False
            row[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    return row


def generator_support_proof() -> dict[str, Any]:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    common_tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    generate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    factor_range = False
    size_formula = False
    for node in ast.walk(generate):
        if not isinstance(node, ast.Assign):
            continue
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "factor"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "common"
            and node.value.func.attr == "randint"
            and [getattr(x, "value", None) for x in node.value.args] == [2, 6]
        ):
            factor_range = True
        if isinstance(node.value, ast.Tuple):
            size_formula |= any(
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Mult)
                and isinstance(value.left, ast.Constant)
                and value.left.value == 5
                and isinstance(value.right, ast.Name)
                and value.right.id == "factor"
                for value in node.value.elts
            )
    randint_fn = next(n for n in common_tree.body if isinstance(n, ast.FunctionDef) and n.name == "randint")
    inclusive = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "random"
        and n.func.attr == "randint"
        for n in ast.walk(randint_fn)
    )
    sides = [5 * factor for factor in range(2, 7)]
    return {
        "generator_sha256": sha256(GENERATOR),
        "common_sha256": sha256(COMMON),
        "factor_common_randint_2_6": factor_range,
        "randint_inclusive": inclusive,
        "size_is_5_factor": size_formula,
        "supported_sides": sides,
        "one_hot_area_values": [side * side for side in sides],
        "pass": bool(factor_range and inclusive and size_formula and sides == [10, 15, 20, 25, 30]),
    }


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}


def producer(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def attr_string(node: onnx.NodeProto, name: str) -> bytes | None:
    for attr in node.attribute:
        if attr.name == name:
            return bytes(attr.s)
    return None


def mechanical_and_semantic_proof() -> dict[str, Any]:
    source = onnx.load(AUTHORITY)
    candidate = onnx.load(CANDIDATE)
    sa, ca = arrays(source), arrays(candidate)
    si = {x.name: x for x in source.graph.initializer}
    ci = {x.name: x for x in candidate.graph.initializer}
    sp, cp = producer(source), producer(candidate)
    sides = [10, 15, 20, 25, 30]

    # Transformation 1: side/5 lookup is exactly 2^side-1, recovered from
    # affine_width_factor[-side] == -2^side and int32 BitwiseNot.
    valid_rows = []
    for side in sides:
        source_value = int(sa["valid_cols_table"][side // 5])
        negative_power = int(sa["affine_width_factor"][-side])
        recovered = int(np.bitwise_not(np.int32(negative_power)))
        valid_rows.append({
            "side": side,
            "source_table": source_value,
            "negative_gather": negative_power,
            "candidate_recovered": recovered,
            "expected": (1 << side) - 1,
            "equal": source_value == recovered == (1 << side) - 1,
        })
    valid_node_exact = bool(
        cp["neg_side_index_i8"].op_type == "Neg"
        and list(cp["neg_side_index_i8"].input) == ["side_i8"]
        and cp["neg_side_index"].op_type == "Cast"
        and list(cp["neg_side_index"].input) == ["neg_side_index_i8"]
        and cp["neg_valid_cols_plus1"].op_type == "Gather"
        and list(cp["neg_valid_cols_plus1"].input) == ["affine_width_factor", "neg_side_index"]
        and cp["valid_cols"].op_type == "BitwiseNot"
        and list(cp["valid_cols"].input) == ["neg_valid_cols_plus1"]
    )

    # Transformation 2: all radius_code results are modulo 11. Enumerate all
    # table rows and verify LEFT uint8 shift has no overflow.
    radii = sa["hend_offset_by_mod_i8"].astype(np.int64).reshape(-1)
    shifts = sa["shift_by_mod"].astype(np.int64).reshape(-1)
    shift_rows = []
    for code, (radius, old_shift) in enumerate(zip(radii, shifts)):
        new_shift = int(np.left_shift(np.uint8(1), np.uint8(radius)))
        shift_rows.append({
            "code": code,
            "radius": int(radius),
            "source": int(old_shift),
            "candidate_uint8": new_shift,
            "equal": int(old_shift) == new_shift,
        })
    shift_node_exact = bool(
        cp["radius_u8"].op_type == "Cast"
        and list(cp["radius_u8"].input) == ["hend_offset_i8"]
        and cp["shift_u8"].op_type == "BitShift"
        and list(cp["shift_u8"].input) == ["one_u8", "radius_u8"]
        and attr_string(cp["shift_u8"], "direction") == b"LEFT"
        and cp["shift_factor"].op_type == "Cast"
        and list(cp["shift_factor"].input) == ["shift_u8"]
        and ca["one_u8"].dtype == np.uint8
        and int(ca["one_u8"]) == 1
    )

    # Transformation 3: exact square areas and 0..29 coordinates fit int8.
    coord_rows = []
    source_coords = sa["coords4"].reshape(-1)
    candidate_coords = ca["coords4"].reshape(-1)
    for side in sides:
        area = np.float32(side * side)
        root = np.sqrt(area, dtype=np.float32)
        old_side = np.int8(np.int32(root))
        new_side = np.int8(root)
        coord_rows.append({
            "side": side,
            "sqrt_f32_bits": int(root.view(np.uint32)),
            "source_side": int(old_side),
            "candidate_side": int(new_side),
            "neg_side": int(np.int8(-new_side)),
            "row_masks_equal": bool(np.array_equal(source_coords < np.int32(old_side), candidate_coords < new_side)),
        })
    narrow_node_exact = bool(
        cp["side_i8"].op_type == "Cast"
        and list(cp["side_i8"].input) == ["side_f"]
        and ca["coords4"].dtype == np.int8
        and np.array_equal(source_coords, candidate_coords.astype(np.int32))
        and cp["valid_rows4"].op_type == "Less"
        and list(cp["valid_rows4"].input) == ["coords4", "side_i8"]
    )

    # Transformation 4: side is positive, therefore Clip(side,0,29) followed
    # by rank-4 Unsqueeze equals Min(side, rank4(29)).
    beam_rows = []
    for side in sides:
        old = np.asarray(np.clip(np.int8(side), np.int8(0), np.int8(29)), dtype=np.int8).reshape(1, 1, 1, 1)
        new = np.minimum(np.int8(side), ca["max29_rank4_i8"])
        beam_rows.append({"side": side, "source": int(old.item()), "candidate": int(new.item()), "equal": bool(np.array_equal(old, new))})
    beam_node_exact = bool(
        cp["beam_end_scalar_i8"].op_type == "Min"
        and list(cp["beam_end_scalar_i8"].input) == ["side_i8", "max29_rank4_i8"]
        and ca["max29_rank4_i8"].dtype == np.int8
        and ca["max29_rank4_i8"].shape == (1, 1, 1, 1)
        and int(ca["max29_rank4_i8"].item()) == 29
        and list(cp["beam_indices_i8"].input) == ["beam_starts30_i8", "beam_end_scalar_i8", "sp_bidx_i8"]
    )

    # Transformation 5: split the final duplicate signature pair from the
    # six-entry broadcast table, reuse exactly the same equality for beam.
    reconstructed_indices = np.concatenate([
        ca["h_patch_indices_i8"].reshape(-1), ca["special_h_indices_i8"].reshape(-1)
    ])
    reconstructed_sigs = np.concatenate([
        ca["h_patch_sigs"].reshape(-1), np.repeat(ca["special_patch_sig"].reshape(-1), 2)
    ])
    reconstructed_values = np.concatenate([
        ca["h_patch_values"].reshape(-1), ca["special_h_values"].reshape(-1)
    ])
    source_sig_index = int(sa["k6"].reshape(-1)[0])
    special_exact = bool(
        np.array_equal(reconstructed_indices, sa["h_patch_indices_i8"].reshape(-1))
        and np.array_equal(reconstructed_sigs, sa["h_patch_sigs"].reshape(-1))
        and np.array_equal(reconstructed_values, sa["h_patch_values"].reshape(-1))
        and source_sig_index == 4
        and int(sa["h_patch_sigs"].reshape(-1)[source_sig_index]) == int(ca["special_patch_sig"])
        and cp["special_patch_cond"].op_type == "Equal"
        and list(cp["special_patch_cond"].input) == ["patch_sumR", "special_patch_sig"]
        and cp["special_h_updates"].op_type == "Where"
        and list(cp["special_h_updates"].input) == ["special_patch_cond", "special_h_values", "zero_i32"]
        and list(cp["sp_bupdate"].input) == ["special_patch_cond", "sp_bval", "zero_i32"]
        and list(cp["halo_indices_i8"].input) == ["halo_start30", "halo_end30", "h_patch_indices_i8", "special_h_indices_i8"]
        and list(cp["halo_updates"].input) == ["X", "neg_X_stop", "h_patch_updates", "special_h_updates"]
    )

    source_primary = {node.output[0]: node for node in source.graph.node}
    candidate_primary = {node.output[0]: node for node in candidate.graph.node}
    common_outputs = set(source_primary) & set(candidate_primary)
    changed_common_outputs = sorted(
        output for output in common_outputs
        if source_primary[output].SerializeToString() != candidate_primary[output].SerializeToString()
    )
    expected_changed = sorted([
        "beam_end_scalar_i8", "beam_indices_i8", "halo_indices_i8", "halo_updates",
        "shift_factor", "side_i8", "sp_bupdate", "valid_cols", "valid_rows4",
    ])
    expected_source_only_outputs = sorted(["beam_end_index_i8", "side", "side_factor", "sp_has_sig"])
    expected_candidate_only_outputs = sorted([
        "neg_side_index", "neg_side_index_i8", "neg_valid_cols_plus1", "radius_u8",
        "shift_u8", "special_h_updates", "special_patch_cond",
    ])
    source_only_outputs = sorted(set(source_primary) - set(candidate_primary))
    candidate_only_outputs = sorted(set(candidate_primary) - set(source_primary))

    source_only_init = sorted(set(si) - set(ci))
    candidate_only_init = sorted(set(ci) - set(si))
    changed_common_init = sorted(
        name for name in set(si) & set(ci)
        if si[name].SerializeToString() != ci[name].SerializeToString()
    )
    unchanged_common_init = sorted((set(si) & set(ci)) - set(changed_common_init))

    source_skeleton = copy.deepcopy(source)
    candidate_skeleton = copy.deepcopy(candidate)
    del source_skeleton.graph.node[:]
    del source_skeleton.graph.initializer[:]
    del candidate_skeleton.graph.node[:]
    del candidate_skeleton.graph.initializer[:]
    protobuf_scope = {
        "source_nodes": len(source.graph.node),
        "candidate_nodes": len(candidate.graph.node),
        "unchanged_common_primary_output_nodes": sum(
            source_primary[o].SerializeToString() == candidate_primary[o].SerializeToString()
            for o in common_outputs
        ),
        "changed_common_outputs": changed_common_outputs,
        "source_only_outputs": source_only_outputs,
        "candidate_only_outputs": candidate_only_outputs,
        "unchanged_common_initializers": len(unchanged_common_init),
        "changed_common_initializers": changed_common_init,
        "source_only_initializers": source_only_init,
        "candidate_only_initializers": candidate_only_init,
        "all_unchanged_common_initializers_byte_identical": all(
            si[name].SerializeToString() == ci[name].SerializeToString() for name in unchanged_common_init
        ),
        "all_other_model_fields_byte_identical": source_skeleton.SerializeToString() == candidate_skeleton.SerializeToString(),
    }
    protobuf_scope["pass"] = bool(
        protobuf_scope["unchanged_common_primary_output_nodes"] == 99
        and changed_common_outputs == expected_changed
        and source_only_outputs == expected_source_only_outputs
        and candidate_only_outputs == expected_candidate_only_outputs
        and changed_common_init == ["coords4", "h_patch_indices_i8", "h_patch_sigs", "h_patch_values"]
        and source_only_init == ["five_i32", "shift_by_mod", "unsq4", "valid_cols_table"]
        and candidate_only_init == [
            "max29_rank4_i8", "one_u8", "special_h_indices_i8", "special_h_values", "special_patch_sig"
        ]
        and protobuf_scope["all_unchanged_common_initializers_byte_identical"]
        and protobuf_scope["all_other_model_fields_byte_identical"]
    )

    proof = {
        "valid_cols_affine": {"rows": valid_rows, "nodes_exact": valid_node_exact},
        "shift_uint8": {"rows": shift_rows, "nodes_exact": shift_node_exact},
        "side_coords_narrow": {"rows": coord_rows, "nodes_exact": narrow_node_exact},
        "beam_rank_min": {"rows": beam_rows, "nodes_exact": beam_node_exact},
        "h_patch_special_split": {
            "source_indices": sa["h_patch_indices_i8"].reshape(-1).tolist(),
            "reconstructed_indices": reconstructed_indices.tolist(),
            "source_signatures": sa["h_patch_sigs"].reshape(-1).tolist(),
            "reconstructed_signatures": reconstructed_sigs.tolist(),
            "source_values": sa["h_patch_values"].reshape(-1).tolist(),
            "reconstructed_values": reconstructed_values.tolist(),
            "source_beam_mask_index": source_sig_index,
            "exact": special_exact,
        },
        "protobuf_scope": protobuf_scope,
    }
    proof["pass"] = bool(
        all(row["equal"] for row in valid_rows)
        and valid_node_exact
        and len(shift_rows) == 11
        and all(row["equal"] for row in shift_rows)
        and max(row["candidate_uint8"] for row in shift_rows) == 32
        and shift_node_exact
        and all(
            row["source_side"] == row["candidate_side"] == row["side"]
            and row["neg_side"] == -row["side"]
            and row["row_masks_equal"]
            for row in coord_rows
        )
        and narrow_node_exact
        and all(row["equal"] for row in beam_rows)
        and beam_node_exact
        and special_exact
        and protobuf_scope["pass"]
    )
    return proof


def known_cases() -> list[dict[str, Any]]:
    payload = scoring.load_examples(349)
    return [item for split in ("train", "test", "arc-gen") for item in payload.get(split, [])]


def fresh_cases(seed: int, count: int) -> tuple[list[dict[str, Any]], int]:
    module = importlib.import_module("task_db93a21d")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        case = module.generate()
        if scoring.convert_to_numpy(case) is not None:
            rows.append(case)
    if len(rows) != count:
        raise RuntimeError(f"fresh generation shortfall {len(rows)}/{count}")
    return rows, attempts


def make_session(path: Path, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def evaluate(cases: list[dict[str, Any]], disable: bool, threads: int) -> dict[str, Any]:
    sessions = {
        "authority": make_session(AUTHORITY, disable, threads),
        "candidate": make_session(CANDIDATE, disable, threads),
    }
    row: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "right": {"authority": 0, "candidate": 0},
        "runtime_errors": {"authority": 0, "candidate": 0},
        "nonfinite_values": {"authority": 0, "candidate": 0},
        "first_failure": None,
    }
    for index, case in enumerate(cases):
        benchmark = scoring.convert_to_numpy(case)
        if benchmark is None:
            row["first_failure"] = row["first_failure"] or {"index": index, "error": "conversion_failed"}
            continue
        row["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                output = np.asarray(session.run(None, {session.get_inputs()[0].name: benchmark["input"]})[0])
                outputs[label] = output
                if output.dtype.kind in "fc":
                    row["nonfinite_values"][label] += int(output.size - np.count_nonzero(np.isfinite(output)))
                row["right"][label] += int(output.shape == expected.shape and np.array_equal(output > 0, expected))
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"][label] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index, "model": label, "error": f"{type(exc).__name__}: {exc}"
                }
        if len(outputs) == 2:
            old = np.ascontiguousarray(outputs["authority"])
            new = np.ascontiguousarray(outputs["candidate"])
            raw_equal = old.dtype == new.dtype and old.shape == new.shape and old.tobytes() == new.tobytes()
            row["raw_equal"] += int(raw_equal)
            row["threshold_equal"] += int(np.array_equal(old > 0, new > 0))
            if not raw_equal and row["first_failure"] is None:
                row["first_failure"] = {"index": index, "max_abs_delta": float(np.nanmax(np.abs(old.astype(np.float64) - new.astype(np.float64))))}
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())
    row["accuracy"] = {
        label: row["right"][label] / row["valid"] if row["valid"] else None
        for label in ("authority", "candidate")
    }
    row["exact_equivalent"] = bool(
        row["valid"] == len(cases)
        and row["raw_equal"] == len(cases)
        and row["threshold_equal"] == len(cases)
        and row["runtime_errors_total"] == 0
        and row["nonfinite_values_total"] == 0
    )
    return row


def runtime_shape_truth() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {value.name for value in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    benchmark = scoring.convert_to_numpy(known_cases()[0])
    if benchmark is None:
        return {"truthful": False, "error": "known conversion failed"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    outputs = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = 0
    for name, output in zip(names, outputs):
        value = np.asarray(output)
        if dims(typed[name]) != list(value.shape):
            mismatches.append({"name": name, "declared": dims(typed[name]), "actual": list(value.shape)})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    before = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    if before != ROOT_GUARDS:
        raise RuntimeError(f"root guard mismatch before: {before}")
    if sha256(AUTHORITY) != EXPECTED_AUTHORITY_SHA:
        raise RuntimeError("authority SHA mismatch")
    if sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("candidate SHA mismatch")

    profiles = {"authority": profile(AUTHORITY), "candidate": profile(CANDIDATE)}
    static = {"authority": static_structure(AUTHORITY), "candidate": static_structure(CANDIDATE)}
    generator = generator_support_proof()
    proof = mechanical_and_semantic_proof()
    shape_truth = runtime_shape_truth()

    known = known_cases()
    known_results = {
        label: evaluate(known, disable, threads) for disable, threads, label in CONFIGS
    }
    fresh_results = []
    for seed, count in FRESH:
        cases, attempts = fresh_cases(seed, count)
        stream = {"seed": seed, "requested": count, "generated": len(cases), "attempts": attempts, "configs": {}}
        for disable, threads, label in CONFIGS:
            stream["configs"][label] = evaluate(cases, disable, threads)
        fresh_results.append(stream)
        print(f"fresh seed={seed} complete cases={len(cases)}", flush=True)

    comparisons = list(known_results.values()) + [
        row for stream in fresh_results for row in stream["configs"].values()
    ]
    candidate_static = static["candidate"]
    static_pass = bool(
        candidate_static["full_check"]
        and candidate_static["strict"]
        and candidate_static["strict_data_prop"]
        and candidate_static["standard_domain_only"]
        and candidate_static["functions"] == 0
        and candidate_static["sparse_initializers"] == 0
        and candidate_static["nested_graphs"] == 0
        and not candidate_static["banned_nodes"]
        and not candidate_static["unused_initializers"]
        and not candidate_static["conv_short_bias_ub"]
        and candidate_static["nonfinite_initializers"] == 0
    )
    after = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    report: dict[str, Any] = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": sha256(AUTHORITY),
            "profile": profiles["authority"],
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
            "profile": profiles["candidate"],
        },
        "strict_lower": profiles["candidate"]["cost"] < profiles["authority"]["cost"],
        "cost_delta": profiles["authority"]["cost"] - profiles["candidate"]["cost"],
        "projected_log_gain": math.log(profiles["authority"]["cost"] / profiles["candidate"]["cost"]),
        "generator_support": generator,
        "semantic_proof": proof,
        "static": static,
        "static_pass": static_pass,
        "runtime_shape_truth": shape_truth,
        "known_count": len(known),
        "known_four_configs": known_results,
        "fresh": fresh_results,
        "all_raw_bitwise_equivalent": all(row["exact_equivalent"] for row in comparisons),
        "runtime_errors_total": sum(row["runtime_errors_total"] for row in comparisons),
        "nonfinite_values_total": sum(row["nonfinite_values_total"] for row in comparisons),
        "minimum_fresh_candidate_accuracy": min(
            row["accuracy"]["candidate"] for stream in fresh_results for row in stream["configs"].values()
        ),
        "root_guards_before": before,
        "root_guards_after": after,
    }
    report["pass"] = bool(
        report["strict_lower"]
        and report["cost_delta"] == 16
        and profiles["authority"] == {"memory": 3233, "params": 315, "cost": 3548}
        and profiles["candidate"] == {"memory": 3239, "params": 293, "cost": 3532}
        and generator["pass"]
        and proof["pass"]
        and static_pass
        and shape_truth["truthful"]
        and report["all_raw_bitwise_equivalent"]
        and report["runtime_errors_total"] == 0
        and report["nonfinite_values_total"] == 0
        and before == after == ROOT_GUARDS
    )
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "cost": [profiles["authority"]["cost"], profiles["candidate"]["cost"]],
        "known": len(known),
        "fresh": [stream["generated"] for stream in fresh_results],
        "minimum_fresh_candidate_accuracy": report["minimum_fresh_candidate_accuracy"],
        "all_raw_bitwise_equivalent": report["all_raw_bitwise_equivalent"],
        "runtime_errors_total": report["runtime_errors_total"],
        "nonfinite_values_total": report["nonfinite_values_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
