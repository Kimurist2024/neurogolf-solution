#!/usr/bin/env python3
"""Fail-closed audit for the task349 residual exact/generator-support shave."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task349.onnx"
CANDIDATE = HERE / "candidates/task349_residual_patch_final.onnx"
RESULT = HERE / "audit_final.json"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_db93a21d.py"
COMMON = ROOT / "inputs/arc-gen-repo/tasks/common.py"
EXPECTED_AUTHORITY_SHA = "f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2"
EXPECTED_CANDIDATE_SHA = "8ab46bc1217c80c1d15c6064ea12a502c15274e12f79d9546f3d4620b76b72a3"
EXPECTED_GENERATOR_SHA = "680f6bdc6ac51591c42f0486a2db6cd8c430b3e218c93f1434838da125d92d62"
EXPECTED_COMMON_SHA = "56f9be46fc563f2754c3d506ff0d4aa8fa97c181c1c6db209a21ffd4707b9bc9"
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
FRESH = ((205_349_17, 2500), (205_349_31, 2500))

REVIEW_HELPERS = ROOT / "scripts/golf/loop_8004_42_plus20/agent_review_task349_affine_204"
sys.path.insert(0, str(REVIEW_HELPERS))
import audit as shared  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_proof() -> dict[str, Any]:
    generator_tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    common_tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    generate = next(n for n in generator_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    factor_range = False
    size_formula = False
    for node in ast.walk(generate):
        if not isinstance(node, ast.Assign):
            continue
        if (
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "factor"
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "common"
            and node.value.func.attr == "randint" and len(node.value.args) == 2
            and all(isinstance(arg, ast.Constant) for arg in node.value.args)
            and [arg.value for arg in node.value.args] == [2, 6]
        ):
            factor_range = True
        targets = [n.id for n in ast.walk(node.targets[0]) if isinstance(n, ast.Name)]
        if "size" in targets and isinstance(node.value, ast.Tuple):
            size_formula |= any(
                isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult)
                and isinstance(value.left, ast.Constant) and value.left.value == 5
                and isinstance(value.right, ast.Name) and value.right.id == "factor"
                for value in node.value.elts
            )
    randint = next(n for n in common_tree.body if isinstance(n, ast.FunctionDef) and n.name == "randint")
    inclusive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "random" and n.func.attr == "randint"
        for n in ast.walk(randint)
    )
    sides = [10, 15, 20, 25, 30]
    return {
        "generator_sha256": sha256(GENERATOR),
        "common_sha256": sha256(COMMON),
        "factor_range_2_through_6": factor_range,
        "python_randint_inclusive": inclusive,
        "size_equals_5_times_factor": size_formula,
        "supported_sides": sides,
        "pass": bool(factor_range and inclusive and size_formula),
    }


def init_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def exact_node(node: onnx.NodeProto | None, op: str, inputs: list[str], outputs: list[str]) -> bool:
    return node is not None and node.op_type == op and list(node.input) == inputs and list(node.output) == outputs


def semantic_and_mechanical_proof() -> dict[str, Any]:
    source = onnx.load(AUTHORITY)
    candidate = onnx.load(CANDIDATE)
    src_arr = init_arrays(source)
    cand_arr = init_arrays(candidate)
    src_init = {x.name: x for x in source.graph.initializer}
    cand_init = {x.name: x for x in candidate.graph.initializer}
    src_prod = producer_map(source)
    cand_prod = producer_map(candidate)

    sides = [10, 15, 20, 25, 30]
    valid_rows = []
    width = src_arr["affine_width_factor"].astype(np.int64)
    table = src_arr["valid_cols_table"].astype(np.int64)
    for side in sides:
        neg_power = int(width[-side])
        derived = int(np.int32(~np.int32(neg_power)))
        expected = int(table[side // 5])
        valid_rows.append({
            "side": side, "negative_index": -side, "negative_power": neg_power,
            "derived": derived, "expected": expected, "equal": derived == expected == (1 << side) - 1,
        })

    radius = src_arr["hend_offset_by_mod_i8"].astype(np.int16)
    shifts = src_arr["shift_by_mod"].astype(np.int64)
    shift_rows = [
        {"code": i, "radius": int(r), "shift": int(shifts[i]), "derived": 1 << int(r),
         "equal": int(shifts[i]) == 1 << int(r)}
        for i, r in enumerate(radius)
    ]

    side_rows = []
    coords_i32 = src_arr["coords4"].reshape(-1).astype(np.int32)
    coords_i8 = cand_arr["coords4"].reshape(-1).astype(np.int8)
    for side in sides:
        side_f = np.sqrt(np.float32(side * side), dtype=np.float32)
        authority_side = np.int8(np.int32(side_f))
        candidate_side = np.int8(side_f)
        source_beam = np.asarray(np.clip(authority_side, np.int8(0), np.int8(29)), dtype=np.int8).reshape(1, 1, 1, 1)
        candidate_beam = np.minimum(candidate_side, np.full((1, 1, 1, 1), 29, dtype=np.int8))
        source_valid_rows = coords_i32 < np.int32(authority_side)
        candidate_valid_rows = coords_i8 < candidate_side
        side_rows.append({
            "side": side,
            "sqrt_f32": float(side_f),
            "authority_cast_i32_i8": int(authority_side),
            "candidate_cast_i8": int(candidate_side),
            "negative_i8": int(np.int8(-candidate_side)),
            "beam_end_index_equal": bool(np.array_equal(source_beam, candidate_beam)),
            "valid_rows_equal": bool(np.array_equal(source_valid_rows, candidate_valid_rows)),
        })

    expected_source_only = {"shift_by_mod", "valid_cols_table", "five_i32", "unsq4"}
    expected_candidate_only = {
        "one_u8", "max29_rank4_i8", "special_patch_sig", "special_h_indices_i8", "special_h_values"
    }
    changed_common = {"coords4", "h_patch_indices_i8", "h_patch_sigs", "h_patch_values"}
    common_unchanged = sorted((set(src_init) & set(cand_init)) - changed_common)
    common_initializers_identical = all(
        src_init[name].SerializeToString() == cand_init[name].SerializeToString()
        for name in common_unchanged
    )

    affected_outputs = {
        "side", "side_i8", "side_factor", "valid_cols", "shift_factor",
        "neg_side_index_i8", "neg_side_index", "neg_valid_cols_plus1", "radius_u8", "shift_u8",
        "beam_end_scalar_i8", "beam_end_index_i8", "beam_indices_i8", "beam_end_is30",
        "halo_indices_i8", "halo_updates", "sp_has_sig", "sp_bupdate", "valid_rows4",
        "special_patch_cond", "special_h_updates",
    }
    src_other = [
        n.SerializeToString() for n in source.graph.node if not any(o in affected_outputs for o in n.output)
    ]
    cand_other = [
        n.SerializeToString() for n in candidate.graph.node if not any(o in affected_outputs for o in n.output)
    ]
    source_skeleton = copy.deepcopy(source)
    candidate_skeleton = copy.deepcopy(candidate)
    del source_skeleton.graph.node[:]
    del source_skeleton.graph.initializer[:]
    del candidate_skeleton.graph.node[:]
    del candidate_skeleton.graph.initializer[:]

    source_h_i = src_arr["h_patch_indices_i8"].reshape(-1)
    source_h_s = src_arr["h_patch_sigs"].reshape(-1)
    source_h_v = src_arr["h_patch_values"].reshape(-1)
    reconstructed_i = np.concatenate([
        cand_arr["h_patch_indices_i8"].reshape(-1), cand_arr["special_h_indices_i8"].reshape(-1)
    ])
    reconstructed_s = np.concatenate([
        cand_arr["h_patch_sigs"].reshape(-1),
        np.repeat(cand_arr["special_patch_sig"].reshape(-1), 2),
    ])
    reconstructed_v = np.concatenate([
        cand_arr["h_patch_values"].reshape(-1), cand_arr["special_h_values"].reshape(-1)
    ])

    node_checks = {
        "side_direct_i8": exact_node(cand_prod.get("side_i8"), "Cast", ["side_f"], ["side_i8"]),
        "negative_side_i8": exact_node(cand_prod.get("neg_side_index_i8"), "Neg", ["side_i8"], ["neg_side_index_i8"]),
        "negative_side_i32": exact_node(cand_prod.get("neg_side_index"), "Cast", ["neg_side_index_i8"], ["neg_side_index"]),
        "valid_negative_gather": exact_node(cand_prod.get("neg_valid_cols_plus1"), "Gather", ["affine_width_factor", "neg_side_index"], ["neg_valid_cols_plus1"]),
        "valid_bitwise_not": exact_node(cand_prod.get("valid_cols"), "BitwiseNot", ["neg_valid_cols_plus1"], ["valid_cols"]),
        "radius_u8": exact_node(cand_prod.get("radius_u8"), "Cast", ["hend_offset_i8"], ["radius_u8"]),
        "shift_u8": exact_node(cand_prod.get("shift_u8"), "BitShift", ["one_u8", "radius_u8"], ["shift_u8"]),
        "shift_i32": exact_node(cand_prod.get("shift_factor"), "Cast", ["shift_u8"], ["shift_factor"]),
        "beam_min": exact_node(cand_prod.get("beam_end_scalar_i8"), "Min", ["side_i8", "max29_rank4_i8"], ["beam_end_scalar_i8"]),
        "special_condition": exact_node(cand_prod.get("special_patch_cond"), "Equal", ["patch_sumR", "special_patch_sig"], ["special_patch_cond"]),
        "special_h_updates": exact_node(cand_prod.get("special_h_updates"), "Where", ["special_patch_cond", "special_h_values", "zero_i32"], ["special_h_updates"]),
        "special_beam_condition_reused": cand_prod.get("sp_bupdate") is not None and list(cand_prod["sp_bupdate"].input)[0] == "special_patch_cond",
    }

    proof = {
        "valid_cols_rows": valid_rows,
        "valid_cols_all_supported_sides": all(row["equal"] for row in valid_rows),
        "shift_rows": shift_rows,
        "shift_all_11_codes": all(row["equal"] for row in shift_rows),
        "shift_intermediate_uint8_range": [int(shifts.min()), int(shifts.max())],
        "side_rows": side_rows,
        "side_support_equivalent": all(
            row["authority_cast_i32_i8"] == row["candidate_cast_i8"] == row["side"]
            and row["negative_i8"] == -row["side"]
            and row["beam_end_index_equal"] and row["valid_rows_equal"]
            for row in side_rows
        ),
        "source_only_initializers": sorted(set(src_init) - set(cand_init)),
        "candidate_only_initializers": sorted(set(cand_init) - set(src_init)),
        "common_unchanged_initializers_byte_identical": common_initializers_identical,
        "coords_narrow_exact": bool(
            cand_arr["coords4"].dtype == np.int8 and np.array_equal(coords_i32, coords_i8.astype(np.int32))
        ),
        "max29_rank4_exact": bool(
            cand_arr["max29_rank4_i8"].dtype == np.int8
            and cand_arr["max29_rank4_i8"].shape == (1, 1, 1, 1)
            and int(cand_arr["max29_rank4_i8"].reshape(-1)[0]) == 29
        ),
        "h_patch_indices_reconstruct": bool(np.array_equal(source_h_i, reconstructed_i)),
        "h_patch_signatures_reconstruct": bool(np.array_equal(source_h_s, reconstructed_s)),
        "h_patch_values_reconstruct": bool(np.array_equal(source_h_v, reconstructed_v)),
        "node_checks": node_checks,
        "all_unaffected_nodes_byte_identical_and_ordered": src_other == cand_other,
        "all_other_model_fields_byte_identical": source_skeleton.SerializeToString() == candidate_skeleton.SerializeToString(),
    }
    proof["pass"] = bool(
        proof["valid_cols_all_supported_sides"]
        and proof["shift_all_11_codes"]
        and proof["shift_intermediate_uint8_range"] == [1, 32]
        and proof["side_support_equivalent"]
        and set(proof["source_only_initializers"]) == expected_source_only
        and set(proof["candidate_only_initializers"]) == expected_candidate_only
        and proof["common_unchanged_initializers_byte_identical"]
        and proof["coords_narrow_exact"] and proof["max29_rank4_exact"]
        and proof["h_patch_indices_reconstruct"] and proof["h_patch_signatures_reconstruct"]
        and proof["h_patch_values_reconstruct"] and all(node_checks.values())
        and proof["all_unaffected_nodes_byte_identical_and_ordered"]
        and proof["all_other_model_fields_byte_identical"]
    )
    return proof


def main() -> None:
    ort.set_default_logger_severity(4)
    before = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    if before != ROOT_GUARDS:
        raise RuntimeError(f"root guard mismatch: {before}")
    if sha256(AUTHORITY) != EXPECTED_AUTHORITY_SHA or sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("authority/candidate SHA mismatch")
    if sha256(GENERATOR) != EXPECTED_GENERATOR_SHA or sha256(COMMON) != EXPECTED_COMMON_SHA:
        raise RuntimeError("generator source SHA mismatch")

    shared.AUTHORITY = AUTHORITY
    shared.CANDIDATE = CANDIDATE
    profiles = {"authority": shared.profile(AUTHORITY), "candidate": shared.profile(CANDIDATE)}
    static = {"authority": shared.structure(AUTHORITY), "candidate": shared.structure(CANDIDATE)}
    shape_truth = shared.runtime_shape_truth()
    generator = generator_proof()
    proof = semantic_and_mechanical_proof()

    known = shared.known_cases()
    known_results = {}
    for disable, threads, label in CONFIGS:
        known_results[label] = shared.evaluate(known, disable, threads)

    fresh_results = []
    for seed, count in FRESH:
        cases, attempts = shared.fresh_cases(seed, count)
        stream = {"seed": seed, "requested": count, "generated": len(cases), "attempts": attempts, "configs": {}}
        for disable, threads, label in CONFIGS:
            stream["configs"][label] = shared.evaluate(cases, disable, threads)
        fresh_results.append(stream)
        print(f"fresh seed={seed} complete cases={len(cases)}", flush=True)

    comparisons = list(known_results.values()) + [
        row for stream in fresh_results for row in stream["configs"].values()
    ]
    after = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    authority_after = sha256(AUTHORITY)
    structure_pass = all((
        static["candidate"]["full_check"], static["candidate"]["strict"],
        static["candidate"]["strict_data_prop"], static["candidate"]["standard_domain_only"],
        static["candidate"]["functions"] == 0, static["candidate"]["sparse_initializers"] == 0,
        static["candidate"]["nested_graphs"] == 0, static["candidate"]["conv_family_nodes"] == 0,
        static["candidate"]["nonfinite_initializers"] == 0,
    ))
    report: dict[str, Any] = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": EXPECTED_AUTHORITY_SHA},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": EXPECTED_CANDIDATE_SHA},
        "profiles": profiles,
        "strict_lower": profiles["candidate"]["cost"] < profiles["authority"]["cost"],
        "cost_delta": profiles["authority"]["cost"] - profiles["candidate"]["cost"],
        "projected_log_gain": math.log(profiles["authority"]["cost"] / profiles["candidate"]["cost"]),
        "generator_proof": generator,
        "semantic_and_mechanical_proof": proof,
        "static": static,
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
        "authority_sha_after": authority_after,
    }
    report["pass"] = bool(
        report["strict_lower"] and report["cost_delta"] == 16
        and generator["pass"] and proof["pass"] and structure_pass and shape_truth["truthful"]
        and report["all_raw_bitwise_equivalent"]
        and report["runtime_errors_total"] == 0 and report["nonfinite_values_total"] == 0
        and before == after == ROOT_GUARDS and authority_after == EXPECTED_AUTHORITY_SHA
    )
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"], "profiles": profiles, "cost_delta": report["cost_delta"],
        "proof": proof["pass"], "known": len(known),
        "fresh": [stream["generated"] for stream in fresh_results],
        "minimum_fresh_candidate_accuracy": report["minimum_fresh_candidate_accuracy"],
        "runtime_errors_total": report["runtime_errors_total"],
        "nonfinite_values_total": report["nonfinite_values_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
