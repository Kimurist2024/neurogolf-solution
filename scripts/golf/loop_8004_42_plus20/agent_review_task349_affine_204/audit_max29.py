#!/usr/bin/env python3
"""Independent generator-support audit of task349's max30 scalar removal."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper, shape_inference

import audit as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task349.onnx"
INTERMEDIATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_task349_affine_203/task349_affine_no_scalar.onnx"
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_task349_affine_203/task349_affine_max29.onnx"
RESULT = HERE / "audit_max29.json"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_db93a21d.py"
COMMON = ROOT / "inputs/arc-gen-repo/tasks/common.py"

EXPECTED_AUTHORITY_SHA = "179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7"
EXPECTED_INTERMEDIATE_SHA = "849d49e462ca94b5e4f9120434a39e1982d9dce521863e80b29d80ad9b02406b"
EXPECTED_CANDIDATE_SHA = "f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2"
EXPECTED_GENERATOR_SHA = "680f6bdc6ac51591c42f0486a2db6cd8c430b3e218c93f1434838da125d92d62"
EXPECTED_COMMON_SHA = "56f9be46fc563f2754c3d506ff0d4aa8fa97c181c1c6db209a21ffd4707b9bc9"
FRESH = ((204_349_47, 2500), (204_349_61, 2500))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_ast_proof() -> dict[str, Any]:
    generator_tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    common_tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    generate = next(
        node for node in generator_tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    factor_range = False
    size_formula = False
    for node in ast.walk(generate):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "factor"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "common"
                and node.value.func.attr == "randint"
                and len(node.value.args) == 2
                and all(isinstance(arg, ast.Constant) for arg in node.value.args)
                and [arg.value for arg in node.value.args] == [2, 6]
            ):
                factor_range = True
            targets = [target.id for target in ast.walk(node.targets[0]) if isinstance(target, ast.Name)]
            if "size" in targets and isinstance(node.value, ast.Tuple):
                for value in node.value.elts:
                    if (
                        isinstance(value, ast.BinOp)
                        and isinstance(value.op, ast.Mult)
                        and isinstance(value.left, ast.Constant)
                        and value.left.value == 5
                        and isinstance(value.right, ast.Name)
                        and value.right.id == "factor"
                    ):
                        size_formula = True
    randint = next(
        node for node in common_tree.body if isinstance(node, ast.FunctionDef) and node.name == "randint"
    )
    randint_delegates_python_inclusive = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "random"
        and node.func.attr == "randint"
        for node in ast.walk(randint)
    )
    supported_sides = [5 * factor for factor in range(2, 7)]
    return {
        "generator_sha256": sha256(GENERATOR),
        "common_sha256": sha256(COMMON),
        "factor_is_common_randint_2_6": factor_range,
        "common_randint_delegates_random_randint": randint_delegates_python_inclusive,
        "size_is_5_times_factor": size_formula,
        "supported_square_sides": supported_sides,
        "maximum_side": max(supported_sides),
        "pass": bool(factor_range and randint_delegates_python_inclusive and size_formula),
    }


def dtype_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    return {
        value.name: int(value.type.tensor_type.elem_type)
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }


def mechanical_diff() -> dict[str, Any]:
    source = onnx.load(INTERMEDIATE)
    candidate = onnx.load(CANDIDATE)
    src_init = {x.name: x for x in source.graph.initializer}
    cand_init = {x.name: x for x in candidate.graph.initializer}
    src_arr = {name: np.asarray(numpy_helper.to_array(value)) for name, value in src_init.items()}
    producer = {output: node for node in candidate.graph.node for output in node.output}

    changed_outputs = {"halo_end_is30", "beam_end_is30"}
    source_changed = [n for n in source.graph.node if any(output in changed_outputs for output in n.output)]
    candidate_changed = [n for n in candidate.graph.node if any(output in changed_outputs for output in n.output)]
    source_by_output = {n.output[0]: n for n in source_changed}
    candidate_by_output = {n.output[0]: n for n in candidate_changed}
    exact_rewrites = {}
    for output, input_name in (("halo_end_is30", "halo_end"), ("beam_end_is30", "side_i8")):
        before = source_by_output.get(output)
        after = candidate_by_output.get(output)
        exact_rewrites[output] = bool(
            before is not None
            and before.op_type == "Equal"
            and list(before.input) == [input_name, "max30_i8"]
            and list(before.output) == [output]
            and after is not None
            and after.op_type == "Greater"
            and list(after.input) == [input_name, "max29_i8"]
            and list(after.output) == [output]
        )

    src_other_nodes = [n.SerializeToString() for n in source.graph.node if not any(o in changed_outputs for o in n.output)]
    cand_other_nodes = [n.SerializeToString() for n in candidate.graph.node if not any(o in changed_outputs for o in n.output)]
    common_names = sorted(set(src_init) & set(cand_init))
    common_initializers_identical = all(
        src_init[name].SerializeToString() == cand_init[name].SerializeToString() for name in common_names
    )
    src_skeleton = copy.deepcopy(source)
    cand_skeleton = copy.deepcopy(candidate)
    del src_skeleton.graph.node[:]
    del src_skeleton.graph.initializer[:]
    del cand_skeleton.graph.node[:]
    del cand_skeleton.graph.initializer[:]

    halo_clip = producer.get("halo_end")
    side_i8 = producer.get("side_i8")
    side = producer.get("side")
    side_f = producer.get("side_f")
    area = producer.get("area")
    types = dtype_map(candidate)
    types.update({name: int(value.data_type) for name, value in cand_init.items()})
    int8_names = ["halo_end", "side_i8", "max29_i8"]
    integer_truth_table = [
        {"x": x, "equal_30": x == 30, "greater_29": x > 29}
        for x in range(-128, 31)
    ]
    return {
        "source_only_initializers": sorted(set(src_init) - set(cand_init)),
        "candidate_only_initializers": sorted(set(cand_init) - set(src_init)),
        "max30_i8_exact": bool(
            "max30_i8" in src_arr
            and src_arr["max30_i8"].dtype == np.dtype(np.int8)
            and src_arr["max30_i8"].shape == ()
            and int(src_arr["max30_i8"]) == 30
        ),
        "max29_i8_exact": bool(
            "max29_i8" in src_arr
            and src_arr["max29_i8"].dtype == np.dtype(np.int8)
            and src_arr["max29_i8"].shape == ()
            and int(src_arr["max29_i8"]) == 29
        ),
        "exact_node_rewrites": exact_rewrites,
        "all_common_initializers_byte_identical": common_initializers_identical,
        "all_other_nodes_byte_identical_and_ordered": src_other_nodes == cand_other_nodes,
        "all_other_model_fields_byte_identical": src_skeleton.SerializeToString() == cand_skeleton.SerializeToString(),
        "halo_end_clip_is_zero_to_side": bool(
            halo_clip is not None
            and halo_clip.op_type == "Clip"
            and list(halo_clip.input)[1:] == ["zero_i8", "side_i8"]
        ),
        "side_integer_path": {
            "side_i8": None if side_i8 is None else [side_i8.op_type, list(side_i8.input)],
            "side": None if side is None else [side.op_type, list(side.input)],
            "side_f": None if side_f is None else [side_f.op_type, list(side_f.input)],
            "area": None if area is None else [area.op_type, list(area.input)],
        },
        "side_integer_path_exact": bool(
            side_i8 is not None and side_i8.op_type == "Cast" and list(side_i8.input) == ["side"]
            and side is not None and side.op_type == "Cast" and list(side.input) == ["side_f"]
            and side_f is not None and side_f.op_type == "Sqrt" and list(side_f.input) == ["area"]
            and area is not None and area.op_type == "ReduceSum" and list(area.input) == ["input"]
        ),
        "comparison_inputs_are_int8": all(types.get(name) == TensorProto.INT8 for name in int8_names),
        "integer_identity_checked_count": len(integer_truth_table),
        "equal30_iff_greater29_for_every_int8_x_le_30": all(
            row["equal_30"] == row["greater_29"] for row in integer_truth_table
        ),
    }


def support_trace(cases: list[dict[str, Any]]) -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    for name in ("side_i8", "halo_end"):
        if name not in {value.name for value in model.graph.output}:
            model.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    current = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    observed_sides: set[int] = set()
    halo_min = 127
    halo_max = -128
    bound_violations = 0
    errors = 0
    for case in cases:
        benchmark = base.scoring.convert_to_numpy(case)
        if benchmark is None:
            errors += 1
            continue
        try:
            side_value, halo_value = current.run(
                ["side_i8", "halo_end"], {current.get_inputs()[0].name: benchmark["input"]}
            )
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        side_array = np.asarray(side_value)
        halo_array = np.asarray(halo_value)
        side_int = int(side_array.reshape(-1)[0])
        observed_sides.add(side_int)
        if halo_array.size:
            halo_min = min(halo_min, int(halo_array.min()))
            halo_max = max(halo_max, int(halo_array.max()))
            bound_violations += int(np.count_nonzero((halo_array < 0) | (halo_array > side_int)))
    return {
        "cases": len(cases),
        "observed_sides": sorted(observed_sides),
        "halo_min": halo_min,
        "halo_max": halo_max,
        "bound_violations": bound_violations,
        "errors": errors,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    guards_before = {name: sha256(ROOT / name) for name in base.ROOT_GUARDS}
    if guards_before != base.ROOT_GUARDS:
        raise RuntimeError(f"protected root guard mismatch before audit: {guards_before}")
    expected = {
        AUTHORITY: EXPECTED_AUTHORITY_SHA,
        INTERMEDIATE: EXPECTED_INTERMEDIATE_SHA,
        CANDIDATE: EXPECTED_CANDIDATE_SHA,
        GENERATOR: EXPECTED_GENERATOR_SHA,
        COMMON: EXPECTED_COMMON_SHA,
    }
    actual = {str(path): sha256(path) for path in expected}
    if any(actual[str(path)] != digest for path, digest in expected.items()):
        raise RuntimeError(f"input SHA mismatch: {actual}")

    prior = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    prior_exact_pass = bool(
        prior.get("pass")
        and prior.get("candidate", {}).get("sha256") == EXPECTED_INTERMEDIATE_SHA
        and prior.get("all_input_pass_through_exact_proof")
        and prior.get("all_raw_bitwise_equivalent")
    )
    generator = generator_ast_proof()
    diff = mechanical_diff()
    base.CANDIDATE = CANDIDATE
    base.EXPECTED_CANDIDATE_SHA = EXPECTED_CANDIDATE_SHA
    profiles = {"authority": base.profile(AUTHORITY), "intermediate": base.profile(INTERMEDIATE), "candidate": base.profile(CANDIDATE)}
    static = {"authority": base.structure(AUTHORITY), "candidate": base.structure(CANDIDATE)}
    shape_truth = base.runtime_shape_truth()

    known = base.known_cases()
    known_results = {}
    for disable, threads, label in base.CONFIGS:
        known_results[label] = base.evaluate(known, disable, threads)

    fresh_results = []
    support_traces = []
    for seed, count in FRESH:
        cases, attempts = base.fresh_cases(seed, count)
        stream = {"seed": seed, "requested": count, "generated": len(cases), "attempts": attempts, "configs": {}}
        for disable, threads, label in base.CONFIGS:
            stream["configs"][label] = base.evaluate(cases, disable, threads)
        fresh_results.append(stream)
        support_traces.append({"seed": seed, **support_trace(cases)})
        print(f"fresh seed={seed} complete cases={len(cases)}", flush=True)

    comparisons = list(known_results.values()) + [
        row for stream in fresh_results for row in stream["configs"].values()
    ]
    mechanical_pass = all((
        diff["source_only_initializers"] == ["max30_i8"],
        diff["candidate_only_initializers"] == [],
        diff["max30_i8_exact"],
        diff["max29_i8_exact"],
        all(diff["exact_node_rewrites"].values()),
        diff["all_common_initializers_byte_identical"],
        diff["all_other_nodes_byte_identical_and_ordered"],
        diff["all_other_model_fields_byte_identical"],
        diff["halo_end_clip_is_zero_to_side"],
        diff["side_integer_path_exact"],
        diff["comparison_inputs_are_int8"],
        diff["equal30_iff_greater29_for_every_int8_x_le_30"],
    ))
    support_pass = bool(
        generator["pass"]
        and generator["maximum_side"] == 30
        and all(
            trace["errors"] == 0
            and trace["bound_violations"] == 0
            and set(trace["observed_sides"]).issubset(set(generator["supported_square_sides"]))
            and trace["halo_max"] <= 30
            for trace in support_traces
        )
    )
    structure_pass = all((
        static["candidate"]["full_check"], static["candidate"]["strict"],
        static["candidate"]["strict_data_prop"], static["candidate"]["standard_domain_only"],
        static["candidate"]["functions"] == 0, static["candidate"]["sparse_initializers"] == 0,
        static["candidate"]["nested_graphs"] == 0, static["candidate"]["conv_family_nodes"] == 0,
        static["candidate"]["nonfinite_initializers"] == 0,
    ))
    guards_after = {name: sha256(ROOT / name) for name in base.ROOT_GUARDS}
    report: dict[str, Any] = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": sha256(AUTHORITY)},
        "intermediate": {"path": str(INTERMEDIATE.relative_to(ROOT)), "sha256": sha256(INTERMEDIATE)},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(CANDIDATE)},
        "profiles": profiles,
        "strict_lower_than_authority": profiles["candidate"]["cost"] < profiles["authority"]["cost"],
        "cost_delta_from_authority": profiles["authority"]["cost"] - profiles["candidate"]["cost"],
        "cost_delta_from_intermediate": profiles["intermediate"]["cost"] - profiles["candidate"]["cost"],
        "projected_log_gain": math.log(profiles["authority"]["cost"] / profiles["candidate"]["cost"]),
        "prior_affine_all_input_exact_audit_pass": prior_exact_pass,
        "generator_ast_proof": generator,
        "max29_mechanical_diff": diff,
        "mechanical_pass": mechanical_pass,
        "generator_support_pass": support_pass,
        "static": static,
        "runtime_shape_truth": shape_truth,
        "known_count": len(known),
        "known_four_configs": known_results,
        "fresh": fresh_results,
        "support_traces": support_traces,
        "all_raw_bitwise_equivalent": all(row["exact_equivalent"] for row in comparisons),
        "runtime_errors_total": sum(row["runtime_errors_total"] for row in comparisons),
        "nonfinite_values_total": sum(row["nonfinite_values_total"] for row in comparisons),
        "minimum_fresh_candidate_accuracy": min(
            row["accuracy"]["candidate"] for stream in fresh_results for row in stream["configs"].values()
        ),
        "root_guards_before": guards_before,
        "root_guards_after": guards_after,
    }
    report["pass"] = bool(
        report["strict_lower_than_authority"]
        and report["cost_delta_from_authority"] == 8
        and report["cost_delta_from_intermediate"] == 1
        and prior_exact_pass
        and mechanical_pass
        and support_pass
        and structure_pass
        and shape_truth["truthful"]
        and report["all_raw_bitwise_equivalent"]
        and report["runtime_errors_total"] == 0
        and report["nonfinite_values_total"] == 0
        and guards_before == guards_after == base.ROOT_GUARDS
    )
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "costs": profiles,
        "known": len(known),
        "fresh": [stream["generated"] for stream in fresh_results],
        "support_traces": support_traces,
        "minimum_fresh_candidate_accuracy": report["minimum_fresh_candidate_accuracy"],
        "runtime_errors_total": report["runtime_errors_total"],
        "nonfinite_values_total": report["nonfinite_values_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
