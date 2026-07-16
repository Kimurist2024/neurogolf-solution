#!/usr/bin/env python3
"""Independent fail-closed audit of the task349 affine hstart rewrite."""

from __future__ import annotations

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
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task349.onnx"
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_task349_affine_203/task349_affine_no_scalar.onnx"
RESULT = HERE / "audit.json"

EXPECTED_AUTHORITY_SHA = "179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7"
EXPECTED_CANDIDATE_SHA = "849d49e462ca94b5e4f9120434a39e1982d9dce521863e80b29d80ad9b02406b"
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
FRESH = ((204_349_17, 2500), (204_349_31, 2500))

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, total = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(total)}


def attr_nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                count += 1
                pending.extend(attr.g.node)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                count += len(attr.graphs)
                for graph in attr.graphs:
                    pending.extend(graph.node)
    return count


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    row: dict[str, Any] = {
        "sha256": sha256(path),
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": int(sum(np.asarray(numpy_helper.to_array(x)).size for x in model.graph.initializer)),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": attr_nested_graph_count(model),
        "conv_family_nodes": sum(n.op_type in {"Conv", "ConvTranspose", "QLinearConv"} for n in model.graph.node),
        "nonfinite_initializers": int(sum(
            np.asarray(numpy_helper.to_array(x)).size - np.count_nonzero(np.isfinite(numpy_helper.to_array(x)))
            for x in model.graph.initializer
            if np.asarray(numpy_helper.to_array(x)).dtype.kind in "fc"
        )),
    }
    row["standard_domain_only"] = all(item.domain in {"", "ai.onnx"} for item in model.opset_import)
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


def mechanical_diff() -> dict[str, Any]:
    source = onnx.load(AUTHORITY)
    candidate = onnx.load(CANDIDATE)
    src_init = {x.name: x for x in source.graph.initializer}
    cand_init = {x.name: x for x in candidate.graph.initializer}
    src_arr = {name: np.asarray(numpy_helper.to_array(value)) for name, value in src_init.items()}
    cand_arr = {name: np.asarray(numpy_helper.to_array(value)) for name, value in cand_init.items()}

    hstart = src_arr["hstart_offset_by_mod_i8"].astype(np.int16)
    radius = src_arr["hend_offset_by_mod_i8"].astype(np.int16)
    twice = np.int16(2) * radius
    top = np.int16(1) - twice
    derived = top - radius

    removed_hstart_nodes = [
        n for n in source.graph.node
        if n.op_type == "Gather" and list(n.input) == ["hstart_offset_by_mod_i8", "radius_code"]
        and list(n.output) == ["hstart_offset_i8"]
    ]
    removed_top_nodes = [
        n for n in source.graph.node
        if n.op_type == "Add" and list(n.input) == ["hstart_offset_i8", "hend_offset_i8"]
        and list(n.output) == ["top_offset_i8"]
    ]
    removed_nodes = removed_hstart_nodes + removed_top_nodes
    added_nodes = [
        n for n in candidate.graph.node
        if n.name in {"radius_twice_for_offsets", "top_offset_from_radius", "hstart_offset_from_radius"}
    ]
    src_kept_nodes = [n.SerializeToString() for n in source.graph.node if n not in removed_nodes]
    cand_kept_nodes = [n.SerializeToString() for n in candidate.graph.node if n not in added_nodes]

    src_skeleton = copy.deepcopy(source)
    cand_skeleton = copy.deepcopy(candidate)
    del src_skeleton.graph.node[:]
    del src_skeleton.graph.initializer[:]
    del cand_skeleton.graph.node[:]
    del cand_skeleton.graph.initializer[:]
    common_names = sorted(set(src_init) & set(cand_init))
    common_initializers_identical = all(
        src_init[name].SerializeToString() == cand_init[name].SerializeToString()
        for name in common_names
    )
    expected_added = {
        "radius_twice_for_offsets": ("Add", ["hend_offset_i8", "hend_offset_i8"], ["radius_twice_i8"]),
        "top_offset_from_radius": ("Sub", ["one_i8", "radius_twice_i8"], ["top_offset_i8"]),
        "hstart_offset_from_radius": ("Sub", ["top_offset_i8", "hend_offset_i8"], ["hstart_offset_i8"]),
    }
    added_exact = len(added_nodes) == 3 and all(
        (node.op_type, list(node.input), list(node.output)) == expected_added[node.name]
        for node in added_nodes
    )
    return {
        "hstart_table_int8": src_arr["hstart_offset_by_mod_i8"].astype(np.int8).tolist(),
        "radius_table_int8": src_arr["hend_offset_by_mod_i8"].astype(np.int8).tolist(),
        "twice_radius_int16": twice.tolist(),
        "derived_top_int16": top.tolist(),
        "derived_hstart_int16": derived.tolist(),
        "identity_all_11": bool(hstart.size == 11 and radius.size == 11 and np.array_equal(hstart, derived)),
        "source_top_identity_all_11": bool(np.array_equal(hstart + radius, top)),
        "intermediate_range": {
            "twice_min": int(twice.min()), "twice_max": int(twice.max()),
            "top_min": int(top.min()), "top_max": int(top.max()),
            "hstart_min": int(derived.min()), "hstart_max": int(derived.max()),
        },
        "int8_overflow_absent": bool(
            twice.min() >= -128 and twice.max() <= 127
            and top.min() >= -128 and top.max() <= 127
            and derived.min() >= -128 and derived.max() <= 127
        ),
        "source_hstart_gather_removed_exactly_once": len(removed_hstart_nodes) == 1,
        "source_top_add_removed_exactly_once": len(removed_top_nodes) == 1,
        "candidate_affine_nodes_added_exactly": added_exact,
        "source_only_initializers": sorted(set(src_init) - set(cand_init)),
        "candidate_only_initializers": sorted(set(cand_init) - set(src_init)),
        "existing_one_i8_exact": bool(
            "one_i8" in src_arr
            and src_arr["one_i8"].dtype == np.dtype(np.int8)
            and src_arr["one_i8"].shape == ()
            and int(src_arr["one_i8"]) == 1
        ),
        "all_common_initializers_byte_identical": common_initializers_identical,
        "all_other_nodes_byte_identical_and_ordered": src_kept_nodes == cand_kept_nodes,
        "all_other_model_fields_byte_identical": src_skeleton.SerializeToString() == cand_skeleton.SerializeToString(),
    }


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
        raise RuntimeError(f"fresh generation shortfall: {len(rows)}/{count} after {attempts}")
    return rows, attempts


def session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
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
        "authority": session(AUTHORITY.read_bytes(), disable, threads),
        "candidate": session(CANDIDATE.read_bytes(), disable, threads),
    }
    row: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "conversion_skips": 0,
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
            row["conversion_skips"] += 1
            continue
        row["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, current in sessions.items():
            try:
                output = np.asarray(current.run(None, {current.get_inputs()[0].name: benchmark["input"]})[0])
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"][label] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index, "model": label, "error": f"{type(exc).__name__}: {exc}"
                }
                continue
            outputs[label] = output
            if output.dtype.kind in "fc":
                row["nonfinite_values"][label] += int(output.size - np.count_nonzero(np.isfinite(output)))
            row["right"][label] += int(output.shape == expected.shape and np.array_equal(output > 0, expected))
        if len(outputs) == 2:
            authority = np.ascontiguousarray(outputs["authority"])
            candidate = np.ascontiguousarray(outputs["candidate"])
            raw_equal = bool(
                authority.dtype == candidate.dtype
                and authority.shape == candidate.shape
                and authority.tobytes() == candidate.tobytes()
            )
            threshold_equal = bool(np.array_equal(authority > 0, candidate > 0))
            row["raw_equal"] += int(raw_equal)
            row["threshold_equal"] += int(threshold_equal)
            if not raw_equal and row["first_failure"] is None:
                delta = np.abs(authority.astype(np.float64) - candidate.astype(np.float64))
                row["first_failure"] = {"index": index, "max_abs_delta": float(np.nanmax(delta))}
    valid = row["valid"]
    row["accuracy"] = {
        label: row["right"][label] / valid if valid else None for label in ("authority", "candidate")
    }
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())
    row["exact_equivalent"] = bool(
        valid == len(cases)
        and row["raw_equal"] == valid
        and row["threshold_equal"] == valid
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
    outputs = {value.name for value in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in outputs:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    outputs.add(name)
    benchmark = scoring.convert_to_numpy(known_cases()[0])
    if benchmark is None:
        return {"truthful": False, "error": "known case conversion failed"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    current = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    arrays = current.run(names, {current.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = 0
    for name, array in zip(names, arrays):
        value = np.asarray(array)
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
    guards_before = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    if guards_before != ROOT_GUARDS:
        raise RuntimeError(f"protected root guard mismatch before audit: {guards_before}")
    if sha256(AUTHORITY) != EXPECTED_AUTHORITY_SHA or sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("authority/candidate SHA mismatch")

    authority_profile = profile(AUTHORITY)
    candidate_profile = profile(CANDIDATE)
    static = {"authority": structure(AUTHORITY), "candidate": structure(CANDIDATE)}
    diff = mechanical_diff()
    known = known_cases()
    known_results = {}
    for disable, threads, label in CONFIGS:
        known_results[label] = evaluate(known, disable, threads)

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
    guards_after = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    identity_proof = all((
        diff["identity_all_11"],
        diff["source_top_identity_all_11"],
        diff["int8_overflow_absent"],
        diff["source_hstart_gather_removed_exactly_once"],
        diff["source_top_add_removed_exactly_once"],
        diff["candidate_affine_nodes_added_exactly"],
        diff["source_only_initializers"] == ["hstart_offset_by_mod_i8"],
        diff["candidate_only_initializers"] == [],
        diff["existing_one_i8_exact"],
        diff["all_common_initializers_byte_identical"],
        diff["all_other_nodes_byte_identical_and_ordered"],
        diff["all_other_model_fields_byte_identical"],
    ))
    structure_pass = all((
        static["candidate"]["full_check"],
        static["candidate"]["strict"],
        static["candidate"]["strict_data_prop"],
        static["candidate"]["standard_domain_only"],
        static["candidate"]["functions"] == 0,
        static["candidate"]["sparse_initializers"] == 0,
        static["candidate"]["nested_graphs"] == 0,
        static["candidate"]["conv_family_nodes"] == 0,
        static["candidate"]["nonfinite_initializers"] == 0,
    ))
    shape_truth = runtime_shape_truth()
    report: dict[str, Any] = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": sha256(AUTHORITY), "profile": authority_profile},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(CANDIDATE), "profile": candidate_profile},
        "strict_lower": candidate_profile["cost"] < authority_profile["cost"],
        "cost_delta": authority_profile["cost"] - candidate_profile["cost"],
        "projected_log_gain": math.log(authority_profile["cost"] / candidate_profile["cost"]),
        "static": static,
        "mechanical_diff": diff,
        "all_input_pass_through_exact_proof": identity_proof,
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
        "root_guards_before": guards_before,
        "root_guards_after": guards_after,
    }
    report["pass"] = bool(
        report["strict_lower"]
        and report["cost_delta"] == 7
        and structure_pass
        and shape_truth["truthful"]
        and identity_proof
        and report["all_raw_bitwise_equivalent"]
        and report["runtime_errors_total"] == 0
        and report["nonfinite_values_total"] == 0
        and guards_before == guards_after == ROOT_GUARDS
    )
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "cost": [authority_profile["cost"], candidate_profile["cost"]],
        "known": len(known),
        "fresh": [stream["generated"] for stream in fresh_results],
        "minimum_fresh_candidate_accuracy": report["minimum_fresh_candidate_accuracy"],
        "runtime_errors_total": report["runtime_errors_total"],
        "nonfinite_values_total": report["nonfinite_values_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
