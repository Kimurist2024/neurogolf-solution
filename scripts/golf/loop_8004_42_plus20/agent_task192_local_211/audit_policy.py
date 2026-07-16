#!/usr/bin/env python3
"""Fail-closed audit of task192's POLICY90-preserving center-basis shave."""

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
SOURCE = ROOT / "others/71407/task192.onnx"
CANDIDATE = HERE / "candidates/task192_policy90_center_direct.onnx"
EXACT_CONTROL = HERE / "candidates/task192_center_direct_argmax.onnx"
RESULT = HERE / "audit_policy.json"
EXPECTED_SOURCE_SHA = "e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5"
EXPECTED_CANDIDATE_SHA = "1200fe8473c045ec89abaaf1860d1d0758316523855c9ff13d4c3fc092412047"
EXPECTED_EXACT_SHA = "5c5eaefa81acce481dbc93855dbcc2f9ef821e055f8c982eadcd07f63c764a9d"
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
FRESH = ((211_192_71, 2500), (211_192_89, 2500))
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress", "Hardmax"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def attr_float(node: onnx.NodeProto, name: str) -> float | None:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return None


def nested_graphs(model: onnx.ModelProto) -> int:
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


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    names = {x.name for x in model.graph.initializer}
    used = {name for node in model.graph.node for name in node.input if name in names}
    row: dict[str, Any] = {
        "sha256": sha256(path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": int(sum(np.asarray(numpy_helper.to_array(x)).size for x in model.graph.initializer)),
        "opsets": [{"domain": x.domain, "version": int(x.version)} for x in model.opset_import],
        "standard_domain_only": all(x.domain in {"", "ai.onnx"} for x in model.opset_import),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested_graphs(model),
        "conv_family": sum(x.op_type in {"Conv", "ConvTranspose", "QLinearConv"} for x in model.graph.node),
        "banned": sorted({x.op_type for x in model.graph.node if x.op_type in BANNED or "Sequence" in x.op_type}),
        "unused_initializers": sorted(names - used),
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


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}


def algebraic_proof() -> dict[str, Any]:
    source = onnx.load(SOURCE)
    candidate = onnx.load(CANDIDATE)
    sa, ca = arrays(source), arrays(candidate)
    source_nodes = {x.output[0]: x for x in source.graph.node}
    candidate_nodes = {x.output[0]: x for x in candidate.graph.node}

    old_fixed = np.concatenate([sa["nonzero"], sa["background"]], axis=0)
    selected_symbols = np.eye(10, dtype=np.float32)
    per_selected = []
    for selected_index in range(10):
        selected = selected_symbols[selected_index:selected_index + 1]
        old_basis = np.concatenate([sa["nonzero"], sa["background"], selected], axis=0)
        new_basis = np.concatenate([ca["center_basis"], selected], axis=0)
        old_center = sa["center_map"] @ old_basis
        new_center = ca["center_basis"]
        old_neighbor = sa["neighbor_map"] @ old_basis
        new_neighbor = ca["neighbor_map"] @ new_basis
        old_route = sa["route_out"] @ old_basis
        new_route = ca["route_out"] @ new_basis
        per_selected.append({
            "selected": selected_index,
            "center_equal": bool(np.array_equal(old_center, new_center)),
            "neighbor_equal": bool(np.array_equal(old_neighbor, new_neighbor)),
            "route_equal": bool(np.array_equal(old_route, new_route)),
        })

    hist_exact = bool(
        np.array_equal(ca["center_basis"][1:2], sa["nonzero"])
        and np.array_equal(ca["hist_selector"], np.asarray([[0.0, 1.0]], dtype=np.float32))
    )
    hard_sigmoid_exact = bool(
        source_nodes["selected"].op_type == candidate_nodes["selected"].op_type == "HardSigmoid"
        and attr_float(source_nodes["selected"], "alpha") == attr_float(candidate_nodes["selected"], "alpha") == 1.0
        and attr_float(source_nodes["selected"], "beta") == attr_float(candidate_nodes["selected"], "beta") == -33.0
    )
    adjacency_exact = bool(
        next(x for x in source.graph.initializer if x.name == "adj").SerializeToString()
        == next(x for x in candidate.graph.initializer if x.name == "adj").SerializeToString()
    )
    proof = {
        "source_fixed_basis_rows": old_fixed.tolist(),
        "candidate_center_basis_rows": ca["center_basis"].tolist(),
        "histogram_nonzero_row_exact": hist_exact,
        "hard_sigmoid_alpha_beta_unchanged": hard_sigmoid_exact,
        "adjacency_byte_identical": adjacency_exact,
        "all_ten_selected_vectors": per_selected,
        "argument": (
            "For basis [inside,nonzero,selected], the center is directly [inside,nonzero]; "
            "neighbor_map recovers [inside,selected]; route_out recovers "
            "[inside-nonzero, -9*(inside-nonzero)+selected] = "
            "[background,-9*background+selected]. The selected HardSigmoid is unchanged."
        ),
    }
    proof["pass"] = bool(
        hist_exact and hard_sigmoid_exact and adjacency_exact
        and all(x["center_equal"] and x["neighbor_equal"] and x["route_equal"] for x in per_selected)
    )
    return proof


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


def runtime_shape_truth() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        x.name: x for x in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {x.name for x in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    example = known_cases()[0]
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        return {"truthful": False, "error": "conversion failed"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    values = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = 0
    for name, value in zip(names, values):
        array = np.asarray(value)
        if dims(typed[name]) != list(array.shape):
            mismatches.append({"name": name, "declared": dims(typed[name]), "actual": list(array.shape)})
        if array.dtype.kind in "fc":
            nonfinite += int(array.size - np.count_nonzero(np.isfinite(array)))
    return {
        "traced": len(names), "mismatches": mismatches, "mismatch_count": len(mismatches),
        "nonfinite_values": nonfinite, "truthful": not mismatches and nonfinite == 0,
    }


def known_cases() -> list[dict[str, Any]]:
    payload = scoring.load_examples(192)
    return [row for split in ("train", "test", "arc-gen") for row in payload.get(split, [])]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_7e0986d6")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    rows = []
    while len(rows) < count:
        row = generator.generate()
        if scoring.convert_to_numpy(row) is not None:
            rows.append(row)
    return rows


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
    sessions = {"source": make_session(SOURCE, disable, threads), "candidate": make_session(CANDIDATE, disable, threads)}
    row: dict[str, Any] = {
        "total": len(cases), "valid": 0, "raw_equal": 0, "sign_equal": 0,
        "right": {"source": 0, "candidate": 0},
        "errors": {"source": 0, "candidate": 0},
        "nonfinite": {"source": 0, "candidate": 0}, "first_failure": None,
    }
    for index, case in enumerate(cases):
        benchmark = scoring.convert_to_numpy(case)
        if benchmark is None:
            row["first_failure"] = row["first_failure"] or {"index": index, "error": "conversion"}
            continue
        row["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                output = np.asarray(session.run(None, {session.get_inputs()[0].name: benchmark["input"]})[0])
                outputs[label] = output
                row["right"][label] += int(output.shape == expected.shape and np.array_equal(output > 0, expected))
                if output.dtype.kind in "fc":
                    row["nonfinite"][label] += int(output.size - np.count_nonzero(np.isfinite(output)))
            except Exception as exc:  # noqa: BLE001
                row["errors"][label] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index, "model": label, "error": f"{type(exc).__name__}: {exc}"
                }
        if len(outputs) == 2:
            old = np.ascontiguousarray(outputs["source"])
            new = np.ascontiguousarray(outputs["candidate"])
            equal = old.dtype == new.dtype and old.shape == new.shape and old.tobytes() == new.tobytes()
            row["raw_equal"] += int(equal)
            row["sign_equal"] += int(np.array_equal(old > 0, new > 0))
            if not equal and row["first_failure"] is None:
                row["first_failure"] = {"index": index, "max_abs": float(np.max(np.abs(old - new)))}
    row["errors_total"] = sum(row["errors"].values())
    row["nonfinite_total"] = sum(row["nonfinite"].values())
    row["accuracy"] = {label: row["right"][label] / row["valid"] for label in ("source", "candidate")}
    row["pass_through"] = bool(
        row["valid"] == len(cases) and row["raw_equal"] == len(cases) and row["sign_equal"] == len(cases)
        and row["errors_total"] == 0 and row["nonfinite_total"] == 0
    )
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    before = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    if before != ROOT_GUARDS:
        raise RuntimeError(f"root guard mismatch: {before}")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA or sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("source/candidate SHA mismatch")
    if sha256(EXACT_CONTROL) != EXPECTED_EXACT_SHA:
        raise RuntimeError("exact control SHA mismatch")

    profiles = {"source": profile(SOURCE), "candidate": profile(CANDIDATE), "exact_control": profile(EXACT_CONTROL)}
    structures = {"source": structure(SOURCE), "candidate": structure(CANDIDATE), "exact_control": structure(EXACT_CONTROL)}
    proof = algebraic_proof()
    shape_truth = runtime_shape_truth()
    known = known_cases()
    known_results = {label: evaluate(known, disable, threads) for disable, threads, label in CONFIGS}
    fresh = []
    for seed, count in FRESH:
        cases = fresh_cases(seed, count)
        stream = {"seed": seed, "count": len(cases), "configs": {}}
        for disable, threads, label in CONFIGS:
            stream["configs"][label] = evaluate(cases, disable, threads)
        fresh.append(stream)
        print(f"fresh seed={seed} complete count={len(cases)}", flush=True)
    comparisons = list(known_results.values()) + [row for stream in fresh for row in stream["configs"].values()]
    candidate_static = structures["candidate"]
    static_pass = bool(
        candidate_static["full_check"] and candidate_static["strict"] and candidate_static["strict_data_prop"]
        and candidate_static["standard_domain_only"] and candidate_static["functions"] == 0
        and candidate_static["sparse_initializers"] == 0 and candidate_static["nested_graphs"] == 0
        and candidate_static["conv_family"] == 0 and not candidate_static["banned"]
        and not candidate_static["unused_initializers"] and candidate_static["nonfinite_initializers"] == 0
    )
    after = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    report: dict[str, Any] = {
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE), "profile": profiles["source"]},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(CANDIDATE), "profile": profiles["candidate"]},
        "exact_control": {"path": str(EXACT_CONTROL.relative_to(ROOT)), "sha256": sha256(EXACT_CONTROL), "profile": profiles["exact_control"], "decision": "REJECT_NOT_BELOW_1138"},
        "classification": "POLICY90_INHERITED_RAW_PASS_THROUGH_NOT_ALL_SUPPORT_EXACT",
        "strict_lower": profiles["candidate"]["cost"] < profiles["source"]["cost"],
        "cost_delta": profiles["source"]["cost"] - profiles["candidate"]["cost"],
        "projected_gain": math.log(profiles["source"]["cost"] / profiles["candidate"]["cost"]),
        "algebraic_proof": proof,
        "structure": structures,
        "static_pass": static_pass,
        "runtime_shape_truth": shape_truth,
        "known_count": len(known),
        "known_four_configs": known_results,
        "fresh": fresh,
        "all_raw_pass_through": all(x["pass_through"] for x in comparisons),
        "minimum_fresh_accuracy": min(x["accuracy"]["candidate"] for stream in fresh for x in stream["configs"].values()),
        "errors_total": sum(x["errors_total"] for x in comparisons),
        "nonfinite_total": sum(x["nonfinite_total"] for x in comparisons),
        "root_guards_before": before,
        "root_guards_after": after,
    }
    report["pass"] = bool(
        report["strict_lower"] and report["cost_delta"] == 4
        and profiles["source"] == {"memory": 200, "params": 938, "cost": 1138}
        and profiles["candidate"] == {"memory": 200, "params": 934, "cost": 1134}
        and proof["pass"] and static_pass and shape_truth["truthful"]
        and report["all_raw_pass_through"] and report["minimum_fresh_accuracy"] >= 0.90
        and report["errors_total"] == 0 and report["nonfinite_total"] == 0
        and before == after == ROOT_GUARDS
    )
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"], "cost": [profiles["source"]["cost"], profiles["candidate"]["cost"]],
        "exact_control_cost": profiles["exact_control"]["cost"], "known": len(known),
        "fresh": [x["count"] for x in fresh], "minimum_fresh_accuracy": report["minimum_fresh_accuracy"],
        "all_raw_pass_through": report["all_raw_pass_through"], "errors_total": report["errors_total"],
        "nonfinite_total": report["nonfinite_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
