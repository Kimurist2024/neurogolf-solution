#!/usr/bin/env python3
"""Independent fail-closed audit of the task066 cost-561 Selu candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank/task066.onnx")
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_task066_selu_200/task066_selu_cost561.onnx"
EXPECTED_AUTHORITY_SHA = "bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e"
EXPECTED_CANDIDATE_SHA = "2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((66_206_001, 2000), (66_206_002, 2000))
TRACE = ("ti", "selF", "selLog", "selQ")

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


def model_bytes() -> tuple[bytes, bytes]:
    authority = AUTHORITY.read_bytes()
    candidate = CANDIDATE.read_bytes()
    assert sha256(authority) == EXPECTED_AUTHORITY_SHA
    assert sha256(candidate) == EXPECTED_CANDIDATE_SHA
    return authority, candidate


def dim_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def static_audit(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_error"] = f"{type(exc).__name__}: {exc}"

    inferred_models = {}
    for label, data_prop in (("strict", False), ("strict_data_prop", True)):
        try:
            inferred_models[label] = shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=data_prop
            )
            result[label] = True
        except Exception as exc:  # noqa: BLE001
            result[label] = False
            result[f"{label}_error"] = f"{type(exc).__name__}: {exc}"

    inferred = inferred_models.get("strict_data_prop")
    unresolved = []
    if inferred is not None:
        typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
        for node in inferred.graph.node:
            for name in node.output:
                if not name:
                    continue
                value = typed.get(name)
                if value is None or any(dim is None or dim <= 0 for dim in dim_shape(value)):
                    unresolved.append(name)

    excluded = set()
    for node in model.graph.node:
        upper = node.op_type.upper()
        if any(token in upper for token in scoring._EXCLUDED_OP_TYPES) or "SEQUENCE" in upper:
            excluded.add(node.op_type)
    nested = [
        f"{node.op_type}:{attr.name}"
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    result.update(
        {
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "params_direct": int(
                sum(np.asarray(numpy_helper.to_array(item)).size for item in model.graph.initializer)
            ),
            "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
            and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nested_graphs": nested,
            "banned_ops": sorted(excluded),
            "conv_bias_ub": check_conv_bias(model),
            "unresolved_or_dynamic_node_outputs": unresolved,
        }
    )
    result["pass"] = bool(
        result["full_check"]
        and result["strict"]
        and result["strict_data_prop"]
        and result["standard_domains"]
        and not result["functions"]
        and not result["sparse_initializers"]
        and not result["nested_graphs"]
        and not result["banned_ops"]
        and not result["conv_bias_ub"]
        and not result["unresolved_or_dynamic_node_outputs"]
    )
    return result


def graph_delta(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    changed_nodes = []
    for index, (left, right) in enumerate(zip(authority.graph.node, candidate.graph.node, strict=True)):
        if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True):
            changed_nodes.append(
                {
                    "index": index,
                    "authority": {
                        "op": left.op_type,
                        "inputs": list(left.input),
                        "outputs": list(left.output),
                    },
                    "candidate": {
                        "op": right.op_type,
                        "inputs": list(right.input),
                        "outputs": list(right.output),
                        "attributes": {
                            attr.name: helper.get_attribute_value(attr) for attr in right.attribute
                        },
                    },
                }
            )
    left_init = {item.name: item for item in authority.graph.initializer}
    right_init = {item.name: item for item in candidate.graph.initializer}
    common_init_equal = all(
        left_init[name].SerializeToString(deterministic=True)
        == right_init[name].SerializeToString(deterministic=True)
        for name in set(left_init) & set(right_init)
    )
    # Check all top-level/graph fields other than node and initializer payloads.
    left_shell = copy.deepcopy(authority)
    right_shell = copy.deepcopy(candidate)
    del left_shell.graph.node[:]
    del right_shell.graph.node[:]
    del left_shell.graph.initializer[:]
    del right_shell.graph.initializer[:]
    shells_equal = left_shell.SerializeToString(deterministic=True) == right_shell.SerializeToString(deterministic=True)

    selq_consumers = [
        {"index": i, "op": node.op_type, "output": list(node.output)}
        for i, node in enumerate(candidate.graph.node)
        if "selQ" in node.input
    ]
    exact = bool(
        len(authority.graph.node) == len(candidate.graph.node) == 77
        and len(changed_nodes) == 1
        and changed_nodes[0]["index"] == 64
        and changed_nodes[0]["authority"]
        == {"op": "Div", "inputs": ["selLog", "ln2"], "outputs": ["selQ"]}
        and changed_nodes[0]["candidate"]["op"] == "Selu"
        and changed_nodes[0]["candidate"]["inputs"] == ["selLog"]
        and changed_nodes[0]["candidate"]["outputs"] == ["selQ"]
        and set(left_init) - set(right_init) == {"ln2"}
        and not (set(right_init) - set(left_init))
        and common_init_equal
        and shells_equal
        and selq_consumers == [{"index": 65, "op": "Cast", "output": ["ti"]}]
    )
    return {
        "changed_nodes": changed_nodes,
        "removed_initializers": sorted(set(left_init) - set(right_init)),
        "added_initializers": sorted(set(right_init) - set(left_init)),
        "common_initializers_proto_equal": common_init_equal,
        "all_other_model_fields_equal": shells_equal,
        "selQ_consumers": selq_consumers,
        "exact_whitelist_delta": exact,
    }


def official_profile(data: bytes, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task066_review206_", dir="/tmp") as work:
        row = scoring.score_and_verify(
            onnx.load_model_from_string(data), 66, work, label=label, require_correct=True
        )
    assert row is not None
    return {
        "memory": int(row["memory"]),
        "params": int(row["params"]),
        "cost": int(row["cost"]),
        "correct": bool(row["correct"]),
    }


def traced_model(data: bytes, names: tuple[str, ...]) -> bytes:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
    existing = {value.name for value in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(copy.deepcopy(typed[name]))
            existing.add(name)
    return model.SerializeToString()


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def known_cases() -> list[dict[str, Any]]:
    examples = scoring.load_examples(66)
    return [row for split in ("train", "test", "arc-gen") for row in examples[split]]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_2dd70a9a")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [generator.generate() for _ in range(count)]


def evaluate_pair(
    authority: ort.InferenceSession,
    candidate: ort.InferenceSession,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "authority_gold": 0,
        "candidate_gold": 0,
        "final_raw_equal": 0,
        "final_threshold_equal": 0,
        "ti_raw_equal": 0,
        "selQ_raw_equal": 0,
        "runtime_errors": 0,
        "authority_final_nonfinite": 0,
        "candidate_final_nonfinite": 0,
        "authority_intermediate_nonfinite": {name: 0 for name in ("selF", "selLog", "selQ")},
        "candidate_intermediate_nonfinite": {name: 0 for name in ("selF", "selLog", "selQ")},
        "selF_min_finite": math.inf,
        "selF_zero_negative_or_nan": 0,
        "first_difference": None,
    }
    for index, example in enumerate(cases):
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            continue
        result["valid"] += 1
        outputs = {}
        for label, session in (("authority", authority), ("candidate", candidate)):
            try:
                outputs[label] = [
                    np.asarray(item)
                    for item in session.run(None, {session.get_inputs()[0].name: converted["input"]})
                ]
            except Exception as exc:  # noqa: BLE001
                result["runtime_errors"] += 1
                result["first_difference"] = result["first_difference"] or {
                    "case": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) != 2:
            continue
        # Output order is graph output followed by TRACE=(ti,selF,selLog,selQ).
        a_final, a_ti, a_self, a_log, a_q = outputs["authority"]
        c_final, c_ti, c_self, c_log, c_q = outputs["candidate"]
        expected = converted["output"].astype(bool)
        result["authority_gold"] += int(np.array_equal(a_final > 0.0, expected))
        result["candidate_gold"] += int(np.array_equal(c_final > 0.0, expected))
        raw_equal = a_final.dtype == c_final.dtype and a_final.shape == c_final.shape and a_final.tobytes() == c_final.tobytes()
        ti_equal = a_ti.dtype == c_ti.dtype and a_ti.shape == c_ti.shape and a_ti.tobytes() == c_ti.tobytes()
        q_equal = a_q.dtype == c_q.dtype and a_q.shape == c_q.shape and a_q.tobytes() == c_q.tobytes()
        result["final_raw_equal"] += int(raw_equal)
        result["final_threshold_equal"] += int(np.array_equal(a_final > 0.0, c_final > 0.0))
        result["ti_raw_equal"] += int(ti_equal)
        result["selQ_raw_equal"] += int(q_equal)
        result["authority_final_nonfinite"] += int(a_final.size - np.count_nonzero(np.isfinite(a_final)))
        result["candidate_final_nonfinite"] += int(c_final.size - np.count_nonzero(np.isfinite(c_final)))
        for label, values in (
            ("authority", {"selF": a_self, "selLog": a_log, "selQ": a_q}),
            ("candidate", {"selF": c_self, "selLog": c_log, "selQ": c_q}),
        ):
            for name, value in values.items():
                result[f"{label}_intermediate_nonfinite"][name] += int(
                    value.size - np.count_nonzero(np.isfinite(value))
                )
        finite_self = c_self[np.isfinite(c_self)]
        if finite_self.size:
            result["selF_min_finite"] = min(result["selF_min_finite"], float(np.min(finite_self)))
        result["selF_zero_negative_or_nan"] += int(np.count_nonzero((c_self <= 0) | np.isnan(c_self)))
        if not (raw_equal and ti_equal):
            result["first_difference"] = result["first_difference"] or {
                "case": index,
                "raw_equal": raw_equal,
                "ti_equal": ti_equal,
            }
    result["pass_through"] = bool(
        result["valid"] == len(cases)
        and result["final_raw_equal"] == result["valid"]
        and result["final_threshold_equal"] == result["valid"]
        and result["ti_raw_equal"] == result["valid"]
        and result["runtime_errors"] == 0
        and result["authority_final_nonfinite"] == 0
        and result["candidate_final_nonfinite"] == 0
        and result["selF_zero_negative_or_nan"] == 0
    )
    return result


def operator_model(count: int) -> bytes:
    stored_ln2 = np.asarray(0.69287109375, dtype=np.float16)
    gamma = float(np.float32(1.0 / float(stored_ln2)))
    nodes = [
        helper.make_node("Cast", ["x"], ["f"], to=TensorProto.FLOAT16),
        helper.make_node("Log", ["f"], ["log"]),
        helper.make_node("Div", ["log", "ln2"], ["div"]),
        helper.make_node("Selu", ["log"], ["selu"], alpha=1.0, gamma=gamma),
        helper.make_node("Cast", ["div"], ["div_ti"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["selu"], ["selu_ti"], to=TensorProto.UINT8),
    ]
    outputs = [
        helper.make_tensor_value_info("div", TensorProto.FLOAT16, [count]),
        helper.make_tensor_value_info("selu", TensorProto.FLOAT16, [count]),
        helper.make_tensor_value_info("div_ti", TensorProto.UINT8, [count]),
        helper.make_tensor_value_info("selu_ti", TensorProto.UINT8, [count]),
    ]
    graph = helper.make_graph(
        nodes,
        "task066_review206_operator",
        [helper.make_tensor_value_info("x", TensorProto.UINT32, [count])],
        outputs,
        initializer=[numpy_helper.from_array(stored_ln2, "ln2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def exhaustive_carrier() -> dict[str, Any]:
    # Reachable selected masks are 1..2**20-1.  Include both 0 and 2**20 as
    # sentinels, making this a strict superset of the proven generator domain.
    values = np.arange((1 << 20) + 1, dtype=np.uint32)
    model = operator_model(values.size)
    modes = {}
    carrier_outputs = {}
    for disable, threads, label in CONFIGS:
        div, selu, div_ti, selu_ti = [
            np.asarray(item) for item in make_session(model, disable, threads).run(None, {"x": values})
        ]
        fp_diff = np.flatnonzero(div.view(np.uint16) != selu.view(np.uint16))
        carrier_diff = np.flatnonzero(div_ti != selu_ti)
        modes[label] = {
            "tested_inclusive": [0, 1 << 20],
            "count": int(values.size),
            "float16_difference_count": int(fp_diff.size),
            "first_float16_difference_inputs": [int(values[i]) for i in fp_diff[:12]],
            "uint8_carrier_difference_count": int(carrier_diff.size),
            "div_nonfinite": int(div.size - np.count_nonzero(np.isfinite(div))),
            "selu_nonfinite": int(selu.size - np.count_nonzero(np.isfinite(selu))),
        }
        carrier_outputs[label] = selu_ti
    cross_mode = all(
        np.array_equal(next(iter(carrier_outputs.values())), row) for row in carrier_outputs.values()
    )
    return {
        "modes": modes,
        "candidate_carrier_cross_mode_equal": cross_mode,
        "pass": bool(
            cross_mode and all(row["uint8_carrier_difference_count"] == 0 for row in modes.values())
        ),
    }


def runtime_shape_truth(data: bytes, disable: bool) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    case = scoring.convert_to_numpy(known_cases()[0])
    assert case is not None
    session = make_session(traced.SerializeToString(), disable, 1)
    outputs = session.run(names, {session.get_inputs()[0].name: case["input"]})
    mismatches = []
    for name, output in zip(names, outputs, strict=True):
        declared = typed[name]
        expected_shape = dim_shape(declared)
        expected_dtype = declared.type.tensor_type.elem_type
        actual = np.asarray(output)
        actual_dtype = helper.np_dtype_to_tensor_dtype(actual.dtype)
        if expected_shape != list(actual.shape) or expected_dtype != actual_dtype:
            mismatches.append(
                {
                    "name": name,
                    "declared_shape": expected_shape,
                    "runtime_shape": list(actual.shape),
                    "declared_dtype": int(expected_dtype),
                    "runtime_dtype": int(actual_dtype),
                }
            )
    return {
        "optimization": "disable_all" if disable else "default",
        "node_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    authority_data, candidate_data = model_bytes()
    authority_trace = traced_model(authority_data, TRACE)
    candidate_trace = traced_model(candidate_data, TRACE)

    known = known_cases()
    fresh = [(seed, fresh_cases(seed, count)) for seed, count in FRESH]
    print(f"known={len(known)} fresh={[len(rows) for _, rows in fresh]}", flush=True)

    evaluations: dict[str, Any] = {"known": {}, "fresh": []}
    eval_rows = []
    for disable, threads, label in CONFIGS:
        auth_session = make_session(authority_trace, disable, threads)
        cand_session = make_session(candidate_trace, disable, threads)
        row = evaluate_pair(auth_session, cand_session, known)
        evaluations["known"][label] = row
        eval_rows.append(row)
        print(
            f"known {label}: gold={row['candidate_gold']}/{row['valid']} "
            f"raw={row['final_raw_equal']} ti={row['ti_raw_equal']}",
            flush=True,
        )
        for seed, cases in fresh:
            while len(evaluations["fresh"]) < len(fresh):
                next_seed, next_cases = fresh[len(evaluations["fresh"])]
                evaluations["fresh"].append({"seed": next_seed, "count": len(next_cases), "modes": {}})
            stream_index = next(i for i, item in enumerate(fresh) if item[0] == seed)
            fresh_row = evaluate_pair(auth_session, cand_session, cases)
            evaluations["fresh"][stream_index]["modes"][label] = fresh_row
            eval_rows.append(fresh_row)
            print(
                f"fresh {seed} {label}: gold={fresh_row['candidate_gold']}/{fresh_row['valid']} "
                f"raw={fresh_row['final_raw_equal']} ti={fresh_row['ti_raw_equal']}",
                flush=True,
            )

    delta = graph_delta(authority_data, candidate_data)
    static = {"authority": static_audit(authority_data), "candidate": static_audit(candidate_data)}
    profiles = {
        "authority": official_profile(authority_data, "authority_review206"),
        "candidate": official_profile(candidate_data, "candidate_review206"),
    }
    carrier = exhaustive_carrier()
    shapes = [runtime_shape_truth(candidate_data, disable) for disable in (True, False)]
    summary = {
        "strict_lower": profiles["candidate"]["cost"] < profiles["authority"]["cost"],
        "cost_delta": profiles["authority"]["cost"] - profiles["candidate"]["cost"],
        "score_gain": math.log(profiles["authority"]["cost"] / profiles["candidate"]["cost"]),
        "delta_exact": delta["exact_whitelist_delta"],
        "static_pass": static["candidate"]["pass"],
        "carrier_exhaustive_pass": carrier["pass"],
        "truthful_shapes": all(row["truthful"] for row in shapes),
        "known_gold_four_configs": all(
            row["candidate_gold"] == row["valid"] for row in evaluations["known"].values()
        ),
        "all_sampled_pass_through": all(row["pass_through"] for row in eval_rows),
        "runtime_errors_total": sum(row["runtime_errors"] for row in eval_rows),
        "final_nonfinite_total": sum(
            row["authority_final_nonfinite"] + row["candidate_final_nonfinite"] for row in eval_rows
        ),
        "intermediate_nonfinite_inherited": {
            model: {
                name: sum(row[f"{model}_intermediate_nonfinite"][name] for row in eval_rows)
                for name in ("selF", "selLog", "selQ")
            }
            for model in ("authority", "candidate")
        },
    }
    summary["audit_pass"] = bool(
        summary["strict_lower"]
        and summary["delta_exact"]
        and summary["static_pass"]
        and summary["carrier_exhaustive_pass"]
        and summary["truthful_shapes"]
        and summary["known_gold_four_configs"]
        and summary["all_sampled_pass_through"]
        and summary["runtime_errors_total"] == 0
        and summary["final_nonfinite_total"] == 0
    )
    result = {
        "authority": {"path": str(AUTHORITY), "sha256": sha256(authority_data)},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(candidate_data)},
        "profiles": profiles,
        "graph_delta": delta,
        "static": static,
        "runtime_shapes": shapes,
        "operator_exhaustive": carrier,
        "evaluations": evaluations,
        "summary": summary,
    }
    print("AUDIT_JSON_BEGIN")
    print(json.dumps(safe(result), indent=2))
    print("AUDIT_JSON_END")
    assert summary["audit_pass"]


if __name__ == "__main__":
    main()
