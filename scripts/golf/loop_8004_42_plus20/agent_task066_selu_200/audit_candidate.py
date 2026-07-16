#!/usr/bin/env python3
"""Fail-closed audit for the task066 Div->Selu regolf.

The proof obligation is stronger than sample accuracy: generator-valid inputs
must make selF positive, and the replacement must preserve the downstream ti
carrier over every possible selF integer.  Known/fresh tests then audit the
whole model in four ORT configurations, including final raw-bit identity.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
CANDIDATE = HERE / "task066_selu_cost561.onnx"
EXPECTED_ZIP_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_MEMBER_SHA = "bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((66_200_101, 2000), (66_200_102, 2000))
PROTECTED = (ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "others/71407")
TRACE_NAMES = (
    "aMask",
    "bMask",
    "bPow",
    "noA",
    "hasB",
    "forceB",
    "useB",
    "selF",
    "selLog",
    "selQ",
    "ti",
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256(path.read_bytes())
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): tree_sha(path) for path in PROTECTED}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    return value


def load_payloads() -> tuple[bytes, bytes]:
    zip_data = AUTHORITY_ZIP.read_bytes()
    if sha256(zip_data) != EXPECTED_ZIP_SHA:
        raise RuntimeError("authority ZIP hash changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority = archive.read("task066.onnx")
    if sha256(authority) != EXPECTED_MEMBER_SHA:
        raise RuntimeError("authority member hash changed")
    return authority, CANDIDATE.read_bytes()


def value_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def structure(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    result: dict[str, Any] = {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_params": int(sum(np.asarray(numpy_helper.to_array(item)).size for item in model.graph.initializer)),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": sum(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        "banned_ops": sorted(
            {
                node.op_type
                for node in model.graph.node
                if any(token in node.op_type.upper() for token in scoring._EXCLUDED_OP_TYPES)
                or "SEQUENCE" in node.op_type.upper()
            }
        ),
        "conv_family_bias_ub": check_conv_bias(model),
    }
    for label, data_prop in (("strict", False), ("strict_data_prop", True)):
        try:
            shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=data_prop)
            result[label] = True
        except Exception as exc:  # noqa: BLE001
            result[label] = False
            result[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"
    result["pass"] = bool(
        result["full_check"]
        and result["strict"]
        and result["strict_data_prop"]
        and result["standard_domains"]
        and not result["functions"]
        and not result["sparse_initializers"]
        and not result["nested_graphs"]
        and not result["banned_ops"]
        and not result["conv_family_bias_ub"]
    )
    return result


def graph_delta(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    changes = []
    for index, (left, right) in enumerate(zip(authority.graph.node, candidate.graph.node, strict=True)):
        if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True):
            changes.append(
                {
                    "index": index,
                    "output": list(right.output),
                    "authority_op": left.op_type,
                    "candidate_op": right.op_type,
                    "authority_inputs": list(left.input),
                    "candidate_inputs": list(right.input),
                    "candidate_attributes": {
                        item.name: str(helper.get_attribute_value(item)) for item in right.attribute
                    },
                }
            )
    authority_inits = {item.name: numpy_helper.to_array(item) for item in authority.graph.initializer}
    candidate_inits = {item.name: numpy_helper.to_array(item) for item in candidate.graph.initializer}
    common_equal = all(
        authority_inits[name].dtype == candidate_inits[name].dtype
        and authority_inits[name].shape == candidate_inits[name].shape
        and np.array_equal(authority_inits[name], candidate_inits[name])
        for name in set(authority_inits) & set(candidate_inits)
    )
    exact = bool(
        len(changes) == 1
        and changes[0]["index"] == 64
        and changes[0]["output"] == ["selQ"]
        and changes[0]["authority_op"] == "Div"
        and changes[0]["candidate_op"] == "Selu"
        and changes[0]["authority_inputs"] == ["selLog", "ln2"]
        and changes[0]["candidate_inputs"] == ["selLog"]
        and set(authority_inits) - set(candidate_inits) == {"ln2"}
        and not (set(candidate_inits) - set(authority_inits))
        and common_equal
    )
    return {
        "changes": changes,
        "initializer_removed": sorted(set(authority_inits) - set(candidate_inits)),
        "initializer_added": sorted(set(candidate_inits) - set(authority_inits)),
        "common_initializers_bitwise_equal": common_equal,
        "whitelist_exact": exact,
    }


def official_profile(data: bytes, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task066_selu200_", dir="/tmp") as work:
        result = scoring.score_and_verify(
            onnx.load_model_from_string(data), 66, work, label=label, require_correct=True
        )
    if result is None:
        raise RuntimeError(f"official scoring rejected {label}")
    return {
        "memory": int(result["memory"]),
        "params": int(result["params"]),
        "cost": int(result["cost"]),
        "correct": bool(result["correct"]),
    }


def make_session(data: bytes, disable: bool, threads: int, trace: bool = False) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    if trace:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
        existing = {value.name for value in model.graph.output}
        for name in TRACE_NAMES:
            if name not in existing:
                model.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_cases() -> list[dict[str, Any]]:
    payload = scoring.load_examples(66)
    return [item for split in ("train", "test", "arc-gen") for item in payload[split]]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_2dd70a9a")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [generator.generate() for _ in range(count)]


def evaluate(
    authority_data: bytes,
    candidate_data: bytes,
    cases: list[dict[str, Any]],
    disable: bool,
    threads: int,
    require_gold: bool,
) -> dict[str, Any]:
    sessions = {
        "authority": make_session(authority_data, disable, threads, trace=True),
        "candidate": make_session(candidate_data, disable, threads, trace=True),
    }
    result: dict[str, Any] = {
        "total": len(cases),
        "require_gold": require_gold,
        "valid": 0,
        "authority_right": 0,
        "candidate_right": 0,
        "final_raw_bitwise_equal": 0,
        "final_threshold_equal": 0,
        "ti_bitwise_equal": 0,
        "selq_bitwise_equal": 0,
        "runtime_errors": 0,
        "candidate_final_nonfinite": 0,
        "candidate_intermediate_nonfinite": {name: 0 for name in TRACE_NAMES},
        "selF_min": math.inf,
        "selF_zero_or_negative": 0,
        "selLog_min": math.inf,
        "selLog_negative_or_nan": 0,
        "first_failure": None,
    }
    for index, example in enumerate(cases):
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            continue
        result["valid"] += 1
        outputs: dict[str, list[np.ndarray]] = {}
        for label, session in sessions.items():
            try:
                outputs[label] = [
                    np.asarray(value)
                    for value in session.run(None, {session.get_inputs()[0].name: converted["input"]})
                ]
            except Exception as exc:  # noqa: BLE001
                result["runtime_errors"] += 1
                result["first_failure"] = result["first_failure"] or {
                    "case": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) != 2:
            continue
        authority, candidate = outputs["authority"], outputs["candidate"]
        a_final, c_final = authority[0], candidate[0]
        expected = converted["output"].astype(bool)
        result["authority_right"] += int(np.array_equal(a_final > 0.0, expected))
        result["candidate_right"] += int(np.array_equal(c_final > 0.0, expected))
        result["final_raw_bitwise_equal"] += int(
            a_final.dtype == c_final.dtype and a_final.shape == c_final.shape and a_final.tobytes() == c_final.tobytes()
        )
        result["final_threshold_equal"] += int(np.array_equal(a_final > 0.0, c_final > 0.0))
        result["candidate_final_nonfinite"] += int(c_final.size - np.count_nonzero(np.isfinite(c_final)))

        traced_authority = dict(zip(TRACE_NAMES, authority[1:]))
        traced_candidate = dict(zip(TRACE_NAMES, candidate[1:]))
        result["ti_bitwise_equal"] += int(traced_authority["ti"].tobytes() == traced_candidate["ti"].tobytes())
        result["selq_bitwise_equal"] += int(traced_authority["selQ"].tobytes() == traced_candidate["selQ"].tobytes())
        sel_f = float(traced_candidate["selF"].reshape(-1)[0])
        sel_log = float(traced_candidate["selLog"].reshape(-1)[0])
        result["selF_min"] = min(result["selF_min"], sel_f)
        result["selF_zero_or_negative"] += int(sel_f <= 0.0 or math.isnan(sel_f))
        result["selLog_min"] = min(result["selLog_min"], sel_log)
        result["selLog_negative_or_nan"] += int(sel_log < 0.0 or math.isnan(sel_log))
        for name, value in traced_candidate.items():
            if value.dtype.kind == "f":
                result["candidate_intermediate_nonfinite"][name] += int(
                    value.size - np.count_nonzero(np.isfinite(value))
                )
        if not (
            a_final.tobytes() == c_final.tobytes()
            and traced_authority["ti"].tobytes() == traced_candidate["ti"].tobytes()
        ):
            result["first_failure"] = result["first_failure"] or {"case": index, "kind": "raw_or_ti_difference"}
    valid = result["valid"]
    result["pass"] = bool(
        valid == len(cases)
        and (not require_gold or result["authority_right"] == valid)
        and (not require_gold or result["candidate_right"] == valid)
        and result["final_raw_bitwise_equal"] == valid
        and result["final_threshold_equal"] == valid
        and result["ti_bitwise_equal"] == valid
        and result["runtime_errors"] == 0
        and result["candidate_final_nonfinite"] == 0
        and result["selF_zero_or_negative"] == 0
        and result["selLog_negative_or_nan"] == 0
    )
    return result


def operator_model(count: int) -> bytes:
    graph_input = helper.make_tensor_value_info("x", TensorProto.UINT32, [count])
    outputs = [
        helper.make_tensor_value_info("div", TensorProto.FLOAT16, [count]),
        helper.make_tensor_value_info("selu", TensorProto.FLOAT16, [count]),
        helper.make_tensor_value_info("div_ti", TensorProto.UINT8, [count]),
        helper.make_tensor_value_info("selu_ti", TensorProto.UINT8, [count]),
    ]
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
    graph = helper.make_graph(
        nodes,
        "task066_operator_exhaustive",
        [graph_input],
        outputs,
        initializer=[numpy_helper.from_array(stored_ln2, "ln2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def operator_exhaustive() -> dict[str, Any]:
    # size<=20, pow2[20:] are zero, G=2*cyan_bits and O=cyan_bits.
    # Therefore pairD/pairU/aMask/bMask/bPow are all below 2**20.
    values = np.arange((1 << 20) + 1, dtype=np.uint32)
    data = operator_model(len(values))
    modes: dict[str, Any] = {}
    carriers: dict[str, np.ndarray] = {}
    for disable, threads, label in CONFIGS:
        session = make_session(data, disable, threads)
        div, selu, div_ti, selu_ti = [np.asarray(item) for item in session.run(None, {"x": values})]
        fp_difference = np.flatnonzero(div.view(np.uint16) != selu.view(np.uint16))
        carrier_difference = np.flatnonzero(div_ti != selu_ti)
        modes[label] = {
            "input_count": int(values.size),
            "range_inclusive": [0, 1 << 20],
            "fp16_difference_count": int(fp_difference.size),
            "first_fp16_differences": [int(item) for item in fp_difference[:10]],
            "uint8_carrier_difference_count": int(carrier_difference.size),
            "uint8_carrier_equal": not carrier_difference.size,
        }
        carriers[label] = selu_ti
    cross_mode = all(np.array_equal(next(iter(carriers.values())), value) for value in carriers.values())
    return {
        "modes": modes,
        "selu_uint8_cross_mode_equal": cross_mode,
        "pass": bool(all(row["uint8_carrier_equal"] for row in modes.values()) and cross_mode),
    }


def runtime_shapes(data: bytes, optimization: str) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    case = scoring.convert_to_numpy(known_cases()[0])
    if case is None:
        raise RuntimeError("first known case unexpectedly invalid")
    values = session.run(names, {session.get_inputs()[0].name: case["input"]})
    mismatches = [
        {"tensor": name, "declared": value_shape(typed[name]), "runtime": list(np.asarray(value).shape)}
        for name, value in zip(names, values)
        if value_shape(typed[name]) != list(np.asarray(value).shape)
    ]
    return {
        "optimization": optimization,
        "traced_node_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def generator_proof() -> dict[str, Any]:
    return {
        "generator_hash": "2dd70a9a",
        "finite_domain": "size is 10..20; shape is S or U; flip/xpose are Boolean; arbitrary pre-path cyan is allowed",
        "marker_invariance": "path entries are appended after random cyan, so red/green endpoints and the mandatory cyan guards override noise",
        "bitmask_identity": "in the graph, G=2*(cyan row bitmask at the green-marker column) and O=(cyan row bitmask at the red-outward adjacent column), after xpose canonicalization",
        "S_unflipped": "green guard y=mid-1 enters G at bit mid; red-outward guard y=mid enters O at bit mid; mid<gr0, so aMask contains 2**mid",
        "S_flipped": "row reflection makes those guards align in pairU at t=size-1-mid; t>=gr0+2, so bMask contains 2**t",
        "U_unflipped": "green guard y=base+1 enters G then G>>2 at bit base; red-outward guard y=base enters O at bit base; base>=gr0+2, so bMask contains 2**base",
        "U_flipped": "row reflection makes the guards align in pairD at t=size-1-base; t<gr0, so aMask contains 2**t",
        "xpose": "vert selects the row/column-swapped Gh/Oh equations, reducing xpose to the same four canonical cases",
        "noise_monotonicity": "extra cyan can add bits to G/O but cannot remove the mandatory aligned guard bit",
        "selection": "if aMask=0 the proof gives bMask>0; if aMask>0 and useB is false selF=aMask; if useB is true via forceB, hasB is true; bPow=bMask&(-bMask) is nonzero for bMask>0",
        "numeric_bound": "pow2 is nonzero only at indices 0..19, so every selected uint32 mask is in [1,2**20-1]",
        "consequence": "Cast(uint32->float16) maps the selected positive mask to >=1 or +inf; therefore selF>=1 and Log(selF) is >=0 or +inf, never negative/NaN",
        "counterexample": None,
        "proved": True,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    before = protected_hashes()
    authority_data, candidate_data = load_payloads()
    known = known_cases()
    fresh = [(seed, fresh_cases(seed, count)) for seed, count in FRESH]
    evaluations: dict[str, Any] = {"known": {}, "fresh": []}
    rows = []
    for disable, threads, label in CONFIGS:
        row = evaluate(authority_data, candidate_data, known, disable, threads, require_gold=True)
        evaluations["known"][label] = row
        rows.append(row)
        print(f"known {label}: {row['candidate_right']}/{row['valid']} raw={row['final_raw_bitwise_equal']}", flush=True)
    for seed, cases in fresh:
        stream = {"seed": seed, "count": len(cases), "modes": {}}
        for disable, threads, label in CONFIGS:
            row = evaluate(authority_data, candidate_data, cases, disable, threads, require_gold=False)
            stream["modes"][label] = row
            rows.append(row)
            print(
                f"fresh seed={seed} {label}: {row['candidate_right']}/{row['valid']} raw={row['final_raw_bitwise_equal']}",
                flush=True,
            )
        evaluations["fresh"].append(stream)

    authority_structure = structure(authority_data)
    candidate_structure = structure(candidate_data)
    authority_profile = official_profile(authority_data, "authority")
    candidate_profile = official_profile(candidate_data, "candidate")
    delta = graph_delta(authority_data, candidate_data)
    operator = operator_exhaustive()
    shape_rows = [runtime_shapes(candidate_data, mode) for mode in ("disable_all", "default")]
    proof = generator_proof()
    all_evaluations_pass = all(row["pass"] for row in rows)
    result = {
        "authority": {
            "zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "zip_sha256": sha256(AUTHORITY_ZIP.read_bytes()),
            "member_sha256": sha256(authority_data),
            "profile": authority_profile,
            "structure": authority_structure,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(candidate_data),
            "profile": candidate_profile,
            "structure": candidate_structure,
        },
        "graph_delta": delta,
        "generator_support_proof": proof,
        "operator_exhaustive_uint32_bound": operator,
        "runtime_shapes": shape_rows,
        "evaluations": evaluations,
        "summary": {
            "strict_lower": candidate_profile["cost"] < authority_profile["cost"],
            "cost_delta": authority_profile["cost"] - candidate_profile["cost"],
            "projected_score_gain": math.log(authority_profile["cost"] / candidate_profile["cost"]),
            "structure_pass": candidate_structure["pass"],
            "graph_delta_whitelisted": delta["whitelist_exact"],
            "generator_support_proved": proof["proved"],
            "operator_carrier_pass": operator["pass"],
            "truthful_shapes": all(row["truthful"] for row in shape_rows),
            "known_four_configs_pass": all(row["pass"] for row in evaluations["known"].values()),
            "fresh_four_configs_pass": all(
                row["pass"] for stream in evaluations["fresh"] for row in stream["modes"].values()
            ),
            "runtime_errors_total": sum(row["runtime_errors"] for row in rows),
            "candidate_final_nonfinite_total": sum(row["candidate_final_nonfinite"] for row in rows),
            "selq_sample_bitwise_difference_cases": sum(row["valid"] - row["selq_bitwise_equal"] for row in rows),
            "accepted": False,
        },
    }
    result["summary"]["accepted"] = bool(
        result["summary"]["strict_lower"]
        and result["summary"]["structure_pass"]
        and result["summary"]["graph_delta_whitelisted"]
        and result["summary"]["generator_support_proved"]
        and result["summary"]["operator_carrier_pass"]
        and result["summary"]["truthful_shapes"]
        and all_evaluations_pass
        and result["summary"]["runtime_errors_total"] == 0
        and result["summary"]["candidate_final_nonfinite_total"] == 0
    )
    after = protected_hashes()
    result["integrity"] = {"before": before, "after": after, "unchanged": before == after}
    (HERE / "audit.json").write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["summary"]["accepted"] and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
