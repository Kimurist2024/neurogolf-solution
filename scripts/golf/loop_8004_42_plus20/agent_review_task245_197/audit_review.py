#!/usr/bin/env python3
"""Independent fail-closed review of root's task245 Selu shave."""

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
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_task245_regolf_196/task245_selu_cost384.onnx"
EXPECTED_ZIP_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_BASE_SHA = "228b6ad9f24579bc6f5840da4e5a18f08343b76a26538f500e7e77d328d6e9d5"
EXPECTED_CAND_SHA = "1b777a51c55fa98ed720fb993a9305bcca2218627592d23e46d8d5a6bce91ba9"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((245_197_101, 3000), (245_197_102, 3000))
CODE_NAMES = ("rr_code", "rc_code", "gr_code", "gc_code")
LOG_NAMES = ("rr_log", "rc_log", "gr_log", "gc_log")
PROTECTED = (ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "others/71407")

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return digest(path.read_bytes())
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        h.update(item.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}


def json_safe(value: Any) -> Any:
    """Keep the evidence strict-JSON even when an exposed fp16 source is +inf."""
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
    if digest(zip_data) != EXPECTED_ZIP_SHA:
        raise RuntimeError("authority zip hash changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        base = archive.read("task245.onnx")
    candidate = CANDIDATE.read_bytes()
    if digest(base) != EXPECTED_BASE_SHA or digest(candidate) != EXPECTED_CAND_SHA:
        raise RuntimeError("payload hash mismatch")
    return base, candidate


def equations(model: onnx.ModelProto) -> dict[str, str]:
    result = {}
    for node in model.graph.node:
        if node.op_type != "Einsum":
            continue
        attr = next(item for item in node.attribute if item.name == "equation")
        result[node.output[0]] = helper.get_attribute_value(attr).decode()
    return result


def structure(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    row: dict[str, Any] = {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": int(sum(np.asarray(numpy_helper.to_array(item)).size for item in model.graph.initializer)),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "domains_standard": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": sum(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if any(token in node.op_type.upper() for token in scoring._EXCLUDED_OP_TYPES)
            or "SEQUENCE" in node.op_type.upper()
        ],
        "conv_bias_findings": [],
    }
    try:
        row["conv_bias_findings"] = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        row["conv_bias_findings"] = [{"error": f"{type(exc).__name__}: {exc}"}]
    try:
        onnx.checker.check_model(model, full_check=True)
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
    row["pass_except_inherited_data_prop"] = bool(
        row["full_check"]
        and row["strict"]
        and row["domains_standard"]
        and not row["functions"]
        and not row["sparse_initializers"]
        and not row["nested_graphs"]
        and not row["banned_ops"]
        and not row["conv_bias_findings"]
    )
    return row


def graph_delta(base_data: bytes, cand_data: bytes) -> dict[str, Any]:
    base = onnx.load_model_from_string(base_data)
    cand = onnx.load_model_from_string(cand_data)
    base_inits = {item.name: np.asarray(numpy_helper.to_array(item)) for item in base.graph.initializer}
    cand_inits = {item.name: np.asarray(numpy_helper.to_array(item)) for item in cand.graph.initializer}
    init_removed = sorted(set(base_inits) - set(cand_inits))
    init_added = sorted(set(cand_inits) - set(base_inits))
    common_init_equal = all(
        base_inits[name].dtype == cand_inits[name].dtype
        and base_inits[name].shape == cand_inits[name].shape
        and np.array_equal(base_inits[name], cand_inits[name])
        for name in set(base_inits) & set(cand_inits)
    )
    node_changes = []
    for index, (left, right) in enumerate(zip(base.graph.node, cand.graph.node, strict=True)):
        if left.SerializeToString(deterministic=True) == right.SerializeToString(deterministic=True):
            continue
        node_changes.append(
            {
                "index": index,
                "base_op": left.op_type,
                "candidate_op": right.op_type,
                "base_inputs": list(left.input),
                "candidate_inputs": list(right.input),
                "output": list(right.output),
                "base_attrs": {a.name: str(helper.get_attribute_value(a)) for a in left.attribute},
                "candidate_attrs": {a.name: str(helper.get_attribute_value(a)) for a in right.attribute},
            }
        )
    base_vi = {
        value.name: [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]
        for value in base.graph.value_info
    }
    cand_vi = {
        value.name: [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]
        for value in cand.graph.value_info
    }
    vi_changes = {
        name: {"base": base_vi.get(name), "candidate": cand_vi.get(name)}
        for name in sorted(set(base_vi) | set(cand_vi))
        if base_vi.get(name) != cand_vi.get(name)
    }
    expected_einsum = {9, 11, 13, 15}
    expected_div_selu = {17, 18, 19, 20}
    changed_indices = {row["index"] for row in node_changes}
    whitelist = bool(
        init_removed == ["two_f16"]
        and not init_added
        and common_init_equal
        and changed_indices == expected_einsum | expected_div_selu
        and all(row["base_op"] == "Einsum" and row["candidate_op"] == "Einsum" for row in node_changes if row["index"] in expected_einsum)
        and all(row["base_op"] == "Div" and row["candidate_op"] == "Selu" for row in node_changes if row["index"] in expected_div_selu)
        and set(vi_changes) == set(CODE_NAMES) | set(LOG_NAMES)
        and all(change == {"base": [], "candidate": [1]} for change in vi_changes.values())
    )
    return {
        "initializer_removed": init_removed,
        "initializer_added": init_added,
        "unchanged_initializers_bitwise": common_init_equal,
        "node_changes": node_changes,
        "value_info_changes": vi_changes,
        "whitelist_exact": whitelist,
    }


def official_profile(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="task245review197_", dir="/tmp") as work:
        measured = scoring.score_and_verify(
            onnx.load_model_from_string(data), 245, work, label=label, require_correct=True
        )
    if measured is None:
        raise RuntimeError(f"competition score_and_verify rejected {label}")
    return {
        "memory": int(measured["memory"]),
        "params": int(measured["params"]),
        "cost": int(measured["cost"]),
        "correct": bool(measured["correct"]),
    }


def make_session(data: bytes, disable: bool, threads: int, traced: bool = False) -> ort.InferenceSession:
    model = copy.deepcopy(onnx.load_model_from_string(data))
    if traced:
        typed = {value.name: value for value in model.graph.value_info}
        existing = {value.name for value in model.graph.output}
        for name in CODE_NAMES + LOG_NAMES:
            if name not in existing:
                model.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])


def raw_bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.uint8)


def evaluate_cases(
    base_data: bytes,
    cand_data: bytes,
    cases: list[dict[str, Any]],
    disable: bool,
    threads: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "candidate_right": 0,
        "authority_right": 0,
        "final_raw_equal": 0,
        "final_threshold_equal": 0,
        "code_log_bitwise_equal": 0,
        "runtime_errors": {"candidate": 0, "authority": 0},
        "candidate_final_nonfinite": 0,
        "first_failure": None,
        "source_stats": {
            name: {"min": math.inf, "max": -math.inf, "le_one": 0, "nonfinite": 0}
            for name in CODE_NAMES
        },
        "log_stats": {
            name: {"min": math.inf, "max": -math.inf, "le_zero": 0, "nonfinite": 0}
            for name in LOG_NAMES
        },
    }
    sessions = {
        "authority": make_session(base_data, disable, threads, traced=True),
        "candidate": make_session(cand_data, disable, threads, traced=True),
    }
    for index, example in enumerate(cases):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        row["valid"] += 1
        outputs: dict[str, list[np.ndarray]] = {}
        for label, session in sessions.items():
            try:
                values = session.run(None, {session.get_inputs()[0].name: benchmark["input"]})
                outputs[label] = [np.asarray(value) for value in values]
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"][label] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) != 2:
            continue
        authority, candidate = outputs["authority"], outputs["candidate"]
        a_final, c_final = authority[0], candidate[0]
        expected = benchmark["output"].astype(bool)
        row["authority_right"] += int(a_final.shape == expected.shape and np.array_equal(a_final > 0, expected))
        row["candidate_right"] += int(c_final.shape == expected.shape and np.array_equal(c_final > 0, expected))
        raw_equal = bool(
            a_final.dtype == c_final.dtype
            and a_final.shape == c_final.shape
            and np.array_equal(raw_bits(a_final), raw_bits(c_final))
        )
        threshold_equal = bool(np.array_equal(a_final > 0, c_final > 0))
        row["final_raw_equal"] += int(raw_equal)
        row["final_threshold_equal"] += int(threshold_equal)
        row["candidate_final_nonfinite"] += int(c_final.size - np.count_nonzero(np.isfinite(c_final)))
        intermediate_equal = True
        for offset, name in enumerate(CODE_NAMES + LOG_NAMES, start=1):
            left = authority[offset].reshape(-1)
            right = candidate[offset].reshape(-1)
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(raw_bits(left), raw_bits(right)):
                intermediate_equal = False
            value = float(right[0])
            if name in CODE_NAMES:
                stats = row["source_stats"][name]
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["le_one"] += int(value <= 1.0)
                stats["nonfinite"] += int(not np.isfinite(value))
            else:
                stats = row["log_stats"][name]
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["le_zero"] += int(value <= 0.0)
                stats["nonfinite"] += int(not np.isfinite(value))
        row["code_log_bitwise_equal"] += int(intermediate_equal)
        if not (raw_equal and threshold_equal and intermediate_equal):
            row["first_failure"] = row["first_failure"] or {"index": index, "comparison": "raw_or_intermediate"}
    valid = row["valid"]
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["pass"] = bool(
        valid == len(cases)
        and row["candidate_right"] == valid
        and row["authority_right"] == valid
        and row["final_raw_equal"] == valid
        and row["final_threshold_equal"] == valid
        and row["code_log_bitwise_equal"] == valid
        and row["runtime_errors_total"] == 0
        and row["candidate_final_nonfinite"] == 0
        and all(stats["le_one"] == 0 for stats in row["source_stats"].values())
        and all(stats["le_zero"] == 0 for stats in row["log_stats"].values())
    )
    return row


def operator_model(kind: str, count: int) -> bytes:
    graph_input = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [count])
    graph_output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [count])
    if kind == "Div":
        nodes = [helper.make_node("Div", ["x", "two"], ["y"])]
        inits = [numpy_helper.from_array(np.asarray([2.0], dtype=np.float16), "two")]
    else:
        nodes = [helper.make_node("Selu", ["x"], ["y"], alpha=1.0, gamma=0.5)]
        inits = []
    graph = helper.make_graph(nodes, f"review_{kind}", [graph_input], [graph_output], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def operator_exhaustive() -> dict[str, Any]:
    bits = np.arange(0x0000, 0x7C01, dtype=np.uint16)  # +0 through +inf inclusive
    values = bits.view(np.float16)
    modes = {}
    selu_by_mode = {}
    for disable, threads, label in CONFIGS:
        outputs = {}
        for kind in ("Div", "Selu"):
            session = make_session(operator_model(kind, len(values)), disable, threads)
            outputs[kind] = np.asarray(
                session.run(None, {session.get_inputs()[0].name: values})[0]
            ).view(np.uint16)
        difference = np.flatnonzero(outputs["Div"] != outputs["Selu"])
        modes[label] = {
            "different_count": int(difference.size),
            "bitwise_equal": not difference.size,
            "includes_positive_infinity": True,
        }
        selu_by_mode[label] = outputs["Selu"]
    cross_mode = all(np.array_equal(next(iter(selu_by_mode.values())), value) for value in selu_by_mode.values())
    return {
        "input_count": int(values.size),
        "bit_range": ["0x0000", "0x7c00"],
        "modes": modes,
        "selu_cross_mode_bitwise_equal": cross_mode,
        "pass": bool(all(row["bitwise_equal"] for row in modes.values()) and cross_mode),
    }


def known_cases() -> list[dict[str, Any]]:
    payload = scoring.load_examples(245)
    return [item for split in ("train", "test", "arc-gen") for item in payload[split]]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_a1570a43")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    rows = []
    while len(rows) < count:
        example = generator.generate()
        if scoring.convert_to_numpy(example) is not None:
            rows.append(example)
    return rows


def aggregate_domain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "source_min": {name: math.inf for name in CODE_NAMES},
        "source_max": {name: -math.inf for name in CODE_NAMES},
        "source_le_one": {name: 0 for name in CODE_NAMES},
        "source_nonfinite": {name: 0 for name in CODE_NAMES},
        "log_min": {name: math.inf for name in LOG_NAMES},
        "log_max": {name: -math.inf for name in LOG_NAMES},
        "log_le_zero": {name: 0 for name in LOG_NAMES},
        "log_nonfinite": {name: 0 for name in LOG_NAMES},
    }
    for row in rows:
        for name, stats in row["source_stats"].items():
            result["source_min"][name] = min(result["source_min"][name], stats["min"])
            result["source_max"][name] = max(result["source_max"][name], stats["max"])
            result["source_le_one"][name] += stats["le_one"]
            result["source_nonfinite"][name] += stats["nonfinite"]
        for name, stats in row["log_stats"].items():
            result["log_min"][name] = min(result["log_min"][name], stats["min"])
            result["log_max"][name] = max(result["log_max"][name], stats["max"])
            result["log_le_zero"][name] += stats["le_zero"]
            result["log_nonfinite"][name] += stats["nonfinite"]
    result["strict_positive"] = bool(
        all(value == 0 for value in result["source_le_one"].values())
        and all(value == 0 for value in result["log_le_zero"].values())
    )
    return result


def main() -> int:
    ort.set_default_logger_severity(4)
    before = protected_hashes()
    base, candidate = load_payloads()
    base_model = onnx.load_model_from_string(base)
    cand_model = onnx.load_model_from_string(candidate)
    base_structure = structure(base)
    cand_structure = structure(candidate)
    known = known_cases()
    generated = [(seed, fresh_cases(seed, count)) for seed, count in FRESH]
    evaluations: dict[str, Any] = {"known": {}, "fresh": []}
    domain_rows = []
    for disable, threads, label in CONFIGS:
        row = evaluate_cases(base, candidate, known, disable, threads)
        evaluations["known"][label] = row
        domain_rows.append(row)
    for seed, cases in generated:
        stream = {"seed": seed, "count": len(cases), "modes": {}}
        for disable, threads, label in CONFIGS:
            row = evaluate_cases(base, candidate, cases, disable, threads)
            stream["modes"][label] = row
            domain_rows.append(row)
        evaluations["fresh"].append(stream)
        print(f"fresh seed={seed} count={len(cases)}", flush=True)
    base_profile = official_profile(base, "authority")
    cand_profile = official_profile(candidate, "candidate")
    all_eval = list(evaluations["known"].values()) + [
        row for stream in evaluations["fresh"] for row in stream["modes"].values()
    ]
    same_data_prop_error = (
        not base_structure["strict_data_prop"]
        and not cand_structure["strict_data_prop"]
        and base_structure.get("strict_data_prop_error") == cand_structure.get("strict_data_prop_error")
    )
    result = {
        "authority": {
            "zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "zip_sha256": digest(AUTHORITY_ZIP.read_bytes()),
            "member_sha256": digest(base),
            "profile": base_profile,
            "structure": base_structure,
            "einsum_equations": equations(base_model),
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(candidate),
            "profile": cand_profile,
            "structure": cand_structure,
            "einsum_equations": equations(cand_model),
        },
        "graph_delta": graph_delta(base, candidate),
        "generator_proof": {
            "hash": "a1570a43",
            "sprite": "conway_sprite(5,5,14) starts with 25 cells and removes at most 14; no row/column may vanish",
            "red_row": "a retained local row=4 cell maps to source row >=4; at least one other retained cell has a strictly positive exponential weight, so rr_code>1",
            "red_col": "a retained local col=4 cell maps to source col >=4; at least one other retained cell has a strictly positive exponential weight, so rc_code>1",
            "green_row": "green corners include row brow+6>=6; q4, pos_min, theta_base are nonnegative and theta_base has positive sum, so gr_code>1",
            "green_col": "green corners include col bcol+6>=6; q4, pos_min, theta_base are nonnegative and theta_base has positive sum, so gc_code>1",
            "log_consequence": "all four Log inputs are >1, hence all four Log outputs are strictly positive or +inf, never negative/zero/NaN",
        },
        "einsum_rank_proof": {
            "batch_n": 1,
            "authority": "each equation reduces n,c,r,s to a scalar; Div by two_f16[1] broadcasts it to shape[1]",
            "candidate": "each equation retains n and reduces c,r,s directly to shape[1]",
            "real_equivalence": "for static n=1 both contain exactly the same contraction terms and the downstream carrier is shape[1]",
            "measured_code_log_bitwise": all(row["code_log_bitwise_equal"] == row["valid"] for row in all_eval),
        },
        "operator_exhaustive_nonnegative_f16": operator_exhaustive(),
        "evaluations": evaluations,
        "domain_aggregate": aggregate_domain(domain_rows),
        "data_prop_inheritance": {
            "same_error_text": same_data_prop_error,
            "authority_error": base_structure.get("strict_data_prop_error"),
            "candidate_error": cand_structure.get("strict_data_prop_error"),
            "new_structural_errors": 0 if same_data_prop_error and cand_structure["pass_except_inherited_data_prop"] else 1,
        },
        "summary": {
            "strict_lower": cand_profile["cost"] < base_profile["cost"],
            "cost_delta": base_profile["cost"] - cand_profile["cost"],
            "graph_delta_whitelisted": graph_delta(base, candidate)["whitelist_exact"],
            "known_four_configs_pass": all(row["pass"] for row in evaluations["known"].values()),
            "fresh_four_configs_pass": all(
                row["pass"] for stream in evaluations["fresh"] for row in stream["modes"].values()
            ),
            "operator_pass": False,
            "source_domain_pass": False,
            "data_prop_inherited_only": same_data_prop_error,
            "runtime_errors_total": sum(row["runtime_errors_total"] for row in all_eval),
            "candidate_final_nonfinite_total": sum(row["candidate_final_nonfinite"] for row in all_eval),
            "accepted": False,
        },
    }
    result["summary"]["operator_pass"] = result["operator_exhaustive_nonnegative_f16"]["pass"]
    result["summary"]["source_domain_pass"] = result["domain_aggregate"]["strict_positive"]
    result["summary"]["accepted"] = bool(
        result["summary"]["strict_lower"]
        and result["summary"]["graph_delta_whitelisted"]
        and cand_structure["pass_except_inherited_data_prop"]
        and result["summary"]["known_four_configs_pass"]
        and result["summary"]["fresh_four_configs_pass"]
        and result["summary"]["operator_pass"]
        and result["summary"]["source_domain_pass"]
        and result["summary"]["data_prop_inherited_only"]
        and result["summary"]["runtime_errors_total"] == 0
        and result["summary"]["candidate_final_nonfinite_total"] == 0
    )
    after = protected_hashes()
    result["integrity"] = {"before": before, "after": after, "unchanged": before == after}
    (HERE / "audit.json").write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not result["summary"]["accepted"] or before != after:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
