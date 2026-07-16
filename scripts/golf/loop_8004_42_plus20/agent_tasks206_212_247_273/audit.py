#!/usr/bin/env python3
"""Read-only SOUND audit of the 8009.46 members for tasks 206/212/247/273.

The script deliberately does not emit or promote a candidate.  It verifies the
generator rule, measures the exact ZIP member, inventories only universally
exact mechanical reductions, and exercises the incumbent under four ORT
configurations on known and freshly generated cases.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
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
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline"
TASKS = {
    206: "88a10436",
    212: "8d510a79",
    247: "a3325580",
    273: "af902bf9",
}
AUTHORITY_COSTS = {206: 194, 212: 240, 247: 212, 273: 193}
SEEDS = (71407261, 71407262)
# Four configurations are deliberately exercised.  Giant-Einsum task212 is
# unusually slow with ORT optimizations disabled, so use 100 independent cases
# per seed here and reconcile the result with its existing 500-case dual-mode
# audit in REPORT.md.
FRESH_PER_SEED = 100
CONFIGS = (
    ("disable_all_t1", True, 1),
    ("disable_all_t4", True, 4),
    ("default_t1", False, 1),
    ("default_t4", False, 4),
)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TFIDFVECTORIZER", "HARDMAX", "SCATTERELEMENTS", "SCATTERND"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"raw_task{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def rule_output(rule, grid: list[list[int]]) -> list[list[int]]:
    return rule(copy.deepcopy(grid))


def json_rule_audit(task: int, rule) -> dict[str, Any]:
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    rows = []
    for split in ("train", "test", "arc-gen"):
        right = 0
        first_failure = None
        for index, example in enumerate(payload.get(split, [])):
            actual = rule_output(rule, example["input"])
            ok = actual == example["output"]
            right += int(ok)
            if not ok and first_failure is None:
                first_failure = index
        rows.append(
            {
                "split": split,
                "total": len(payload.get(split, [])),
                "right": right,
                "first_failure": first_failure,
            }
        )
    return {"splits": rows, "pass": all(row["right"] == row["total"] for row in rows)}


def shape_of(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def make_session(model: onnx.ModelProto, disable_all: bool, threads: int) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        clean.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def structure(task: int, model: onnx.ModelProto, first_input: np.ndarray) -> dict[str, Any]:
    errors = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"strict:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    static = all(shape_of(value) is not None for value in values if value.type.HasField("tensor_type"))
    declared_output = shape_of(inferred.graph.output[0])
    try:
        raw = make_session(model, True, 1).run(None, {"input": first_input})[0]
        runtime_output = list(raw.shape)
        finite = bool(np.isfinite(raw).all())
    except Exception as exc:  # noqa: BLE001
        runtime_output = None
        finite = False
        errors.append(f"runtime:{type(exc).__name__}:{exc}")
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    domains = {item.domain for item in model.opset_import} | {node.domain for node in model.graph.node}
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive": static,
        "canonical_input": shape_of(inferred.graph.input[0]) == [1, 10, 30, 30],
        "canonical_truthful_output": declared_output == [1, 10, 30, 30] == runtime_output,
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "no_banned": all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "no_nested_graphs": all(
            attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        "no_lookup": all(node.op_type.upper() not in LOOKUP for node in model.graph.node),
        "no_shape_cloak": all(node.op_type != "CenterCropPad" for node in model.graph.node),
        "no_giant_einsum": max_einsum <= 16,
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_runtime": finite,
    }
    return {
        "task": task,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "max_einsum_inputs": max_einsum,
        "declared_output_shape": declared_output,
        "runtime_output_shape": runtime_output,
        "checks": checks,
        "sound_admissible": all(checks.values()),
        "errors": errors,
    }


def mechanical_census(model: onnx.ModelProto) -> dict[str, Any]:
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    duplicates = []
    names = list(arrays)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            a, b = arrays[name], arrays[other]
            if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b):
                duplicates.append([name, other])
    repeated_axes = []
    for name, array in arrays.items():
        for axis, size in enumerate(array.shape):
            if size > 1:
                first = np.take(array, [0], axis=axis)
                if np.array_equal(array, np.repeat(first, size, axis=axis)):
                    repeated_axes.append({"initializer": name, "axis": axis, "size": size})
    identities = []
    for index, node in enumerate(model.graph.node):
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if node.op_type == "Identity":
            identities.append({"node": index, "kind": "Identity"})
        elif node.op_type in {"Concat", "Sum", "Max", "Min"} and len(node.input) == 1:
            identities.append({"node": index, "kind": "single_input_variadic"})
        elif node.op_type == "Transpose" and attrs.get("perm") == list(range(len(attrs.get("perm", [])))):
            identities.append({"node": index, "kind": "identity_transpose"})
    signatures: dict[bytes, int] = {}
    cse = []
    for index, node in enumerate(model.graph.node):
        clone = copy.deepcopy(node)
        clone.name = ""
        del clone.output[:]
        key = clone.SerializeToString(deterministic=True)
        if key in signatures:
            cse.append({"node": index, "prior": signatures[key], "op": node.op_type})
        else:
            signatures[key] = index
    return {
        "unused_initializers": sorted(set(arrays) - used),
        "duplicate_initializer_aliases": duplicates,
        "repeated_initializer_axes": repeated_axes,
        "identity_nodes": identities,
        "cse_nodes": cse,
        "universally_exact_cost_reducing_probe_count": len(duplicates) + len(identities) + len(cse),
        "note": "Repeated-axis contraction is only a lead: Einsum/Scatter dimensional labels must also remain valid.",
    }


def converted_known(task: int) -> list[dict[str, np.ndarray]]:
    return [
        item
        for split in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(split, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]


def init_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "raw_equal_reference": 0,
        "raw_unequal_reference": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "small_positive_values_0_to_0p25": 0,
        "first_failure": None,
    }


def update(stats: dict[str, Any], raw: np.ndarray, expected: np.ndarray, reference: np.ndarray) -> None:
    correct = raw.shape == expected.shape and np.array_equal(raw > 0, expected)
    stats["right" if correct else "wrong"] += 1
    same = np.array_equal(raw, reference, equal_nan=True)
    stats["raw_equal_reference" if same else "raw_unequal_reference"] += 1
    stats["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(raw)))
    stats["small_positive_values_0_to_0p25"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
    if (not correct or not same) and stats["first_failure"] is None:
        stats["first_failure"] = {
            "correct": correct,
            "raw_equal_reference": same,
            "shape": list(raw.shape),
        }


def run_cases(model: onnx.ModelProto, cases: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    sessions = {
        label: make_session(model, disabled, threads)
        for label, disabled, threads in CONFIGS
    }
    stats = {label: init_stats() for label, _, _ in CONFIGS}
    reference_label = CONFIGS[0][0]
    for case in cases:
        expected = case["output"] > 0
        raws: dict[str, np.ndarray] = {}
        for label, _, _ in CONFIGS:
            try:
                raws[label] = sessions[label].run(None, {"input": case["input"]})[0]
            except Exception as exc:  # noqa: BLE001
                stats[label]["runtime_errors"] += 1
                if stats[label]["first_failure"] is None:
                    stats[label]["first_failure"] = {"runtime_error": f"{type(exc).__name__}:{exc}"}
        if reference_label not in raws:
            continue
        reference = raws[reference_label]
        for label, raw in raws.items():
            update(stats[label], raw, expected, reference)
    return {
        "total": len(cases),
        "configs": stats,
        "pass": all(
            row["right"] == len(cases)
            and row["wrong"] == 0
            and row["raw_equal_reference"] == len(cases)
            and row["raw_unequal_reference"] == 0
            and row["runtime_errors"] == 0
            and row["nonfinite_values"] == 0
            and row["small_positive_values_0_to_0p25"] == 0
            for row in stats.values()
        ),
    }


def fresh_cases(task: int, task_hash: str, rule, seed: int) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    module = importlib.import_module(f"task_{task_hash}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    cases = []
    attempts = 0
    rule_right = 0
    while len(cases) < FRESH_PER_SEED and attempts < FRESH_PER_SEED * 20:
        attempts += 1
        example = module.generate()
        rule_right += int(rule_output(rule, example["input"]) == example["output"])
        converted = scoring.convert_to_numpy(example)
        if converted is not None:
            cases.append(converted)
    return cases, {
        "seed": seed,
        "attempts": attempts,
        "converted": len(cases),
        "raw_rule_right": rule_right,
        "raw_rule_total": attempts,
        "raw_rule_pass": rule_right == attempts,
    }


def official_profile(task: int, model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"task{task:03d}_score_", dir=HERE) as directory:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, directory, label="baseline", require_correct=False
        )
    if result is None:
        return {"error": "score_and_verify returned None"}
    return {
        "memory": int(result["memory"]),
        "params": int(result["params"]),
        "cost": int(result["cost"]),
        "correct": bool(result["correct"]),
        "authority_cost_match": int(result["cost"]) == AUTHORITY_COSTS[task],
    }


def main() -> int:
    output: dict[str, Any] = {
        "baseline": "submission_base_8009.46.zip",
        "fresh_per_seed": FRESH_PER_SEED,
        "seeds": list(SEEDS),
        "configs": [label for label, _, _ in CONFIGS],
        "tasks": {},
    }
    for task, task_hash in TASKS.items():
        path = BASE / f"task{task:03d}.onnx"
        model = onnx.load(path)
        rule = load_raw_rule(task)
        known = converted_known(task)
        fresh_rows = []
        for seed in SEEDS:
            cases, generation = fresh_cases(task, task_hash, rule, seed)
            generation["runtime"] = run_cases(model, cases)
            fresh_rows.append(generation)
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "generator_hash": task_hash,
            "rule_json": json_rule_audit(task, rule),
            "official_profile": official_profile(task, model),
            "structure": structure(task, model, known[0]["input"]),
            "mechanical_census": mechanical_census(model),
            "known_four_config": run_cases(model, known),
            "fresh_two_seed_four_config": fresh_rows,
        }
        row["candidate_eligible"] = bool(
            row["rule_json"]["pass"]
            and row["official_profile"].get("authority_cost_match")
            and row["structure"]["sound_admissible"]
            and row["mechanical_census"]["universally_exact_cost_reducing_probe_count"] > 0
            and row["known_four_config"]["pass"]
            and all(item["raw_rule_pass"] and item["runtime"]["pass"] for item in fresh_rows)
        )
        output["tasks"][str(task)] = row
        (HERE / "evidence" / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
        print(
            f"task{task:03d} cost={row['official_profile'].get('cost')} "
            f"rule={row['rule_json']['pass']} known4={row['known_four_config']['pass']} "
            f"fresh={all(item['runtime']['pass'] for item in fresh_rows)} "
            f"sound={row['structure']['sound_admissible']} "
            f"probes={row['mechanical_census']['universally_exact_cost_reducing_probe_count']}",
            flush=True,
        )
    output["winner"] = None
    output["projected_gain"] = 0.0
    output["pass"] = all(
        row["rule_json"]["pass"]
        and row["official_profile"].get("authority_cost_match")
        and row["known_four_config"]["pass"]
        and all(item["raw_rule_pass"] and item["runtime"]["pass"] for item in row["fresh_two_seed_four_config"])
        for row in output["tasks"].values()
    )
    (HERE / "evidence" / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0 if output["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
