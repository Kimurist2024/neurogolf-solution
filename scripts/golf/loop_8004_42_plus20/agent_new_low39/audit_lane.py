#!/usr/bin/env python3
"""Fail-closed audit for the low39 eight-task target expansion."""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
PREVIOUS = ROOT / "submission_base_8004.50.zip"
TARGETS = (32, 41, 215, 211, 120, 235, 258, 292)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

RULES = {
    32: "Sort every column ascending, which moves each column's nonzero cells to its bottom while preserving their colors.",
    41: "Scan the grid row-major with cumulative XOR state a and emit cell OR a at each position.",
    215: "For output row h, repeat the lexicographically maximum input row among rows h mod 3.",
    211: "Vertically concatenate reverse(input), input, reverse(input), and mirror every row horizontally as reverse(row)+row.",
    120: "Replace a positive cell C by color 8 iff its 3x3 neighborhood sum is greater than 8*C; otherwise keep C.",
    235: "Emit three identical rows of three decoded glyph colors, using the fixed second/third-row probes and the generator's modulo-9 formula.",
    258: "Copy the grid and paint 2 exactly in horizontal 1,0,1 gaps (the generator's missing alternating red cells).",
    292: "Preserve every cell except apply bitwise value OR value>>1 in columns whose index is 0 mod 3.",
}

LOWER_LEADS = {
    32: [ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task032_r01_static46.onnx"],
    120: [ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task120_r01_static41.onnx"],
    211: [ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task211_r01_static64.onnx"],
    292: [
        ROOT / "scripts/golf/scratch_codex/task292/candidate_rank1_sign_50.onnx",
        ROOT / "scripts/golf/scratch_codex/task292/candidate_shared_rank1_d50.onnx",
        ROOT / "scripts/golf/scratch_codex/task292/candidate_rank2_core_sign54.onnx",
    ],
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_private_exact15.audit_exact import (  # noqa: E402
    trace_shapes,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def static_cost(model: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, int]:
    infos = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    excluded = {value.name for value in inferred.graph.input}
    excluded.update(value.name for value in inferred.graph.output)
    excluded.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in excluded or name in seen:
                continue
            seen.add(name)
            value = infos.get(name)
            dims = shape(value) if value is not None else None
            if dims is None:
                raise RuntimeError(f"non-static output {name}")
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structure(model: onnx.ModelProto, task: int) -> dict[str, object]:
    errors = []
    checker = strict = False
    inferred = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        strict = True
    except Exception as exc:
        errors.append(f"strict:{type(exc).__name__}:{exc}")
    inspected = inferred if inferred is not None else model
    values = list(inspected.graph.input) + list(inspected.graph.value_info) + list(inspected.graph.output)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    giant = [
        {"index": index, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Einsum" and len(node.input) > 16
    ]
    huge = [
        {"index": index, "op": node.op_type, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if len(node.input) > 64
    ]
    lookup = [
        node.op_type
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND", "ScatterElements"}
    ]
    domains = sorted(
        {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
        | {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
    )
    try:
        runtime = trace_shapes(model, task)
    except Exception as exc:
        runtime = {"shape_cloak": True, "error": f"{type(exc).__name__}: {exc}"}
    declared = None
    if inferred is not None:
        try:
            declared = static_cost(model, inferred)
        except Exception as exc:
            errors.append(f"static_cost:{type(exc).__name__}:{exc}")
    return {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive": all(shape(value) is not None for value in values),
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type in BANNED or "Sequence" in node.op_type}),
        "conv_bias_findings": check_conv_bias(model),
        "ops": dict(sorted(ops.items())),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "giant_einsum": giant,
        "huge_fanin": huge,
        "lookup_or_scatter": lookup,
        "declared_cost": declared,
        "runtime_shapes": runtime,
        "errors": errors,
    }


def run_known(model: onnx.ModelProto, task: int, disable: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    examples = scoring.load_examples(task)
    total = sum(len(examples[name]) for name in ("train", "test", "arc-gen"))
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {"right": 0, "wrong": 0, "errors": total, "total": total, "session_error": f"{type(exc).__name__}: {exc}"}
    right = wrong = errors = skipped = near_margin = 0
    first_failure = None
    output_shapes: set[tuple[int, ...]] = set()
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            bench = scoring.convert_to_numpy(example)
            if bench is None:
                skipped += 1
                continue
            try:
                raw = session.run(["output"], {"input": bench["input"]})[0]
                output_shapes.add(tuple(int(item) for item in raw.shape))
                near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                if np.array_equal(raw > 0, bench["output"] > 0):
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {"split": split, "index": index, "kind": "runtime", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped": skipped,
        "total": right + wrong + errors,
        "near_margin_count": near_margin,
        "output_shapes": [list(item) for item in sorted(output_shapes)],
        "first_failure": first_failure,
    }


def audit_rule(task: int) -> dict[str, object]:
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low39_task{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    right = wrong = errors = 0
    first_failure = None
    splits = {}
    input_shapes = set()
    output_shapes = set()
    for split, examples in payload.items():
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or "input" not in example:
                continue
            splits[split] = splits.get(split, 0) + 1
            input_shapes.add((len(example["input"]), len(example["input"][0])))
            output_shapes.add((len(example["output"]), len(example["output"][0])))
            try:
                actual = normalize(module.p(copy.deepcopy(example["input"])))
                if actual == example["output"]:
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {"split": split, "index": index, "kind": "error", "error": f"{type(exc).__name__}: {exc}"}
    total = right + wrong + errors
    return {
        "task": task,
        "rule_summary": RULES[task],
        "known": {"right": right, "wrong": wrong, "errors": errors, "total": total, "perfect": right == total and total > 0, "first_failure": first_failure},
        "split_counts": splits,
        "input_shapes": [list(item) for item in sorted(input_shapes)],
        "output_shapes": [list(item) for item in sorted(output_shapes)],
    }


def measure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    with tempfile.TemporaryDirectory(prefix="low39_cost_", dir="/tmp") as temp:
        local = Path(temp) / path.name
        local.write_bytes(path.read_bytes())
        memory, params, cost = cost_of(str(local))
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), "actual_cost": {"memory": memory, "params": params, "cost": cost}, "structure": structure(model, int(path.name[4:7]) if path.name.startswith("task") and path.name[4:7].isdigit() else 292)}


def main() -> None:
    baseline_rows = []
    known_rows = []
    with zipfile.ZipFile(BASE) as current, zipfile.ZipFile(PREVIOUS) as previous:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = current.read(member)
            old = previous.read(member)
            path = HERE / "baselines" / member
            model = onnx.load_model_from_string(data)
            memory, params, cost = cost_of(str(path))
            row = {
                "task": task,
                "member": member,
                "sha256": sha(data),
                "file_bytes": len(data),
                "unchanged_from_8004_50": data == old,
                "previous_sha256": sha(old),
                "actual_cost": {"memory": memory, "params": params, "cost": cost},
                "structure": structure(model, task),
            }
            baseline_rows.append(row)
            known = {"task": task, "disable_all": run_known(model, task, True), "default": run_known(model, task, False)}
            known_rows.append(known)
            print(f"task{task:03d}: cost={cost} known={known['disable_all']['right']}/{known['disable_all']['total']} default={known['default']['right']}/{known['default']['total']}", flush=True)

    baseline = {
        "baseline": {"path": BASE.name, "sha256": sha(BASE.read_bytes()), "previous_path": PREVIOUS.name, "previous_sha256": sha(PREVIOUS.read_bytes())},
        "targets": baseline_rows,
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (HERE / "known_baseline_dual.json").write_text(json.dumps({"targets_completed": len(known_rows), "rows": known_rows}, indent=2) + "\n")

    rules = [audit_rule(task) for task in TARGETS]
    (HERE / "true_rule_audit.json").write_text(json.dumps({"source": "inputs/sakana-gcg-2025/raw/taskNNN.py", "dataset": "inputs/neurogolf-2026/taskNNN.json", "targets_completed": len(rules), "all_perfect": all(row["known"]["perfect"] for row in rules), "rows": rules}, indent=2) + "\n")

    lower = []
    for task, paths in LOWER_LEADS.items():
        for path in paths:
            model = onnx.load(path)
            measured = measure(path)
            measured["task"] = task
            measured["known_dual"] = {"disable_all": run_known(model, task, True), "default": run_known(model, task, False)}
            lower.append(measured)
            print(f"lead task{task:03d}: {path.name} cost={measured['actual_cost']['cost']} known={measured['known_dual']['disable_all']['right']}/{measured['known_dual']['disable_all']['total']}", flush=True)
    (HERE / "lower_leads_dual.json").write_text(json.dumps({"rows": lower}, indent=2) + "\n")

    archive = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text())
    retained = [row for rows in archive["retained"].values() for row in rows if row.get("task") in TARGETS]
    harvest = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text())
    harvest_rows = [row for row in harvest["rows"] if row.get("task") in TARGETS]
    b23 = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_b23/scan_build_manifest.json").read_text())
    exact = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text())
    exact_hits = []
    for kind, values in exact["opportunities"].items():
        for row in values:
            if row.get("task") in TARGETS:
                exact_hits.append({"kind": kind, **row})
    history = {
        "archive_inventory": {"stats": archive["stats"], "retained_numeric_leads": retained},
        "focused_harvest": {"inventory": harvest["inventory"]["counts"], "target_rows": harvest_rows},
        "exact_wave2": {"summary": exact["summary"], "target_hits": exact_hits},
        "exact_initializer_alias_b23": {"scope": b23["scope"], "target_rows": [row for row in b23["tasks"] if row.get("task") in TARGETS]},
        "scratch_reports": {
            "32": "scripts/golf/scratch_codex/task032/REPORT.md and FAILURE_LOG.md",
            "41": "scripts/golf/scratch_codex/task041/REPORT.md and FAILURE_LOG.md",
            "120": "scripts/golf/scratch_codex/task120/REPORT.md and FAILURE_LOG.md",
            "235": "scripts/golf/scratch_codex/task235/REPORT.md and FAILURE_LOG.md",
            "258": "scripts/golf/scratch_codex/task258/REPORT.md and FAILURE_LOG.md",
            "292": "scripts/golf/scratch_codex/task292/REPORT.md and FAILURE_LOG.md",
        },
        "lower_lead_evidence": "lower_leads_dual.json",
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
