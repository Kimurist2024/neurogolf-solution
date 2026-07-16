#!/usr/bin/env python3
"""Fail-closed audit for the low38 eight-target expansion.

This script is read-only with respect to every submission/score authority.  It
extracts copies of the selected members into this lane and writes only evidence.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
PREVIOUS = ROOT / "submission_base_8004.50.zip"
TARGETS = (141, 4, 254, 49, 287, 78, 95, 7)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

RULES = {
    141: "For every cell, sum the values on its two full diagonals, counting the center once.",
    4: "For each row except the bottom, scan right-to-left and shift each cell one place right exactly when the row below contains a nonzero strictly to its right.",
    254: "In the fixed 9x9 gray-bar layout, recolor the shortest bar red (2) and the tallest bar blue (1), preserving bar parity support.",
    49: "Find the globally least-frequent nonzero color and output the rows containing it, cropped to the color's repeated horizontal run.",
    287: "For each cell pair under 180-degree symmetry, choose the current cell when it is color 4, otherwise the opposite cell.",
    78: "Sort each column so zeros precede nonzeros, preserving the nonzero order (gravity upward in the generator representation).",
    95: "Apply the generator's two-pass transpose-local OR dilation to every row/column of the fixed 9x9 binary pattern.",
    7: "Let m0,m1,m2 be maxima of flattened cells at indices modulo 3; emit the cyclic m0,m1,m2 sequence over all 49 output cells.",
}

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_private_exact15.audit_exact import (  # noqa: E402
    trace_shapes,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


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
            dims = tensor_shape(value) if value is not None else None
            if dims is None:
                raise RuntimeError(f"non-static node output: {name}")
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structural(model: onnx.ModelProto, task: int) -> dict[str, object]:
    errors: list[str] = []
    checker = False
    strict = False
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
        errors.append(f"strict_data_prop:{type(exc).__name__}:{exc}")
    inspected = inferred if inferred is not None else model
    values = list(inspected.graph.input) + list(inspected.graph.value_info) + list(inspected.graph.output)
    static_positive = all(tensor_shape(value) is not None for value in values)
    nonstandard_domains = sorted(
        {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
        | {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
    )
    ops = collections.Counter(node.op_type for node in model.graph.node)
    giant_einsum = [
        {"node_index": index, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Einsum" and len(node.input) > 16
    ]
    huge_fanin = [
        {"node_index": index, "op": node.op_type, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if len(node.input) > 64
    ]
    lookup_nodes = [
        node.op_type
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND", "ScatterElements"}
    ]
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    declared_cost = None
    if inferred is not None:
        try:
            declared_cost = static_cost(model, inferred)
        except Exception as exc:
            errors.append(f"static_cost:{type(exc).__name__}:{exc}")
    try:
        shapes = trace_shapes(model, task)
    except Exception as exc:
        shapes = {"error": f"{type(exc).__name__}: {exc}", "shape_cloak": True}
    return {
        "checker_full": checker,
        "strict_shape_data_prop": strict,
        "static_positive": static_positive,
        "standard_domains": not nonstandard_domains,
        "nonstandard_domains": nonstandard_domains,
        "banned_ops": banned,
        "conv_bias_findings": check_conv_bias(model),
        "op_histogram": dict(sorted(ops.items())),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "giant_einsum": giant_einsum,
        "huge_fanin": huge_fanin,
        "lookup_or_scatter_nodes": lookup_nodes,
        "declared_cost": declared_cost,
        "runtime_shape_trace": shapes,
        "errors": errors,
    }


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low38_task{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def audit_rule(task: int) -> dict[str, object]:
    transform = load_rule(task)
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    right = wrong = errors = 0
    first_failure = None
    input_shapes: set[tuple[int, int]] = set()
    output_shapes: set[tuple[int, int]] = set()
    split_counts: dict[str, int] = {}
    for split, examples in payload.items():
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or "input" not in example:
                continue
            split_counts[split] = split_counts.get(split, 0) + 1
            grid = copy.deepcopy(example["input"])
            expected = example["output"]
            input_shapes.add((len(grid), len(grid[0])))
            output_shapes.add((len(expected), len(expected[0])))
            try:
                actual = normalize(transform(grid))
                if actual == expected:
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {
                        "split": split,
                        "index": index,
                        "kind": "error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
    total = right + wrong + errors
    return {
        "task": task,
        "rule_summary": RULES[task],
        "known": {
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "total": total,
            "perfect": right == total and total > 0,
            "first_failure": first_failure,
        },
        "split_counts": split_counts,
        "input_shapes": [list(shape) for shape in sorted(input_shapes)],
        "output_shapes": [list(shape) for shape in sorted(output_shapes)],
    }


def main() -> None:
    baseline_rows = []
    with zipfile.ZipFile(BASE) as current, zipfile.ZipFile(PREVIOUS) as previous:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = current.read(member)
            old = previous.read(member)
            model = onnx.load_model_from_string(data)
            (HERE / "baselines" / member).write_bytes(data)
            with tempfile.TemporaryDirectory(prefix=f"low38_{task}_", dir="/tmp") as temp:
                path = Path(temp) / member
                path.write_bytes(data)
                memory, params, cost = cost_of(str(path))
            baseline_rows.append(
                {
                    "task": task,
                    "member": member,
                    "sha256": sha(data),
                    "file_bytes": len(data),
                    "unchanged_from_8004_50": data == old,
                    "previous_member_sha256": sha(old),
                    "actual_cost": {"memory": memory, "params": params, "cost": cost},
                    "structure": structural(model, task),
                }
            )
            print(f"task{task:03d}: cost={cost} unchanged={data == old}", flush=True)

    exact_scan = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text()
    )
    exact_hits = []
    for section in ("baseline_structural_failures", "opportunities", "candidates"):
        block = exact_scan.get(section, [])
        rows = block.values() if isinstance(block, dict) else block
        for row in rows:
            if isinstance(row, dict) and row.get("task") in TARGETS:
                exact_hits.append({"section": section, **row})
    baseline = {
        "baseline": {
            "path": BASE.name,
            "sha256": sha(BASE.read_bytes()),
            "previous_path": PREVIOUS.name,
            "previous_sha256": sha(PREVIOUS.read_bytes()),
        },
        "targets": baseline_rows,
        "exact_wave2": {
            "tasks_scanned": exact_scan["tasks_scanned"],
            "accepted": exact_scan["summary"].get("accepted", 0),
            "target_hits": exact_hits,
        },
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(baseline, indent=2) + "\n")

    rule_rows = []
    for task in TARGETS:
        row = audit_rule(task)
        rule_rows.append(row)
        print(f"task{task:03d}: rule={row['known']['right']}/{row['known']['total']}", flush=True)
    rule_audit = {
        "source": "inputs/sakana-gcg-2025/raw/taskNNN.py",
        "dataset": "inputs/neurogolf-2026/taskNNN.json",
        "targets_completed": len(rule_rows),
        "all_perfect": all(row["known"]["perfect"] for row in rule_rows),
        "rows": rule_rows,
    }
    (HERE / "true_rule_audit.json").write_text(json.dumps(rule_audit, indent=2) + "\n")

    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    archive_relevant = [
        row
        for task_rows in archive["retained"].values()
        for row in task_rows
        if row.get("task") in TARGETS
    ]
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    harvest_relevant = [row for row in harvest["rows"] if row.get("task") in TARGETS]
    profiles = {}
    for task in (254, 287, 7):
        path = ROOT / f"scripts/golf/loop_8003_40/agent_archive_rescreen/profiles/task{task:03d}_known.json"
        profiles[str(task)] = json.loads(path.read_text())
    b23 = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_b23/scan_build_manifest.json").read_text()
    )
    b23_tasks = [row for row in b23["tasks"] if row.get("task") in TARGETS]
    history = {
        "archive_inventory": {
            "stats": archive["stats"],
            "relevant_retained_lower": archive_relevant,
        },
        "focused_harvest": {
            "inventory": harvest["inventory"]["counts"],
            "relevant_rows": harvest_relevant,
        },
        "known_profiles_for_lower_leads": profiles,
        "exact_initializer_alias_scan_b23": {
            "scope": b23["scope"],
            "target_rows": b23_tasks,
        },
        "task254_safe_rebuild": {
            "report": "scripts/golf/loop_7999_13/lane_b13/REPORT.md",
            "result": "No correct task254 model below cost 76 with <=16 Einsum operands; 60 TT candidates failed, exact TT family floor 114.",
        },
        "task254_external_counterexample": {
            "report": "scripts/golf/loop_8003_40/agent_archive_resume/REPORT.md",
            "result": "cost42 giant-Einsum candidate differs from exact baseline on 412/500 external threshold cases despite generator fresh5000 accuracy.",
        },
        "task007_cost68_known_dual": {
            "evidence": "scripts/golf/loop_8004_42_plus20/agent_new_low38/evidence/task007_cost68_known_dual.json",
            "result": "260/266 in both default and ORT_DISABLE_ALL; reject before fresh testing.",
        },
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
