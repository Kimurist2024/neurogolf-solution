#!/usr/bin/env python3
"""Fail-closed inventory of historical margin-only rejects and scale repairs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "scan.json"
CANDIDATES = HERE / "candidates"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
PRIMARY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/quick_k5.json"
SECONDARY = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/lower_quick_k20.json"
BACKLOG = ROOT / "scripts/golf/agent_policy90_backlog_281/candidates.json"
ACTIVE_MANIFEST = ROOT / "others/71407/MANIFEST.json"
CANONICAL_COSTS = ROOT / (
    "scripts/golf/loop_8004_42_plus20/root_mem_census_119/canonical_costs.json"
)
TASK161_SOURCE_REPORT = ROOT / "scripts/golf/root_task161_policy90_275/REPORT.md"
TASK161_REPAIR_REPORT = ROOT / "scripts/golf/root_task161_margin_repair_279/REPORT.md"
MAX_TASKS = 50
EXPECTED_IO = [1, 10, 30, 30]
GIANT_EINSUM_INPUTS = 15
GIANT_INITIALIZER_ELEMENTS = 100_000

# Conservative public/private-zero and unresolved-unsound monitor catalog used by
# the repository's POLICY90 backlog audit.  An entry is excluded even if a public
# screen happens to look good.
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
LOOKUP_OPS = {
    "TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND",
    "ScatterElements", "CategoryMapper", "OneHot",
}
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from lib import scoring  # noqa: E402


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def nested_graphs(model: onnx.ModelProto) -> int:
    total = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                total += 1
                pending.extend(attribute.g.node)
            elif attribute.type == AttributeProto.GRAPHS:
                total += len(attribute.graphs)
                for graph in attribute.graphs:
                    pending.extend(graph.node)
    return total


def output_producer(model: onnx.ModelProto) -> onnx.NodeProto | None:
    if len(model.graph.output) != 1:
        return None
    output = model.graph.output[0].name
    matches = [node for node in model.graph.node if output in node.output]
    return matches[0] if len(matches) == 1 else None


def terminal_homogeneous_initializers(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Find direct one-use initializer inputs whose positive scale scales output."""
    producer = output_producer(model)
    if producer is None or producer.op_type not in {"Einsum", "MatMul", "Mul", "Conv"}:
        return []
    if producer.op_type == "Conv" and len(producer.input) > 2 and producer.input[2]:
        return []
    initializers = {item.name: item for item in model.graph.initializer}
    use_counts = Counter(item for node in model.graph.node for item in node.input)
    opportunities = []
    for name in sorted(set(producer.input)):
        if name not in initializers:
            continue
        occurrences = list(producer.input).count(name)
        if occurrences != 1 or use_counts[name] != 1:
            continue
        array = numpy_helper.to_array(initializers[name])
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            continue
        if not np.any(array != 0):
            continue
        opportunities.append({
            "initializer": name,
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "elements": int(array.size),
            "producer": producer.name,
            "producer_op": producer.op_type,
            "direct_occurrences_in_terminal_node": occurrences,
            "total_graph_input_occurrences": int(use_counts[name]),
            "proof": "positive scalar multiplication of this one-use direct terminal linear input uniformly scales graph output",
        })
    return opportunities


def official_profile(task: int, data: bytes) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"margin282_{task:03d}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, work,
            label=f"margin282_{task:03d}", require_correct=False,
        )


def structure(task: int, path: Path, authority_cost: int) -> dict[str, Any]:
    data = path.read_bytes()
    model = onnx.load_model_from_string(data)
    full_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        full_error = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"
    initializer_arrays = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    nonfinite = sorted(
        name for name, array in initializer_arrays.items()
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
    )
    domains = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    ops = [node.op_type for node in model.graph.node]
    lookup = sorted(set(ops) & LOOKUP_OPS)
    banned = sorted({op for op in ops if op in BANNED_OPS or "Sequence" in op})
    max_inputs = max((len(node.input) for node in model.graph.node), default=0)
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    giant_initializers = sorted(
        name for name, array in initializer_arrays.items()
        if array.size >= GIANT_INITIALIZER_ELEMENTS
    )
    trace: dict[str, Any]
    try:
        trace = runtime_shape_trace(task, copy.deepcopy(model))
        trace["truthful"] = bool(
            not trace.get("error") and not trace.get("declared_actual_mismatches", [])
        )
    except Exception as exc:  # noqa: BLE001
        trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    profile = official_profile(task, data)
    input_shapes = [tensor_shape(item) for item in model.graph.input]
    output_shapes = [tensor_shape(item) for item in model.graph.output]
    clean = bool(
        full_error is None and strict_error is None
        and input_shapes == [EXPECTED_IO] and output_shapes == [EXPECTED_IO]
        and not domains and not lookup and not banned and not nonfinite
        and nested_graphs(model) == 0 and len(model.functions) == 0
        and len(model.graph.sparse_initializer) == 0
        and not any(item.data_location == TensorProto.EXTERNAL or item.external_data for item in model.graph.initializer)
        and max_einsum_inputs < GIANT_EINSUM_INPUTS and not giant_initializers
        and trace.get("truthful") is True
        and profile is not None and int(profile["cost"]) < authority_cost
    )
    return {
        "path": rel(path),
        "sha256": digest_bytes(data),
        "file_bytes": len(data),
        "full_check": full_error is None,
        "full_check_error": full_error,
        "strict_data_prop": strict_error is None,
        "strict_data_prop_error": strict_error,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "op_histogram": dict(Counter(ops)),
        "nonstandard_domains": domains,
        "lookup_ops": lookup,
        "banned_ops": banned,
        "nonfinite_initializers": nonfinite,
        "nested_graphs": nested_graphs(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": sorted(
            item.name for item in model.graph.initializer
            if item.data_location == TensorProto.EXTERNAL or item.external_data
        ),
        "initializer_elements": int(sum(array.size for array in initializer_arrays.values())),
        "largest_initializer_elements": int(max((array.size for array in initializer_arrays.values()), default=0)),
        "giant_initializers": giant_initializers,
        "max_node_inputs": max_inputs,
        "max_einsum_inputs": max_einsum_inputs,
        "giant_einsum": max_einsum_inputs >= GIANT_EINSUM_INPUTS,
        "runtime_shape_trace": trace,
        "official_profile": profile,
        "authority_cost": authority_cost,
        "strict_lower": bool(profile is not None and int(profile["cost"]) < authority_cost),
        "terminal_homogeneous_initializers": terminal_homogeneous_initializers(model),
        "clean_nonlookup_noncloak_nongiant_strict_lower": clean,
    }


def compact_record(row: dict[str, Any], source: Path) -> dict[str, Any]:
    result = row.get("result", {})
    return {
        "source": rel(source),
        "task": int(row["task"]),
        "path": row.get("path"),
        "baseline_cost": row.get("baseline_cost"),
        "strictly_cheaper": row.get("strictly_cheaper"),
        "runtime_exception": row.get("runtime_exception"),
        "decision": result.get("decision"),
        "cost": result.get("cost"),
        "lib_gold": result.get("lib_gold"),
        "official_gold": result.get("official_gold"),
        "margin_stable": result.get("margin_stable"),
        "margin_min": result.get("margin_min"),
        "fresh_total": result.get("fresh_total"),
        "fresh_fails": result.get("fresh_fails"),
        "fresh_rate": result.get("fresh_rate"),
        "fresh_ok": result.get("fresh_ok"),
    }


def historical_margin_only(record: dict[str, Any]) -> bool:
    return bool(
        record["margin_stable"] is False
        and record["lib_gold"] is True
        and record["official_gold"] is True
        and record["fresh_ok"] is True
        and record["runtime_exception"] is False
        and record["cost"] is not None
        and record["baseline_cost"] is not None
        and int(record["cost"]) < int(record["baseline_cost"])
    )


def main() -> int:
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority changed")
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    secondary = json.loads(SECONDARY.read_text(encoding="utf-8"))
    primary_tasks = sorted({int(row["task"]) for row in primary if row.get("result", {}).get("margin_stable") is False})
    secondary_pool = sorted({
        int(row["task"]) for row in secondary
        if row.get("result", {}).get("margin_stable") is False and int(row["task"]) not in primary_tasks
    })
    selected_tasks = primary_tasks + secondary_pool[: MAX_TASKS - len(primary_tasks)]
    if len(primary_tasks) != 42 or len(selected_tasks) != MAX_TASKS:
        raise RuntimeError(
            f"historical screen cohort changed: primary={len(primary_tasks)} selected={len(selected_tasks)}"
        )

    active_manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    active_tasks = sorted(int(item["task"]) for item in active_manifest["active_candidates"])
    if len(active_tasks) != 22:
        raise RuntimeError(f"expected active22, got {len(active_tasks)}")
    canonical = json.loads(CANONICAL_COSTS.read_text(encoding="utf-8"))
    authority_costs = {int(row["task"]): int(row["cost"]) for row in canonical["ranked"]}

    all_compact = [
        compact_record(row, PRIMARY) for row in primary
        if int(row["task"]) in selected_tasks and row.get("result", {}).get("margin_stable") is False
    ] + [
        compact_record(row, SECONDARY) for row in secondary
        if int(row["task"]) in selected_tasks and int(row["task"]) not in primary_tasks
        and row.get("result", {}).get("margin_stable") is False
    ]

    task_rows = []
    structural_candidates = []
    eligible = []
    for task in selected_tasks:
        records = [row for row in all_compact if row["task"] == task]
        margin_only = [row for row in records if historical_margin_only(row)]
        exclusions = []
        if task in active_tasks:
            exclusions.append("71407_active")
        if task in PRIVATE_ZERO_OR_UNSOUND:
            exclusions.append("private_zero_or_unsound_monitor")
        structures = []
        for record in margin_only:
            path = ROOT / str(record["path"])
            item = structure(task, path, authority_costs[task])
            structures.append(item)
            structural_candidates.append({
                "task": task,
                "historical_record": record,
                "exclusions": exclusions,
                "structure": item,
            })
            if not exclusions and item["clean_nonlookup_noncloak_nongiant_strict_lower"]:
                eligible.append({"task": task, "record": record, "structure": item})
        task_rows.append({
            "task": task,
            "historical_margin_false_records": len(records),
            "historical_margin_only_records": len(margin_only),
            "classification": (
                "ELIGIBLE_MARGIN_REPAIR_SOURCE" if margin_only and not exclusions
                and any(item["clean_nonlookup_noncloak_nongiant_strict_lower"] for item in structures)
                else "MARGIN_ONLY_BUT_POLICY_EXCLUDED" if margin_only
                else "NOT_MARGIN_ONLY_ACCURACY_OR_COST_FAILED"
            ),
            "exclusions": exclusions,
            "records": records,
            "structures": structures,
        })

    # The scan is designed to stop and demand an explicit repair audit if a new
    # eligible source appears.  In the pinned cohort both margin-only sources are
    # policy-excluded, so no model should be emitted.
    if eligible:
        raise RuntimeError(f"eligible repair sources require audit implementation: {eligible}")

    backlog = json.loads(BACKLOG.read_text(encoding="utf-8"))
    backlog_margin = [
        row for row in backlog["candidates"] if row.get("classification") == "REJECT_MARGIN"
    ]
    if backlog_margin:
        raise RuntimeError("POLICY90 backlog acquired margin-only rejects after this scan was designed")

    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_members = len([name for name in archive.namelist() if name.endswith(".onnx")])
    payload = {
        "lane": "agent_margin_repair_scan_282",
        "decision": "NO_ELIGIBLE_MARGIN_REPAIR_CANDIDATES",
        "authority": {
            "zip": rel(AUTHORITY),
            "sha256": AUTHORITY_SHA256,
            "onnx_members": authority_members,
            "cost_source": rel(CANONICAL_COSTS),
            "cost_source_sha256": digest(CANONICAL_COSTS),
        },
        "scope": {
            "maximum_tasks": MAX_TASKS,
            "selected_tasks": selected_tasks,
            "selected_task_count": len(selected_tasks),
            "selection": "all 42 unique margin_stable=false tasks from primary quick_k5, then first 8 unique sorted tasks from secondary lower_quick_k20",
            "primary_task_count": len(primary_tasks),
            "secondary_added_task_count": len(selected_tasks) - len(primary_tasks),
            "historical_margin_false_record_count": len(all_compact),
        },
        "sources": {
            "primary": {"path": rel(PRIMARY), "sha256": digest(PRIMARY)},
            "secondary": {"path": rel(SECONDARY), "sha256": digest(SECONDARY)},
            "policy90_backlog_crosscheck": {
                "path": rel(BACKLOG),
                "sha256": digest(BACKLOG),
                "source_candidate_count": backlog["source_candidate_count"],
                "source_task_count": backlog["source_task_count"],
                "classification_counts": backlog["classification_counts"],
                "reject_margin_count": len(backlog_margin),
            },
            "task161_precedent": {
                "unrepaired_report": rel(TASK161_SOURCE_REPORT),
                "unrepaired_report_sha256": digest(TASK161_SOURCE_REPORT),
                "repaired_report": rel(TASK161_REPAIR_REPORT),
                "repaired_report_sha256": digest(TASK161_REPAIR_REPORT),
                "known_method": "positive x8 scale of one-use terminal initializer poly; cost neutral",
                "excluded_here": "task161 is already repaired and is active in 71407",
            },
        },
        "exclusions": {
            "active_manifest": rel(ACTIVE_MANIFEST),
            "active_manifest_sha256": digest(ACTIVE_MANIFEST),
            "active_task_count": len(active_tasks),
            "active_tasks": active_tasks,
            "private_zero_or_unsound_monitor": sorted(PRIVATE_ZERO_OR_UNSOUND),
        },
        "task_inventory": task_rows,
        "historical_margin_only_candidate_count": len(structural_candidates),
        "historical_margin_only_candidates": structural_candidates,
        "eligible_repair_source_count": len(eligible),
        "eligible_repair_sources": eligible,
        "repairs_emitted": [],
        "preliminary_audits": [],
        "preliminary_audit_status": "SKIPPED_NO_ELIGIBLE_CANDIDATE",
        "summary": {
            "tasks_scanned": len(selected_tasks),
            "historical_margin_only_candidates": len(structural_candidates),
            "task277": "exact accuracy/cost/margin-only, but private-zero monitored and TfIdfVectorizer lookup",
            "task328": "exact accuracy/cost/margin-only, but active71407 and giant 75-input terminal Einsum",
            "policy90_backlog_104_candidates_margin_rejects": len(backlog_margin),
            "eligible": 0,
        },
        "policy": {
            "normal_policy90_only": True,
            "strict_lower_than_8009_46": True,
            "private_zero_excluded": True,
            "lookup_excluded": True,
            "shape_cloak_excluded": True,
            "giant_einsum_threshold_inputs": GIANT_EINSUM_INPUTS,
            "giant_initializer_threshold_elements": GIANT_INITIALIZER_ELEMENTS,
            "root_or_others71407_written": False,
            "kimi_used": False,
        },
    }
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "tasks_scanned": len(selected_tasks),
        "margin_only_candidates": len(structural_candidates),
        "eligible": len(eligible),
        "output": rel(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
