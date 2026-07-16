#!/usr/bin/env python3
"""Fail-closed POLICY90 binary correction-branch ablation scan."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
import onnxoptimizer
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATES_DIR = HERE / "candidates"
EVIDENCE = HERE / "evidence.json"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
COST_CENSUS = ROOT / "scripts/golf/loop_8004_42_plus20/root_mem_census_119/canonical_costs.json"
ACTIVE_MANIFEST = ROOT / "others/71407/MANIFEST.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
PRIORITY_COST_MIN = 150
PRIORITY_COST_MAX = 500
POLICY_THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
TARGET_OPS = {"And", "Or", "Max", "Min", "Add", "Sub", "Mul"}
OPTIMIZER_PASSES = ["eliminate_deadend", "eliminate_unused_initializer"]
ASSIGNED_ACTIVE23 = {
    7, 12, 13, 66, 90, 101, 134, 158, 161, 175, 192, 205,
    209, 226, 245, 310, 319, 328, 333, 344, 349, 355, 366,
}
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("binary291_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load support module: {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha256(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def signature_json(
    signature: tuple[int, tuple[int | str | None, ...]] | None,
) -> dict[str, Any] | None:
    if signature is None:
        return None
    return {
        "dtype": TensorProto.DataType.Name(signature[0]),
        "shape": list(signature[1]),
    }


def branch_ablation(
    original: onnx.ModelProto, node_index: int, operand_index: int,
) -> onnx.ModelProto:
    model = copy.deepcopy(original)
    if not (0 <= node_index < len(model.graph.node)):
        raise ValueError("node index out of range")
    target = model.graph.node[node_index]
    if (
        target.op_type not in TARGET_OPS
        or len(target.input) < 2
        or not (0 <= operand_index < len(target.input))
        or len(target.output) != 1
        or not target.output[0]
        or not target.input[operand_index]
    ):
        raise ValueError("invalid binary-family ablation target")
    old_output = target.output[0]
    replacement = target.input[operand_index]
    graph_outputs = {value.name for value in model.graph.output}
    if old_output in graph_outputs:
        identity = helper.make_node(
            "Identity", [replacement], [old_output],
            name=f"binary_ablation_{node_index}_operand{operand_index}",
        )
        model.graph.node[node_index].CopyFrom(identity)
    else:
        for index, node in enumerate(model.graph.node):
            if index == node_index:
                continue
            for input_index, name in enumerate(node.input):
                if name == old_output:
                    node.input[input_index] = replacement
        del model.graph.node[node_index]
        keep = [value for value in model.graph.value_info if value.name != old_output]
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep)
    return onnxoptimizer.optimize(model, OPTIMIZER_PASSES)


def fresh_generation_clean(row: dict[str, Any]) -> bool:
    return bool(
        row.get("accepted") == FRESH_PER_SEED
        and row.get("generation_errors") == 0
        and row.get("conversion_skips") == 0
    )


def main() -> int:
    started = time.monotonic()
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority changed")
    if len(ASSIGNED_ACTIVE23) != 23:
        raise RuntimeError("assigned active23 snapshot malformed")
    census = json.loads(COST_CENSUS.read_text(encoding="utf-8"))
    authority_costs = {int(row["task"]): int(row["cost"]) for row in census["ranked"]}
    if (
        len(authority_costs) != 400
        or census.get("authority_zip") != "submission_base_8009.46.zip"
    ):
        raise RuntimeError("authority cost census is not immutable 8009.46 census")
    manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    observed_active = {int(row["task"]) for row in manifest["active_candidates"]}
    active_tasks = ASSIGNED_ACTIVE23 | observed_active
    concurrent_active = observed_active - ASSIGNED_ACTIVE23
    excluded_tasks = active_tasks | PRIVATE_ZERO_OR_UNSOUND
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    if len(task_map) != 400:
        raise RuntimeError("task hash map is not complete")
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    authority_data: dict[int, bytes] = {}
    authority_models: dict[int, onnx.ModelProto] = {}
    authority_inventory: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            authority_data[task] = data
            cost = authority_costs[task]
            exclusions = []
            if task in ASSIGNED_ACTIVE23:
                exclusions.append("assigned_active23")
            if task in concurrent_active:
                exclusions.append("concurrent_active")
            if task in PRIVATE_ZERO_OR_UNSOUND:
                exclusions.append("private_zero_or_unsound_monitor")
            record: dict[str, Any] = {
                "task": task,
                "cost": cost,
                "sha256": sha256(data),
                "priority_cost_150_to_500": PRIORITY_COST_MIN <= cost <= PRIORITY_COST_MAX,
                "excluded": bool(exclusions),
                "exclusion_reasons": exclusions,
                "target_nodes": 0,
                "operands": 0,
                "shape_dtype_exact_operands": 0,
            }
            if exclusions:
                authority_inventory.append(record)
                continue
            original = onnx.load_model_from_string(data)
            authority_models[task] = original
            try:
                inferred = onnx.shape_inference.infer_shapes(
                    copy.deepcopy(original), strict_mode=True, data_prop=True
                )
                values = SUPPORT.signature_map(inferred)
                record["strict_shape_data_prop"] = True
            except Exception as exc:  # noqa: BLE001
                record["strict_shape_data_prop"] = False
                record["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
                authority_inventory.append(record)
                continue
            node_rows = []
            for node_index, node in enumerate(inferred.graph.node):
                if (
                    node.op_type not in TARGET_OPS
                    or len(node.input) < 2
                    or len(node.output) != 1
                    or not node.output[0]
                ):
                    continue
                record["target_nodes"] += 1
                output_signature = values.get(node.output[0])
                operand_rows = []
                for operand_index, operand in enumerate(node.input):
                    if not operand:
                        continue
                    record["operands"] += 1
                    operand_signature = values.get(operand)
                    exact = bool(
                        output_signature is not None
                        and operand_signature == output_signature
                        and SUPPORT.fully_static(output_signature)
                    )
                    operand_row = {
                        "operand_index": operand_index,
                        "operand_value": operand,
                        "operand_signature": signature_json(operand_signature),
                        "output_signature": signature_json(output_signature),
                        "exact_shape_dtype_match": exact,
                    }
                    operand_rows.append(operand_row)
                    if exact:
                        record["shape_dtype_exact_operands"] += 1
                        variants.append({
                            "task": task,
                            "authority_cost": cost,
                            "authority_sha256": sha256(data),
                            "priority_cost_150_to_500": record["priority_cost_150_to_500"],
                            "node_index": node_index,
                            "node_name": node.name,
                            "op_type": node.op_type,
                            "node_output": node.output[0],
                            **operand_row,
                        })
                node_rows.append({
                    "node_index": node_index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "node_output": node.output[0],
                    "output_signature": signature_json(output_signature),
                    "operands": operand_rows,
                })
            record["nodes"] = node_rows
            authority_inventory.append(record)

    variants.sort(key=lambda row: (
        not row["priority_cost_150_to_500"], row["authority_cost"], row["task"],
        row["node_index"], row["operand_index"],
    ))
    print(json.dumps({
        "eligible_authorities": sum(not row["excluded"] for row in authority_inventory),
        "authorities_with_target_nodes": sum(row["target_nodes"] > 0 for row in authority_inventory),
        "target_nodes": sum(row["target_nodes"] for row in authority_inventory),
        "shape_dtype_exact_operand_variants": len(variants),
        "priority_variants": sum(row["priority_cost_150_to_500"] for row in variants),
    }), flush=True)

    candidate_rows: list[dict[str, Any]] = []
    candidate_data: dict[str, bytes] = {}
    dedupe: dict[tuple[int, str], int] = {}
    for variant_index, variant in enumerate(variants, start=1):
        task = int(variant["task"])
        row = copy.deepcopy(variant)
        original = authority_models[task]
        try:
            candidate = branch_ablation(original, row["node_index"], row["operand_index"])
            data = candidate.SerializeToString()
            row["candidate_sha256"] = sha256(data)
            row["candidate_file_bytes"] = len(data)
            row["authority_node_count"] = len(original.graph.node)
            row["candidate_node_count"] = len(candidate.graph.node)
            row["authority_initializer_count"] = len(original.graph.initializer)
            row["candidate_initializer_count"] = len(candidate.graph.initializer)
        except Exception as exc:  # noqa: BLE001
            row["classification"] = "REJECT_BUILD"
            row["build_error"] = f"{type(exc).__name__}: {exc}"
            candidate_rows.append(row)
            continue
        key = (task, row["candidate_sha256"])
        if key in dedupe:
            row["classification"] = "REJECT_DUPLICATE_CANDIDATE"
            row["duplicate_of_candidate_index"] = dedupe[key]
            candidate_rows.append(row)
            continue
        dedupe[key] = len(candidate_rows)
        row["structure"] = SUPPORT.structural_audit(task, candidate, data)
        if not row["structure"]["pass"]:
            row["classification"] = "REJECT_STRUCTURE"
        else:
            profile = SUPPORT.official_profile(
                task, candidate,
                f"binary291_{row['op_type']}_{row['node_index']}_{row['operand_index']}",
            )
            row["official_profile"] = profile
            if profile is None:
                row["classification"] = "REJECT_UNSCORABLE"
            else:
                actual_cost = int(profile["cost"])
                row["cost_reduction"] = row["authority_cost"] - actual_cost
                row["projected_gain"] = (
                    math.log(row["authority_cost"] / actual_cost) if actual_cost > 0 else None
                )
                if actual_cost >= row["authority_cost"]:
                    row["classification"] = "REJECT_NOT_STRICT_LOWER_ACTUAL_COST"
                else:
                    row["classification"] = "QUALIFIED_STRICT_LOWER_STRUCTURE"
                    candidate_data[row["candidate_sha256"]] = data
        candidate_rows.append(row)
        if variant_index % 100 == 0 or row["classification"] == "QUALIFIED_STRICT_LOWER_STRUCTURE":
            print(json.dumps({
                "variant": variant_index,
                "total": len(variants),
                "task": task,
                "op": row["op_type"],
                "operand": row["operand_index"],
                "classification": row["classification"],
                "actual_cost": (row.get("official_profile") or {}).get("cost"),
                "authority_cost": row["authority_cost"],
            }), flush=True)

    strict_rows = [
        row for row in candidate_rows
        if row["classification"] == "QUALIFIED_STRICT_LOWER_STRUCTURE"
    ]
    known_cache: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for index, row in enumerate(strict_rows, start=1):
        task = int(row["task"])
        if task not in known_cache:
            known_cache[task] = SUPPORT.known_cases(task)
        cases, counts = known_cache[task]
        row["known_counts"] = counts
        row["known_four"] = SUPPORT.evaluate_four(
            candidate_data[row["candidate_sha256"]], cases
        )
        row["known_pass"] = bool(
            counts["all_cases_convertible"] and SUPPORT.four_pass(row["known_four"])
        )
        row["classification"] = "PASS_KNOWN_FOUR" if row["known_pass"] else "REJECT_KNOWN_FOUR"
        print(json.dumps({
            "known": index,
            "total": len(strict_rows),
            "task": task,
            "op": row["op_type"],
            "operand": row["operand_index"],
            "cost": row["official_profile"]["cost"],
            "pass": row["known_pass"],
            "accuracy": {name: item["accuracy"] for name, item in row["known_four"].items()},
        }), flush=True)

    known_pass_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        if row.get("known_pass"):
            known_pass_by_task[int(row["task"])].append(row)
    for rows in known_pass_by_task.values():
        rows.sort(key=lambda row: (
            int(row["official_profile"]["cost"]), row["candidate_sha256"],
        ))

    fresh_audits = []
    reported = []
    for task in sorted(known_pass_by_task):
        seeds = (291_000_000 + task, 291_100_000 + task)
        generated = [SUPPORT.fresh_cases(task, seed, task_map) for seed in seeds]
        task_pass = None
        for candidate_index, row in enumerate(known_pass_by_task[task]):
            seed_rows = []
            for cases, generation in generated:
                four = SUPPORT.evaluate_four(
                    candidate_data[row["candidate_sha256"]], cases
                )
                passed = bool(
                    fresh_generation_clean(generation) and SUPPORT.four_pass(four)
                )
                seed_rows.append({
                    "generation": generation,
                    "runtime": four,
                    "generation_clean": fresh_generation_clean(generation),
                    "pass": passed,
                })
                print(json.dumps({
                    "fresh_task": task,
                    "seed": generation["seed"],
                    "candidate_cost": row["official_profile"]["cost"],
                    "pass": passed,
                    "accuracy": {name: item["accuracy"] for name, item in four.items()},
                }), flush=True)
            passed = all(item["pass"] for item in seed_rows)
            audit = {
                "task": task,
                "candidate_sha256": row["candidate_sha256"],
                "candidate_cost": row["official_profile"]["cost"],
                "authority_cost": row["authority_cost"],
                "fresh_pass": passed,
                "seeds": list(seeds),
                "count_per_seed": FRESH_PER_SEED,
                "runs": seed_rows,
            }
            fresh_audits.append(audit)
            row["fresh_audit"] = audit
            row["fresh_pass"] = passed
            row["classification"] = "PASS_FRESH_FOUR" if passed else "REJECT_FRESH_FOUR"
            if passed:
                task_pass = row
                for skipped in known_pass_by_task[task][candidate_index + 1:]:
                    skipped["classification"] = "NOT_FRESH_TESTED_MORE_EXPENSIVE_AFTER_TASK_PASS"
                break
        if task_pass is None:
            continue
        filename = (
            f"task{task:03d}_{task_pass['op_type'].lower()}{task_pass['node_index']:03d}_"
            f"operand{task_pass['operand_index']}_cost{task_pass['official_profile']['cost']}.onnx"
        )
        path = CANDIDATES_DIR / filename
        path.write_bytes(candidate_data[task_pass["candidate_sha256"]])
        if digest(path) != task_pass["candidate_sha256"]:
            raise RuntimeError("saved candidate SHA mismatch")
        task_pass["saved_path"] = rel(path)
        reported.append({
            "task": task,
            "path": rel(path),
            "sha256": task_pass["candidate_sha256"],
            "cost": task_pass["official_profile"]["cost"],
            "authority_cost": task_pass["authority_cost"],
            "cost_reduction": task_pass["cost_reduction"],
            "projected_gain": task_pass["projected_gain"],
            "op_type": task_pass["op_type"],
            "node_index": task_pass["node_index"],
            "node_name": task_pass["node_name"],
            "node_output": task_pass["node_output"],
            "operand_index": task_pass["operand_index"],
            "operand_value": task_pass["operand_value"],
            "policy_class": "POLICY90_APPROXIMATE_NOT_EXACT",
            "claims_full_correctness": False,
            "one_cheapest_complete_pass_for_task": True,
        })

    classifications = Counter(row["classification"] for row in candidate_rows)
    payload = {
        "lane": "agent_binary_ablation_scan_291",
        "decision": "PASS_POLICY90_CANDIDATES_FOUND" if reported else "NO_PASSING_BINARY_ABLATION",
        "authority": {
            "zip": rel(AUTHORITY),
            "sha256": AUTHORITY_SHA256,
            "task_count": 400,
            "cost_census": rel(COST_CENSUS),
            "cost_census_sha256": digest(COST_CENSUS),
        },
        "policy": {
            "class": "NORMAL_POLICY90_APPROXIMATE_NOT_EXACT",
            "claims_full_correctness": False,
            "threshold": POLICY_THRESHOLD,
            "target_ops": sorted(TARGET_OPS),
            "priority_cost_range": [PRIORITY_COST_MIN, PRIORITY_COST_MAX],
            "fresh_per_seed": FRESH_PER_SEED,
            "fresh_seed_formula": ["291000000 + task", "291100000 + task"],
            "configs": [
                {
                    "name": name,
                    "optimization": "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL",
                    "threads": threads,
                }
                for name, disable, threads in SUPPORT.CONFIGS
            ],
            "operand_output_shape_dtype_exact_only": True,
            "onnxoptimizer_passes": OPTIMIZER_PASSES,
            "strict_lower_actual_cost_only": True,
            "full_strict_standard_static_runtime_trace_required": True,
            "conv_bias_ub_allowed": False,
            "lookup_giant_shape_cloak_allowed": False,
            "root_submission_scores_others71407_written": False,
            "automatic_promotion": False,
            "kimi_used": False,
        },
        "exclusions": {
            "active_manifest": rel(ACTIVE_MANIFEST),
            "active_manifest_sha256_at_run": digest(ACTIVE_MANIFEST),
            "assigned_active23_count": len(ASSIGNED_ACTIVE23),
            "assigned_active23_tasks": sorted(ASSIGNED_ACTIVE23),
            "observed_active_count_at_run": len(observed_active),
            "observed_active_tasks_at_run": sorted(observed_active),
            "concurrent_active_additions": sorted(concurrent_active),
            "private_zero_or_unsound_monitor": sorted(PRIVATE_ZERO_OR_UNSOUND),
            "excluded_union_count": len(excluded_tasks),
        },
        "coverage": {
            "authority_tasks": 400,
            "eligible_authority_tasks": sum(not row["excluded"] for row in authority_inventory),
            "priority_authority_tasks_total": sum(row["priority_cost_150_to_500"] for row in authority_inventory),
            "priority_eligible_authority_tasks": sum(
                row["priority_cost_150_to_500"] and not row["excluded"]
                for row in authority_inventory
            ),
            "authorities_with_target_nodes": sum(row["target_nodes"] > 0 for row in authority_inventory),
            "target_nodes": sum(row["target_nodes"] for row in authority_inventory),
            "target_operands": sum(row["operands"] for row in authority_inventory),
            "shape_dtype_exact_operand_variants": len(variants),
            "priority_shape_dtype_exact_variants": sum(row["priority_cost_150_to_500"] for row in variants),
            "deduplicated_candidate_count": len(dedupe),
            "strict_lower_structure_count": len(strict_rows),
            "known_four_pass_count": sum(bool(row.get("known_pass")) for row in strict_rows),
            "known_four_pass_tasks": sorted(known_pass_by_task),
            "fresh_audited_candidate_count": len(fresh_audits),
            "reported_task_count": len(reported),
            "reported_tasks": [row["task"] for row in reported],
            "one_cheapest_report_per_task": True,
        },
        "classification_counts": dict(classifications),
        "authority_inventory": authority_inventory,
        "candidate_rows": candidate_rows,
        "fresh_audits": fresh_audits,
        "reported_candidates": reported,
        "aggregate": {
            "known_case_config_executions": int(sum(
                item["total"]
                for row in strict_rows for item in row.get("known_four", {}).values()
            )),
            "fresh_case_config_executions": int(sum(
                item["total"] for audit in fresh_audits for run in audit["runs"]
                for item in run["runtime"].values()
            )),
            "elapsed_seconds": time.monotonic() - started,
        },
        "protected_writes": "none; only scripts/golf/agent_binary_ablation_scan_291",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "coverage": payload["coverage"],
        "classifications": payload["classification_counts"],
        "reported": reported,
        "elapsed_seconds": payload["aggregate"]["elapsed_seconds"],
        "evidence": rel(EVIDENCE),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
