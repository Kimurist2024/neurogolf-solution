#!/usr/bin/env python3
"""Fail-closed safe cost-reduction scan for the 17 non-score25 low-cost tasks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
AUTHORITY = ROOT / "submission_base_8010.03.zip"
AUTHORITY_SHA256 = "d772399d4535176b95039690eca59808059add3c0ca2d42e2124f17c705ec2e6"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
ALL_LOWCOST_TASKS = (
    16, 17, 53, 61, 67, 87, 129, 135, 140, 179, 197,
    223, 241, 276, 305, 307, 309, 312, 326, 337, 373,
)
ALREADY_SCORE25 = {67, 129, 179, 241}
SCAN_TASKS = tuple(task for task in ALL_LOWCOST_TASKS if task not in ALREADY_SCORE25)
EXPECTED_COSTS = {
    16: 10, 17: 10, 53: 6, 61: 10, 67: 0, 87: 5,
    129: 1, 135: 2, 140: 5, 179: 0, 197: 10, 223: 5,
    241: 0, 276: 10, 305: 10, 307: 4, 309: 10, 312: 10,
    326: 4, 337: 10, 373: 8,
}
EXPECTED_IO = [1, 10, 30, 30]
FRESH_PER_SEED = 2_000
OPTIMIZER_PASSES = ["eliminate_deadend", "eliminate_unused_initializer"]
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("lowcost293_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load support: {SUPPORT_PATH}")
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


def make_template(label: str) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_IO)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_IO)
    if label == "identity":
        node = helper.make_node("Identity", ["input"], ["output"])
    elif label == "add_self":
        node = helper.make_node("Add", ["input", "input"], ["output"])
    elif label == "mul_self":
        node = helper.make_node("Mul", ["input", "input"], ["output"])
    elif label == "transpose_identity":
        node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 2, 3])
    elif label == "transpose_hw":
        node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])
    elif label == "einsum_identity":
        node = helper.make_node("Einsum", ["input"], ["output"], equation="bkrc->bkrc")
    elif label == "einsum_transpose_hw":
        node = helper.make_node("Einsum", ["input"], ["output"], equation="bkrc->bkcr")
    elif label == "einsum_square":
        node = helper.make_node(
            "Einsum", ["input", "input"], ["output"], equation="bkrc,bkrc->bkrc"
        )
    elif label == "einsum_positive_scale_identity":
        node = helper.make_node(
            "Einsum", ["input", "input"], ["output"], equation="bkrc,bjhw->bkrc"
        )
    elif label == "einsum_positive_scale_transpose":
        node = helper.make_node(
            "Einsum", ["input", "input"], ["output"], equation="bkrc,bjhw->bkcr"
        )
    elif label == "einsum_identity_transpose_intersection":
        node = helper.make_node(
            "Einsum", ["input", "input"], ["output"], equation="bkrc,bkcr->bkrc"
        )
    else:
        raise ValueError(label)
    graph = helper.make_graph([node], f"lowcost293_{label}", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="lowcost25_scan_293",
    )
    model.ir_version = 8
    return model


TEMPLATE_LABELS = (
    "identity",
    "add_self",
    "mul_self",
    "transpose_identity",
    "transpose_hw",
    "einsum_identity",
    "einsum_transpose_hw",
    "einsum_square",
    "einsum_positive_scale_identity",
    "einsum_positive_scale_transpose",
    "einsum_identity_transpose_intersection",
)


def make_graph_model(
    name: str,
    nodes: list[onnx.NodeProto],
    opset: int,
    value_info: list[onnx.ValueInfoProto] | None = None,
) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_IO)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_IO)
    graph = helper.make_graph(
        nodes, name, [inp], [out], value_info=value_info or [],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)],
        producer_name="lowcost25_scan_293_specialized",
    )
    model.ir_version = 8
    return model


def specialized_models(task: int) -> list[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    rows: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    if task == 53:
        # One-row downward shift. This is the only single-node/zero-param form;
        # exact evaluation also checks the one-hot background-channel boundary.
        node = helper.make_node(
            "Pad", ["input"], ["output"], mode="constant", value=0.0,
            pads=[0, 0, 1, 0, 0, 0, -1, 0],
        )
        rows.append((
            "legacy_pad_shift_down_1",
            make_graph_model("task053_legacy_pad", [node], 10),
            {
                "kind": "specialized_legacy_pad_v10",
                "pads": [0, 0, 1, 0, 0, 0, -1, 0],
                "risk_checked": "new top row is scalar-zero across channels, including background",
            },
        ))
    elif task == 135:
        crop = helper.make_node(
            "Slice", ["input"], ["crop"],
            starts=[0, 0, 0, 6], ends=[1, 10, 3, 9], axes=[0, 1, 2, 3],
        )
        pad = helper.make_node(
            "Pad", ["crop"], ["output"], mode="constant", value=0.0,
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
        )
        rows.append((
            "legacy_slice_top_right3_pad",
            make_graph_model(
                "task135_slice_pad", [crop, pad], 9,
                [helper.make_tensor_value_info("crop", TensorProto.FLOAT, [1, 10, 3, 3])],
            ),
            {
                "kind": "specialized_slice_pad",
                "crop": "rows[0:3], cols[6:9]",
                "expected_intermediate_bytes_floor": 1 * 10 * 3 * 3 * 4,
            },
        ))
    elif task == 326:
        crop = helper.make_node(
            "Slice", ["input"], ["crop"],
            starts=[0, 0, 0, 0], ends=[1, 10, 2, 2], axes=[0, 1, 2, 3],
        )
        pad = helper.make_node(
            "Pad", ["crop"], ["output"], mode="constant", value=0.0,
            pads=[0, 0, 0, 0, 0, 0, 28, 28],
        )
        rows.append((
            "legacy_slice_top_left2_pad",
            make_graph_model(
                "task326_slice_pad", [crop, pad], 9,
                [helper.make_tensor_value_info("crop", TensorProto.FLOAT, [1, 10, 2, 2])],
            ),
            {
                "kind": "specialized_slice_pad",
                "crop": "rows[0:2], cols[0:2]",
                "expected_intermediate_bytes_floor": 1 * 10 * 2 * 2 * 4,
            },
        ))
    elif task == 307:
        upsample = helper.make_node(
            "Upsample", ["input"], ["up60"], mode="nearest",
            scales=[1.0, 1.0, 2.0, 2.0],
        )
        crop = helper.make_node(
            "Slice", ["up60"], ["output"],
            starts=[0, 0, 0, 0], ends=[1, 10, 30, 30], axes=[0, 1, 2, 3],
        )
        rows.append((
            "legacy_upsample2_slice30",
            make_graph_model(
                "task307_upsample_slice", [upsample, crop], 7,
                [helper.make_tensor_value_info("up60", TensorProto.FLOAT, [1, 10, 60, 60])],
            ),
            {
                "kind": "specialized_upsample_slice",
                "scale": [1.0, 1.0, 2.0, 2.0],
                "expected_intermediate_bytes_floor": 1 * 10 * 60 * 60 * 4,
            },
        ))
    return rows


def strip_einsum_initializer_operands(
    authority: onnx.ModelProto,
) -> tuple[onnx.ModelProto | None, dict[str, Any]]:
    if len(authority.graph.node) != 1 or authority.graph.node[0].op_type != "Einsum":
        return None, {"applicable": False, "reason": "authority_not_single_einsum"}
    node = authority.graph.node[0]
    equation = None
    for attribute in node.attribute:
        if attribute.name == "equation":
            equation = helper.get_attribute_value(attribute)
            if isinstance(equation, bytes):
                equation = equation.decode("ascii")
    if not equation or "->" not in equation:
        return None, {"applicable": False, "reason": "missing_explicit_equation"}
    left, right = equation.split("->", 1)
    terms = left.split(",")
    if len(terms) != len(node.input):
        return None, {"applicable": False, "reason": "equation_input_arity_mismatch"}
    initializer_names = {item.name for item in authority.graph.initializer}
    kept = [
        (name, term) for name, term in zip(node.input, terms)
        if name not in initializer_names
    ]
    removed = [
        {"input": name, "term": term} for name, term in zip(node.input, terms)
        if name in initializer_names
    ]
    if not removed:
        return None, {"applicable": True, "reason": "no_initializer_operands"}
    if not kept:
        return None, {"applicable": True, "reason": "all_operands_were_initializers"}
    new_equation = ",".join(term for _, term in kept) + "->" + right
    model = copy.deepcopy(authority)
    replacement = helper.make_node(
        "Einsum", [name for name, _ in kept], list(node.output),
        equation=new_equation, name="strip_initializer_operands",
    )
    model.graph.node[0].CopyFrom(replacement)
    del model.graph.initializer[:]
    model = onnxoptimizer.optimize(model, OPTIMIZER_PASSES)
    return model, {
        "applicable": True,
        "reason": "candidate_built",
        "original_equation": equation,
        "new_equation": new_equation,
        "removed_initializer_operands": removed,
        "kept_operand_count": len(kept),
    }


def unused_initializer_cleanup(
    authority: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    before = [item.name for item in authority.graph.initializer]
    model = onnxoptimizer.optimize(copy.deepcopy(authority), OPTIMIZER_PASSES)
    after = [item.name for item in model.graph.initializer]
    return model, {
        "before": before,
        "after": after,
        "removed": sorted(set(before) - set(after)),
    }


def runtime_exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("total", -1) >= 0
        and row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("accuracy") == 1.0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def exact_four(rows: dict[str, Any]) -> bool:
    return bool(len(rows) == 4 and all(runtime_exact(row) for row in rows.values()))


def fresh_generation_clean(row: dict[str, Any]) -> bool:
    return bool(
        row.get("accepted") == FRESH_PER_SEED
        and row.get("generation_errors") == 0
        and row.get("conversion_skips") == 0
    )


def evaluate_exact(
    task: int,
    data: bytes,
    task_map: dict[str, str],
    run_fresh: bool,
) -> dict[str, Any]:
    cases, counts = SUPPORT.known_cases(task)
    known_four = SUPPORT.evaluate_four(data, cases)
    known_pass = bool(counts["all_cases_convertible"] and exact_four(known_four))
    result: dict[str, Any] = {
        "known_counts": counts,
        "known_four": known_four,
        "known_exact_pass": known_pass,
        "fresh_requested": bool(run_fresh and known_pass),
    }
    if not run_fresh or not known_pass:
        return result
    seed_rows = []
    for seed in (293_000_000 + task, 293_100_000 + task):
        fresh, generation = SUPPORT.fresh_cases(task, seed, task_map)
        four = SUPPORT.evaluate_four(data, fresh)
        passed = bool(fresh_generation_clean(generation) and exact_four(four))
        seed_rows.append({
            "generation": generation,
            "runtime": four,
            "generation_clean": fresh_generation_clean(generation),
            "exact_pass": passed,
        })
        print(json.dumps({
            "fresh_task": task,
            "seed": seed,
            "pass": passed,
            "right": {name: row["right"] for name, row in four.items()},
        }), flush=True)
    result["fresh"] = {
        "seeds": [293_000_000 + task, 293_100_000 + task],
        "count_per_seed": FRESH_PER_SEED,
        "runs": seed_rows,
        "exact_pass": all(row["exact_pass"] for row in seed_rows),
    }
    return result


def main() -> int:
    started = time.monotonic()
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("latest 8010.03 authority SHA mismatch")
    if (
        len(ALL_LOWCOST_TASKS) != 21
        or len(SCAN_TASKS) != 17
        or set(ALL_LOWCOST_TASKS) != set(EXPECTED_COSTS)
    ):
        raise RuntimeError("assigned low-cost inventory malformed")
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    if len(task_map) != 400:
        raise RuntimeError("task map incomplete")
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    authority_models: dict[int, onnx.ModelProto] = {}
    authority_data: dict[int, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in SCAN_TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            authority_data[task] = data
            authority_models[task] = model
            profile = SUPPORT.official_profile(task, model, "authority801003")
            if profile is None or int(profile["cost"]) != EXPECTED_COSTS[task]:
                raise RuntimeError(
                    f"task{task:03d} latest cost mismatch: {profile} != {EXPECTED_COSTS[task]}"
                )
            structure = SUPPORT.structural_audit(task, model, data)
            inventory.append({
                "task": task,
                "authority_member": f"task{task:03d}.onnx",
                "authority_sha256": sha256(data),
                "authority_file_bytes": len(data),
                "authority_profile": profile,
                "authority_structure": structure,
                "private_zero_catalog": task in PRIVATE_ZERO_OR_UNSOUND,
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "initializer_elements": int(SUPPORT.scoring.calculate_params(model) or 0),
                "ops": [node.op_type for node in model.graph.node],
            })
            print(json.dumps({
                "authority": task,
                "cost": profile["cost"],
                "sha256": sha256(data)[:16],
                "structure_pass": structure["pass"],
            }), flush=True)

    attempts = []
    candidate_rows = []
    candidate_data: dict[str, bytes] = {}
    seen: set[tuple[int, str]] = set()
    template_models = {label: make_template(label) for label in TEMPLATE_LABELS}
    for task in sorted(SCAN_TASKS):
        authority = authority_models[task]
        cleanup, cleanup_meta = unused_initializer_cleanup(authority)
        cleanup_data = cleanup.SerializeToString()
        attempts.append({
            "task": task,
            "kind": "authority_unused_initializer_cleanup",
            "metadata": cleanup_meta,
            "changed": cleanup_data != authority_data[task],
            "candidate_sha256": sha256(cleanup_data),
        })
        sources: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = [
            (label, copy.deepcopy(model), {"kind": "no_param_template"})
            for label, model in template_models.items()
        ]
        sources.extend(specialized_models(task))
        stripped, strip_meta = strip_einsum_initializer_operands(authority)
        attempts.append({
            "task": task,
            "kind": "strip_einsum_initializer_operands",
            "metadata": strip_meta,
            "candidate_built": stripped is not None,
        })
        if stripped is not None:
            sources.append(("strip_einsum_initializer_operands", stripped, strip_meta))
        if cleanup_data != authority_data[task]:
            sources.append(("authority_unused_initializer_cleanup", cleanup, cleanup_meta))

        for rank, (label, model, transform_meta) in enumerate(sources):
            data = model.SerializeToString()
            candidate_sha = sha256(data)
            row: dict[str, Any] = {
                "task": task,
                "label": label,
                "template_rank": rank,
                "transform": transform_meta,
                "authority_cost": EXPECTED_COSTS[task],
                "authority_sha256": sha256(authority_data[task]),
                "candidate_sha256": candidate_sha,
                "candidate_file_bytes": len(data),
            }
            key = (task, candidate_sha)
            if key in seen:
                row["classification"] = "REJECT_DUPLICATE_SERIALIZATION"
                candidate_rows.append(row)
                continue
            seen.add(key)
            structure = SUPPORT.structural_audit(task, model, data)
            row["structure"] = structure
            if not structure["pass"]:
                row["classification"] = "REJECT_STRUCTURE_FAIL_CLOSED"
                candidate_rows.append(row)
                continue
            profile = SUPPORT.official_profile(task, model, f"low293_{label}")
            row["official_profile"] = profile
            if profile is None:
                row["classification"] = "REJECT_UNSCORABLE"
                candidate_rows.append(row)
                continue
            candidate_cost = int(profile["cost"])
            row["strict_lower_actual"] = candidate_cost < EXPECTED_COSTS[task]
            row["reaches_cost_zero_or_one"] = candidate_cost in (0, 1)
            if not row["strict_lower_actual"]:
                row["classification"] = "REJECT_NOT_STRICT_LOWER_ACTUAL"
            else:
                row["classification"] = "QUALIFIED_STRICT_LOWER_PRE_KNOWN"
                candidate_data[candidate_sha] = data
            candidate_rows.append(row)

    qualified = [
        row for row in candidate_rows
        if row["classification"] == "QUALIFIED_STRICT_LOWER_PRE_KNOWN"
    ]
    known_cache: dict[tuple[int, str], dict[str, Any]] = {}
    for index, row in enumerate(qualified, start=1):
        task = int(row["task"])
        result = evaluate_exact(
            task, candidate_data[row["candidate_sha256"]], task_map, run_fresh=False
        )
        known_cache[(task, row["candidate_sha256"])] = result
        row["known"] = result
        row["classification"] = (
            "PASS_KNOWN_EXACT_PRE_FRESH"
            if result["known_exact_pass"] else "REJECT_KNOWN_NOT_EXACT"
        )
        print(json.dumps({
            "known": index,
            "total": len(qualified),
            "task": task,
            "label": row["label"],
            "pass": result["known_exact_pass"],
            "right": {name: item["right"] for name, item in result["known_four"].items()},
        }), flush=True)

    known_pass_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in qualified:
        if row["classification"] == "PASS_KNOWN_EXACT_PRE_FRESH":
            known_pass_by_task[int(row["task"])].append(row)
    for rows in known_pass_by_task.values():
        rows.sort(key=lambda row: (
            int(row["official_profile"]["cost"]), row["template_rank"], row["candidate_sha256"],
        ))

    for task, rows in sorted(known_pass_by_task.items()):
        for row in rows:
            result = evaluate_exact(
                task, candidate_data[row["candidate_sha256"]], task_map, run_fresh=True
            )
            # Preserve the already-recorded known result and add fresh evidence.
            row["fresh"] = result.get("fresh")
            passed = bool(result["known_exact_pass"] and result.get("fresh", {}).get("exact_pass"))
            row["classification"] = "PASS_SAFE_IMPROVEMENT_EXACT" if passed else "REJECT_FRESH_NOT_EXACT"

    verified_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in qualified:
        if row["classification"] == "PASS_SAFE_IMPROVEMENT_EXACT":
            verified_by_task[int(row["task"])].append(row)
    reported = []
    for task, rows in sorted(verified_by_task.items()):
        rows.sort(key=lambda row: (
            int(row["official_profile"]["cost"]), row["template_rank"], row["candidate_sha256"],
        ))
        winner = rows[0]
        cost = int(winner["official_profile"]["cost"])
        filename = f"task{task:03d}_{winner['label']}_cost{cost}.onnx"
        path = CANDIDATES_DIR / filename
        path.write_bytes(candidate_data[winner["candidate_sha256"]])
        if digest(path) != winner["candidate_sha256"]:
            raise RuntimeError("saved candidate hash mismatch")
        winner["saved_path"] = rel(path)
        winner["selected_for_task"] = True
        reported.append({
            "task": task,
            "path": rel(path),
            "sha256": winner["candidate_sha256"],
            "label": winner["label"],
            "cost": cost,
            "authority_cost": winner["authority_cost"],
            "strict_lower_actual": True,
            "known_and_fresh_exact_four_configs": True,
            "score": winner["official_profile"]["score"],
            "score25": cost in (0, 1),
            "safe_score_improvement": True,
        })

    task_results = []
    for item in inventory:
        task = int(item["task"])
        passes = verified_by_task.get(task, [])
        task_rows = [row for row in candidate_rows if row["task"] == task]
        if passes:
            best = sorted(passes, key=lambda row: (
                int(row["official_profile"]["cost"]), row["template_rank"], row["candidate_sha256"],
            ))[0]
            status = "NEW_SAFE_COST_IMPROVEMENT"
            reason = f"{best['label']} reached actual cost {best['official_profile']['cost']} and passed exact known/fresh"
        else:
            known_passes = sum(
                row["classification"] in {
                    "REJECT_FRESH_NOT_EXACT", "PASS_SAFE_IMPROVEMENT_EXACT",
                }
                for row in task_rows
            )
            if known_passes:
                reason = "one or more strict-lower candidates passed known exactly but failed fresh exactness"
            else:
                best_accuracy = max(
                    (
                        min(run["accuracy"] for run in row.get("known", {}).get("known_four", {}).values())
                        for row in task_rows if row.get("known", {}).get("known_four")
                    ),
                    default=0.0,
                )
                reason = f"no fail-closed strict-lower candidate was exact on all known cases; best four-config accuracy={best_accuracy:.6f}"
            status = "NO_VERIFIED_SAFE_COST_IMPROVEMENT"
        task_results.append({
            "task": task,
            "authority_cost": EXPECTED_COSTS[task],
            "authority_sha256": item["authority_sha256"],
            "status": status,
            "reason": reason,
            "candidate_attempts": len(task_rows),
            "known_exact_candidates": sum(
                row["classification"] in {"REJECT_FRESH_NOT_EXACT", "PASS_SAFE_IMPROVEMENT_EXACT"}
                for row in task_rows
            ),
            "fresh_exact_candidates": len(passes),
        })

    classifications = Counter(row["classification"] for row in candidate_rows)
    payload = {
        "lane": "lowcost25_scan_293",
        "decision": "SAFE_COST_IMPROVEMENTS_FOUND" if reported else "NO_SAFE_COST_IMPROVEMENT",
        "authority": {
            "zip": rel(AUTHORITY),
            "sha256": AUTHORITY_SHA256,
            "leaderboard": 8010.03,
            "all_cost_le_10_task_count": len(ALL_LOWCOST_TASKS),
            "skipped_existing_score25_tasks": sorted(ALREADY_SCORE25),
            "scanned_task_count": len(SCAN_TASKS),
            "scanned_tasks": list(SCAN_TASKS),
        },
        "policy": {
            "goal": "any safe strict-lower actual cost / score improvement",
            "requires_strict_lower_actual_cost": True,
            "requires_case_exact_accuracy": 1.0,
            "fresh_count_per_seed": FRESH_PER_SEED,
            "fresh_seed_formula": ["293000000 + task", "293100000 + task"],
            "configs": [
                {
                    "name": name,
                    "optimization": "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL",
                    "threads": threads,
                }
                for name, disable, threads in SUPPORT.CONFIGS
            ],
            "templates": list(TEMPLATE_LABELS),
            "specialized_templates": {
                "task053": "legacy Pad v10 one-row shift",
                "task135": "legacy attribute Slice top-right 3x3 + Pad",
                "task307": "legacy Upsample 2x + attribute Slice to 30x30",
                "task326": "legacy attribute Slice top-left 2x2 + Pad",
            },
            "initializer_transforms": [
                "eliminate_deadend", "eliminate_unused_initializer",
                "strip_einsum_initializer_operands_and_equation_terms",
            ],
            "private_zero_lookup_giant_nan_ub_shape_cloak": "fail_closed",
            "automatic_promotion": False,
            "kimi_used": False,
        },
        "coverage": {
            "authority_reprofiled": len(inventory),
            "authority_costs_match_expected": all(
                int(row["authority_profile"]["cost"]) == EXPECTED_COSTS[row["task"]]
                for row in inventory
            ),
            "existing_score25_skipped": len(ALREADY_SCORE25),
            "remaining_tasks_scanned": len(SCAN_TASKS),
            "candidate_rows": len(candidate_rows),
            "qualified_strict_lower_candidates": len(qualified),
            "known_exact_candidates": sum(
                row["classification"] in {"REJECT_FRESH_NOT_EXACT", "PASS_SAFE_IMPROVEMENT_EXACT"}
                for row in candidate_rows
            ),
            "fresh_exact_candidates": sum(
                row["classification"] == "PASS_SAFE_IMPROVEMENT_EXACT" for row in candidate_rows
            ),
            "safe_improvement_tasks": [row["task"] for row in reported],
            "safe_improvement_task_count": len(reported),
        },
        "classification_counts": dict(classifications),
        "authority_inventory": inventory,
        "transform_attempts": attempts,
        "candidate_rows": candidate_rows,
        "task_results": task_results,
        "reported_candidates": reported,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only scripts/golf/lowcost25_scan_293",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "coverage": payload["coverage"],
        "classifications": payload["classification_counts"],
        "reported": reported,
        "elapsed_seconds": payload["elapsed_seconds"],
        "evidence": rel(EVIDENCE),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
