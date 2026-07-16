#!/usr/bin/env python3
"""Worker 3: remove current-graph nodes/optional operands and revalidate."""

from __future__ import annotations

import copy
import json
import tempfile
import time
import zipfile
from collections import Counter
from typing import Any

import onnx
from onnx import helper

import common


OUT = common.HERE / "mutation_scan.json"


def prune_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]


def serialize_source(model: onnx.ModelProto, name: str, family: str, detail: str) -> dict[str, Any]:
    return common.PATTERN.source(model, name, family, detail)


def bypass_variants(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    """Bypass one node to each non-constant input; checker/known gate decides legality."""
    init_names = {item.name for item in base.graph.initializer}
    result = []
    for index, node in enumerate(base.graph.node):
        if len(node.output) != 1 or not node.output[0]:
            continue
        old = node.output[0]
        for slot, source in enumerate(node.input):
            if not source or source in init_names or source == old:
                continue
            model = copy.deepcopy(base)
            target = model.graph.node[index]
            del model.graph.node[index]
            is_output = any(value.name == old for value in model.graph.output)
            for consumer in model.graph.node:
                for position, name in enumerate(consumer.input):
                    if name == old:
                        consumer.input[position] = source
            if is_output:
                # Preserve the canonical output name.  Identity itself is free
                # because its only result is the graph output.
                model.graph.node.extend([helper.make_node("Identity", [source], [old], name=f"bypass_{index}_{slot}")])
            prune_initializers(model)
            result.append(serialize_source(
                model, f"task{task:03d}_bypass_n{index}_i{slot}", "single_node_bypass",
                f"remove {node.op_type} and route dynamic input {slot}",
            ))
    return result


def optional_operand_variants(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    result = []
    optional_last = {"Conv": 2, "ConvTranspose": 2, "Gemm": 2, "Clip": 1}
    init_names = {item.name for item in base.graph.initializer}
    for index, node in enumerate(base.graph.node):
        minimum_index = optional_last.get(node.op_type)
        if minimum_index is None:
            continue
        for slot in range(len(node.input) - 1, minimum_index - 1, -1):
            if not node.input[slot] or node.input[slot] not in init_names:
                continue
            model = copy.deepcopy(base)
            target = model.graph.node[index]
            # Only trailing optionals can be omitted without an empty placeholder.
            if slot != len(target.input) - 1:
                continue
            del target.input[slot]
            prune_initializers(model)
            result.append(serialize_source(
                model, f"task{task:03d}_drop_{node.op_type}_optional_{index}_{slot}",
                "optional_initializer_ablation", f"remove trailing initializer operand {slot}",
            ))
    return result


def main() -> int:
    started = time.monotonic()
    common.HERE.mkdir(parents=True, exist_ok=True)
    common.CANDIDATES.mkdir(parents=True, exist_ok=True)
    common.validate_authority()
    p = common.PATTERN
    costs = common.current_costs(101, 166)
    cases = {task: p.known_cases(task) for task in costs}
    report = {
        "authority": str(common.AUTHORITY.relative_to(common.ROOT)),
        "authority_sha256": common.AUTHORITY_SHA256,
        "authority_diff": common.authority_diff(),
        "scope": "current authority graph surgery for every task at cost 101..166",
        "scope_task_count": len(costs), "tasks": [], "finalists": [], "counters": {},
    }
    counters: Counter[str] = Counter()
    with zipfile.ZipFile(common.AUTHORITY) as archive:
        for task in sorted(costs, key=lambda value: (costs[value], value)):
            base = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            variants = [*bypass_variants(task, base), *optional_operand_variants(task, base)]
            variants.extend(p.einsum_initializer_subsets(task, base))
            variants.extend(p.conv_crop_variants(task, base))
            variants = p.unique_sources(variants)
            task_row = {
                "task": task, "authority_cost": costs[task], "half_target": costs[task] // 2,
                "known_count": len(cases[task]), "variant_count": len(variants),
                "session_ok": 0, "quick_exact": 0, "known_exact": 0,
                "strict_lower": 0, "half_cost": 0, "survivors": [],
            }
            for row in variants:
                counters["candidate_task_evaluations"] += 1
                runtime = p.make_session(row["_model"])
                if runtime is None:
                    counters["session_reject"] += 1
                    continue
                task_row["session_ok"] += 1
                quick = p.evaluate(runtime, cases[task], min(12, len(cases[task])))
                if not p.clean(quick):
                    continue
                task_row["quick_exact"] += 1
                full = p.evaluate(runtime, cases[task])
                if not p.clean(full):
                    continue
                task_row["known_exact"] += 1
                audit = p.structure(row["_model"])
                if not audit["pass"]:
                    counters["structural_reject"] += 1
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix=f"low405_mut_{task:03d}_", dir="/tmp") as work:
                        profile = p.scoring.score_and_verify(row["_model"], task, work, label="candidate", require_correct=True)
                except Exception:
                    profile = None
                if profile is None or int(profile["cost"]) >= costs[task]:
                    continue
                stable, margin = p.scoring.model_margin_stable(row["_model"], task)
                if not stable:
                    counters["margin_reject"] += 1
                    continue
                candidate_cost = int(profile["cost"])
                half = candidate_cost * 2 <= costs[task]
                path = common.CANDIDATES / f"task{task:03d}_mutation_cost{candidate_cost}_{row['sha256'][:12]}.onnx"
                path.write_bytes(row["_data"])
                finalist = {
                    "task": task, "name": row["name"], "family": row["family"],
                    "detail": row["detail"], "sha256": row["sha256"],
                    "authority_cost": costs[task], "candidate_cost": candidate_cost,
                    "half_target_met": half, "known": full, "structure": audit,
                    "margin_stable": bool(stable), "margin_min": margin,
                    "path": str(path.relative_to(common.ROOT)),
                }
                task_row["strict_lower"] += 1
                task_row["half_cost"] += int(half)
                task_row["survivors"].append(finalist)
                report["finalists"].append(finalist)
            report["tasks"].append(task_row)
            OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "task": task, "cost": costs[task], "variants": len(variants),
                "known": task_row["known_exact"], "strict": task_row["strict_lower"],
                "half": task_row["half_cost"],
            }), flush=True)
    report["counters"] = dict(counters)
    report["elapsed_seconds"] = time.monotonic() - started
    report["protected_writes"] = "lane only; authority/root/others untouched"
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finalists": len(report["finalists"]), "counters": report["counters"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

