#!/usr/bin/env python3
"""Search same-static-shape node bypasses with exact gold/fresh admission."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def import_support():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("same_shape_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.SUPPORT.FRESH_PER_SEED = 2_000
    return module


def descriptor(value: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]] | None:
    if not value.type.HasField("tensor_type"):
        return None
    tensor = value.type.tensor_type
    dims = tensor.shape.dim
    if any(dim.HasField("dim_param") or not dim.HasField("dim_value")
           or int(dim.dim_value) <= 0 for dim in dims):
        return None
    return int(tensor.elem_type), tuple(int(dim.dim_value) for dim in dims)


def descriptors(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    result = {
        value.name: item
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        if (item := descriptor(value)) is not None
    }
    result.update({item.name: (int(item.data_type), tuple(int(dim) for dim in item.dims))
                   for item in model.graph.initializer})
    return result


def variants(base: onnx.ModelProto):
    desc = descriptors(base)
    graph_outputs = {value.name for value in base.graph.output}
    initializer_names = {item.name for item in base.graph.initializer}
    for node_index, node in enumerate(base.graph.node):
        if len(node.output) != 1 or not node.output[0]:
            continue
        target = node.output[0]
        target_desc = desc.get(target)
        if target_desc is None:
            continue
        for input_index, source in enumerate(node.input):
            if not source or source == target or desc.get(source) != target_desc:
                continue
            model = copy.deepcopy(base)
            replacement = model.graph.node[node_index]
            if target in graph_outputs:
                replacement.CopyFrom(helper.make_node(
                    "Identity", [source], [target],
                    name=f"bypass_{node_index}_{input_index}",
                ))
            else:
                del model.graph.node[node_index]
                for consumer in model.graph.node:
                    for position, name in enumerate(consumer.input):
                        if name == target:
                            consumer.input[position] = source
                kept = [value for value in model.graph.value_info if value.name != target]
                del model.graph.value_info[:]
                model.graph.value_info.extend(kept)
            used = {name for item in model.graph.node for name in item.input if name}
            kept_init = [item for item in model.graph.initializer
                         if item.name in used or item.name not in initializer_names]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept_init)
            yield model, {
                "node_index": node_index,
                "op": node.op_type,
                "input_index": input_index,
                "source": source,
                "target": target,
                "descriptor": target_desc,
            }


def exact(row: dict[str, object]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0 and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def main() -> int:
    support = import_support()
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        tasks = sorted(
            int(row["task"].removeprefix("task"))
            for row in csv.DictReader(handle)
            if 201 <= int(row["cost"]) <= 350
            and int(row["task"].removeprefix("task")) not in EXCLUDED
        )
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "scope": "ledger cost 201..350 excluding known black/private-zero/failed-gold",
        "method": "bypass a node only through an equal dtype/static-shape input",
        "gold_required": True,
        "fresh_required": 2_000,
        "tasks": [], "finalists": [], "counters": {},
    }
    counters: Counter[str] = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            cases, counts = support.SUPPORT.known_cases(task)
            base_profile = support.POLICY.fast_profile(support.SUPPORT, task, base, cases[0])
            task_row = {
                "task": task, "authority_cost": int(base_profile["cost"]),
                "known_counts": counts, "attempted": 0,
                "known_exact": [], "fresh_rejects": [],
            }
            seen: set[str] = set()
            for model, meta in variants(base):
                counters["encounters"] += 1
                data = model.SerializeToString()
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    counters["duplicate"] += 1
                    continue
                seen.add(digest)
                task_row["attempted"] += 1
                try:
                    reasons = support.quick_preflight(model)
                except Exception as exc:
                    reasons = [f"preflight:{type(exc).__name__}:{exc}"]
                if reasons:
                    counters["preflight_reject"] += 1
                    continue
                profile = support.POLICY.fast_profile(support.SUPPORT, task, model, cases[0])
                if profile is None or int(profile["cost"]) >= int(base_profile["cost"]):
                    counters["cost_reject"] += 1
                    continue
                known = support.failfast_known(data, cases)
                if not exact(known):
                    counters["known_reject"] += 1
                    continue
                attempt = {
                    **meta, "sha256": digest,
                    "candidate_cost": int(profile["cost"]),
                    "gain": math.log(int(base_profile["cost"]) / int(profile["cost"])),
                    "known": support.compact_runtime(known),
                }
                task_row["known_exact"].append(attempt)
                counters["known_exact"] += 1
                fresh_cases, generation = support.SUPPORT.fresh_cases(
                    task, 426_000_000 + task * 100 + int(meta["node_index"]), task_map
                )
                fresh_raw = support.SUPPORT.evaluate_four(data, fresh_cases)
                fresh_pass = len(fresh_cases) >= 2_000 and all(exact(value) for value in fresh_raw.values())
                attempt["fresh_generation"] = generation
                attempt["fresh"] = {
                    name: support.compact_runtime(value) for name, value in fresh_raw.items()
                }
                attempt["fresh_pass"] = fresh_pass
                if not fresh_pass:
                    counters["fresh_reject"] += 1
                    task_row["fresh_rejects"].append(attempt)
                    continue
                structure = support.POLICY.structure_audit(support.SUPPORT, task, model, data)
                attempt["structure"] = structure
                if not structure["pass"]:
                    counters["structure_reject"] += 1
                    continue
                path = HERE / "candidates" / (
                    f"task{task:03d}_bypass_n{meta['node_index']}_i{meta['input_index']}"
                    f"_cost{profile['cost']}_{digest[:12]}.onnx"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                check = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/golf/try_candidate.py"),
                     "--task", str(task), "--onnx", str(path)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                output = check.stdout + check.stderr
                attempt["official_gold"] = check.returncode == 0 and "PASS gold:" in output
                attempt["official_gold_output"] = output[-4000:]
                if not attempt["official_gold"]:
                    counters["official_gold_reject"] += 1
                    path.unlink(missing_ok=True)
                    continue
                attempt["path"] = str(path.relative_to(ROOT))
                report["finalists"].append({"task": task, **attempt})
                counters["finalists"] += 1
            report["tasks"].append(task_row)
            (HERE / "same_shape_bypass_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({
                "task": task, "attempted": task_row["attempted"],
                "known_exact": len(task_row["known_exact"]),
                "finalists": sum(1 for row in report["finalists"] if row["task"] == task),
            }), flush=True)
    report["counters"] = dict(counters)
    (HERE / "same_shape_bypass_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counters": dict(counters), "finalists": report["finalists"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
