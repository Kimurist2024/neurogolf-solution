#!/usr/bin/env python3
"""Fail-closed exact residual scan for tasks 025/117/131/330.

The immutable authority is read only from the lane-local extracted members.
Only structurally valid, actually strict-lower variants are retained.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (25, 117, 131, 330)
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
BASE_COST = {25: 474, 117: 605, 131: 691, 330: 896}
KINDS = (
    "cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb",
    "combined", "normalize", "normalized_combined",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "residual_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
SCAN = load_module(
    "residual_scan_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def list_actions(detail: dict[str, object]) -> int:
    return sum(len(value) for value in detail.values() if isinstance(value, list))


def remove_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def castlike_variants(base: onnx.ModelProto):
    init = {item.name: item for item in base.graph.initializer}
    eligible = [
        index for index, node in enumerate(base.graph.node)
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] in init
    ]
    for size in range(1, len(eligible) + 1):
        for indices in itertools.combinations(eligible, size):
            model = copy.deepcopy(base)
            current_init = {item.name: item for item in model.graph.initializer}
            replacements = []
            for index in indices:
                node = model.graph.node[index]
                witness = current_init[node.input[1]]
                replacements.append(
                    {"node_index": index, "witness": node.input[1], "to": int(witness.data_type)}
                )
                node.op_type = "Cast"
                del node.input[1:]
                del node.attribute[:]
                node.attribute.extend([helper.make_attribute("to", witness.data_type)])
            removed = remove_unused_initializers(model)
            yield model, {
                "castlike_indices": list(indices),
                "replacements": replacements,
                "removed_initializers": removed,
            }


def factor_bounds(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows = []
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        elements = int(array.size)
        itemsize = int(array.dtype.itemsize)
        # Even granting a free existing shape carrier, materializing a factored
        # N-element tensor costs N*itemsize bytes plus one scalar parameter.
        optimistic_delta = elements * itemsize + 1 - elements
        rows.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": elements,
                "optimistic_cost_delta": optimistic_delta,
                "can_strictly_reduce": optimistic_delta < 0,
            }
        )
    return rows


def evaluate(task: int, label: str, model: onnx.ModelProto, detail: dict[str, object],
             seen: set[str]) -> dict[str, object] | None:
    data = model.SerializeToString()
    sha = digest(data)
    if sha in seen:
        return None
    seen.add(sha)
    structural = SCAN.structural(copy.deepcopy(model))
    row: dict[str, object] = {
        "label": label,
        "sha256": sha,
        "detail": detail,
        "structural": structural,
        "strict_lower": False,
    }
    if structural.get("pass"):
        profile = SCAN.official_cost(data, f"residual_{task:03d}_{label}")
        row["profile"] = profile
        row["strict_lower"] = 0 <= profile["cost"] < BASE_COST[task]
        if row["strict_lower"]:
            path = HERE / "candidates" / f"task{task:03d}_{label}_{sha[:12]}.onnx"
            path.write_bytes(data)
            row["path"] = str(path.relative_to(ROOT))
    return row


def main() -> int:
    if digest((ROOT / "submission_base_8009.46.zip").read_bytes()) != AUTHORITY_SHA:
        raise RuntimeError("authority SHA changed")
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    report: dict[str, object] = {
        "authority_zip": "submission_base_8009.46.zip",
        "authority_zip_sha256": AUTHORITY_SHA,
        "tasks": {},
        "strict_lower": [],
    }
    for task in TASKS:
        data = (HERE / "baseline" / f"task{task:03d}.onnx").read_bytes()
        base = onnx.load_model_from_string(data)
        base_profile = SCAN.official_cost(data, f"residual_{task:03d}_base")
        if base_profile["cost"] != BASE_COST[task]:
            raise RuntimeError(f"task{task:03d} authority cost changed: {base_profile}")
        seen = {digest(data)}
        rows = []
        for kind in KINDS:
            candidate, detail = EXACT.transform(base, kind)
            if not list_actions(detail):
                continue
            row = evaluate(task, kind, candidate, detail, seen)
            if row is not None:
                rows.append(row)
        cast_count = 0
        for index, (candidate, detail) in enumerate(castlike_variants(base)):
            cast_count += 1
            row = evaluate(task, f"castattr_{index}", candidate, detail, seen)
            if row is not None:
                rows.append(row)
        strict = [row for row in rows if row["strict_lower"]]
        report["strict_lower"].extend({"task": task, **row} for row in strict)
        bounds = factor_bounds(base)
        report["tasks"][str(task)] = {
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "authority_profile": base_profile,
            "structural": SCAN.structural(copy.deepcopy(base)),
            "graph_inventory": SCAN.graph_inventory(copy.deepcopy(base)),
            "castlike_subset_count": cast_count,
            "factor_bounds": bounds,
            "all_factor_bounds_nonnegative": all(not row["can_strictly_reduce"] for row in bounds),
            "variant_count": len(rows),
            "strict_lower_count": len(strict),
            "rows": rows,
        }
        print(
            f"task{task:03d} cost={base_profile['cost']} variants={len(rows)} "
            f"cast_subsets={cast_count} strict_lower={len(strict)}",
            flush=True,
        )
    report["strict_lower_count"] = len(report["strict_lower"])
    report["variant_count"] = sum(
        int(row["variant_count"]) for row in report["tasks"].values()
    )
    report["variant_status_counts"] = dict(
        Counter(
            "strict_lower" if row["strict_lower"]
            else "structural_reject" if not row["structural"].get("pass")
            else "not_lower"
            for task in report["tasks"].values()
            for row in task["rows"]
        )
    )
    (HERE / "audit" / "residual_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "variant_count": report["variant_count"],
        "status_counts": report["variant_status_counts"],
        "strict_lower_count": report["strict_lower_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
