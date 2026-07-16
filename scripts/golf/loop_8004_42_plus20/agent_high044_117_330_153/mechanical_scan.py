#!/usr/bin/env python3
"""Exact mechanical and factor audit for the lane's immutable baselines."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_COST = {44: 1076, 117: 605, 330: 896}
KINDS = ("cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb", "combined")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_summary(item: onnx.TensorProto) -> dict[str, object]:
    array = numpy_helper.to_array(item)
    return {
        "name": item.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "elements": int(array.size),
        "used_by": [],
    }


def factor_audit(model: onnx.ModelProto) -> dict[str, object]:
    uses: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for slot, name in enumerate(node.input):
            if name:
                uses[name].append({"node_index": index, "op": node.op_type, "input_slot": slot})
    rows = []
    keys: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        row = array_summary(item)
        row["used_by"] = uses.get(item.name, [])
        rows.append(row)
        clone = copy.deepcopy(item)
        clone.name = ""
        keys[(int(item.data_type), tuple(item.dims), clone.SerializeToString())].append(item.name)
    duplicates = [names for names in keys.values() if len(names) > 1]
    dead = [row["name"] for row in rows if not row["used_by"]]
    opset = {item.domain or "": int(item.version) for item in model.opset_import}
    return {
        "nodes": len(model.graph.node),
        "initializers": rows,
        "parameter_elements": sum(int(row["elements"]) for row in rows),
        "unused_initializers": dead,
        "duplicate_full_tensor_groups": duplicates,
        "opset_import": opset,
        "attribute_embedding_audit": {
            "conclusion": "No removable tensor input can be converted to a same-op attribute at the imported opset. QLinear*/Einsum/MatMulInteger operands are schema tensor inputs; Slice/Shape inputs in task117 are also tensor-form at this opset; ConstantOfShape scalar values in task330 are already attributes and cost zero parameters.",
            "constant_of_shape_nodes_with_value_attribute": sum(
                node.op_type == "ConstantOfShape" and any(attr.name == "value" for attr in node.attribute)
                for node in model.graph.node
            ),
        },
        "dtype_audit": "Parameter cost counts elements, not bytes. Initializer-only dtype narrowing cannot lower score; activation dtype changes require an exact supported-kernel rewrite and are rejected when they preserve false declared shapes.",
    }


def main() -> None:
    scanner = load_module(
        "lane153_scanner",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
    )
    auditor = load_module(
        "lane153_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    harvest = load_module(
        "lane153_harvest",
        ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
    )
    candidates = HERE / "candidates"
    candidates.mkdir(exist_ok=True)
    report: dict[str, object] = {
        "authority_zip": "submission_base_8009.46.zip",
        "authority_zip_sha256": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
        "base_cost": BASE_COST,
        "tasks": {},
    }
    seen: set[tuple[int, str]] = set()
    for task, base_cost in BASE_COST.items():
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        base = onnx.load(path)
        rows = []
        for kind in KINDS:
            candidate, actions = scanner.transform(base, kind)
            if not actions["semantic_action_count"]:
                continue
            data = candidate.SerializeToString()
            candidate_sha = sha256(data)
            if (task, candidate_sha) in seen:
                continue
            seen.add((task, candidate_sha))
            candidate_path = candidates / f"task{task:03d}_{kind}_{candidate_sha[:12]}.onnx"
            candidate_path.write_bytes(data)
            structural = scanner.AUDIT.structural_audit(data)
            actual = harvest.actual_screen(data, task) if structural["pass"] else None
            row: dict[str, object] = {
                "task": task,
                "kind": kind,
                "path": str(candidate_path.relative_to(ROOT)),
                "sha256": candidate_sha,
                "actions": actions,
                "structural_audit": structural,
                "actual_cost": actual,
                "strictly_lower": actual is not None and actual < base_cost,
            }
            if row["strictly_lower"]:
                row["deep_audit"] = auditor.audit(f"task{task:03d}_{kind}", task, candidate_path)
                trace = row["deep_audit"].get("runtime_shape_trace", {})
                row["truthful_runtime_shapes"] = not trace.get("error") and not trace.get("declared_actual_mismatches")
                profile = row["deep_audit"].get("official_like_score") or {}
                row["gain"] = math.log(base_cost / int(profile["cost"])) if profile.get("cost") else None
            rows.append(row)
        report["tasks"][str(task)] = {
            "factor_audit": factor_audit(base),
            "candidate_count": len(rows),
            "strict_lower_count": sum(bool(row["strictly_lower"]) for row in rows),
            "rows": rows,
        }
        (HERE / "mechanical_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(task, Counter("strict_lower" if row["strictly_lower"] else "rejected" for row in rows), flush=True)


if __name__ == "__main__":
    main()
