#!/usr/bin/env python3
"""Exact CastLike attributeization plus initializer-factor lower-bound audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (25, 131, 363)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high150_attr_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high150_attr_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prune_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def castlike_subsets(model: onnx.ModelProto):
    init = {item.name: item for item in model.graph.initializer}
    eligible = [
        index
        for index, node in enumerate(model.graph.node)
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] in init
    ]
    for size in range(1, len(eligible) + 1):
        yield from itertools.combinations(eligible, size)


def make_cast_candidate(base: onnx.ModelProto, indices: tuple[int, ...]):
    model = copy.deepcopy(base)
    init = {item.name: item for item in model.graph.initializer}
    replacements = []
    for index in indices:
        node = model.graph.node[index]
        witness = init[node.input[1]]
        replacements.append(
            {"node_index": index, "output": node.output[0], "witness": node.input[1], "to": witness.data_type}
        )
        node.op_type = "Cast"
        del node.input[1:]
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("to", witness.data_type)])
    removed = prune_unused_initializers(model)
    return model, replacements, removed


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def factor_bounds(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows = []
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        elements = int(array.size)
        itemsize = int(array.dtype.itemsize)
        # Reconstructing an N-element initializer with an ordinary node makes
        # the N-element result an intermediate. Even granting a free existing
        # shape carrier and one scalar parameter, the best possible delta is:
        #   N*itemsize + 1 - N.
        # It is positive for every nonempty tensor and dtype here.
        optimistic_delta = elements * itemsize + 1 - elements
        rows.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": elements,
                "itemsize": itemsize,
                "optimistic_cost_delta": optimistic_delta,
                "can_strictly_reduce_via_materialized_factor": optimistic_delta < 0,
            }
        )
    return rows


def main() -> int:
    report: dict[str, object] = {"tasks": {}, "strict_lower": []}
    for task in TASKS:
        data = (HERE / f"baseline/task{task:03d}.onnx").read_bytes()
        base = onnx.load_model_from_string(data)
        base_profile = SCAN.official_cost(data, f"high150_attr_task{task:03d}_base")
        rows = []
        seen = set()
        for indices in castlike_subsets(base):
            candidate, replacements, removed = make_cast_candidate(base, indices)
            candidate_data = candidate.SerializeToString()
            sha = digest(candidate_data)
            if sha in seen:
                continue
            seen.add(sha)
            static = SCAN.structural(copy.deepcopy(candidate))
            row: dict[str, object] = {
                "indices": list(indices),
                "replacements": replacements,
                "removed_initializers": removed,
                "sha256": sha,
                "static": static,
                "strict_lower": False,
            }
            if static.get("pass"):
                profile = SCAN.official_cost(candidate_data, f"high150_attr_task{task:03d}_{len(rows)}")
                row["official_profile"] = profile
                row["strict_lower"] = 0 <= profile["cost"] < base_profile["cost"]
                row["runtime_shape_trace"] = safe_trace(task, candidate_data)
                if row["strict_lower"]:
                    candidate_path = HERE / f"candidates/task{task:03d}_castattr_{sha[:12]}.onnx"
                    candidate_path.write_bytes(candidate_data)
                    row["path"] = str(candidate_path.relative_to(ROOT))
                    report["strict_lower"].append({"task": task, **row})
            rows.append(row)
        bounds = factor_bounds(base)
        report["tasks"][str(task)] = {
            "authority_sha256": digest(data),
            "authority_profile": base_profile,
            "castlike_subset_count": len(rows),
            "castlike_rows": rows,
            "initializer_factor_bounds": bounds,
            "all_materialized_factor_bounds_nonnegative": all(
                not row["can_strictly_reduce_via_materialized_factor"] for row in bounds
            ),
            "schema_note": (
                "At this opset, Slice/Reduce/Squeeze/Unsqueeze/Resize tensors used by these graphs "
                "cannot be moved to free attributes; existing Shape/Attention/CenterCropPad/QLinearConv "
                "settings are already attributes where their schemas permit it."
            ),
        }
        print(
            f"task{task:03d} cast_subsets={len(rows)} lower="
            f"{sum(bool(row['strict_lower']) for row in rows)} factor_lower=0",
            flush=True,
        )
    (HERE / "audit/attribute_factor_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
