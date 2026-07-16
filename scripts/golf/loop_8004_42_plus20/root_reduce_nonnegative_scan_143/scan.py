#!/usr/bin/env python3
"""Profile ReduceL1/ReduceSumSquare -> ReduceSum on the immutable authority.

This is a cost-only triage.  A lower profile is not semantically admitted here;
it must separately prove that the reduced input is nonnegative (ReduceL1) or
binary/otherwise square-invariant (ReduceSumSquare), then pass full gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
OUT = Path(__file__).with_name("scan.json")
TARGET_OPS = {"ReduceL1", "ReduceSumSquare"}


def profile(model: onnx.ModelProto, task: int) -> dict | None:
    try:
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"reduce143_{task:03d}_") as wd:
            path = Path(wd) / f"task{task:03d}.onnx"
            onnx.save(model, path)
            memory, params, cost = cost_of(str(path))
        if cost < 0:
            return {"error": "cost profiler rejected model"}
        return {
            "cost": int(cost),
            "memory": int(memory),
            "params": int(params),
        }
    except Exception as exc:  # cost triage must finish across all 400 payloads
        return {"error": f"{type(exc).__name__}: {exc}"}


def producer_summary(model: onnx.ModelProto, value: str) -> dict:
    for node in model.graph.node:
        if value in node.output:
            return {
                "op_type": node.op_type,
                "name": node.name,
                "inputs": list(node.input),
            }
    for init in model.graph.initializer:
        if init.name == value:
            return {"op_type": "Initializer", "name": init.name, "inputs": []}
    if any(value == item.name for item in model.graph.input):
        return {"op_type": "GraphInput", "name": value, "inputs": []}
    return {"op_type": "Unknown", "name": value, "inputs": []}


def main() -> None:
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith("task") and name.endswith(".onnx")
        )
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            raw = archive.read(member)
            model = onnx.load_model_from_string(raw)
            targets = [
                (idx, node)
                for idx, node in enumerate(model.graph.node)
                if node.op_type in TARGET_OPS
            ]
            if not targets:
                continue
            baseline = profile(model, task)
            for idx, node in targets:
                candidate = copy.deepcopy(model)
                candidate.graph.node[idx].op_type = "ReduceSum"
                cand = profile(candidate, task)
                strict = bool(
                    baseline
                    and cand
                    and "cost" in baseline
                    and "cost" in cand
                    and cand["cost"] < baseline["cost"]
                )
                candidate_raw = candidate.SerializeToString()
                rows.append(
                    {
                        "task": task,
                        "node_index": idx,
                        "source_op": node.op_type,
                        "node_name": node.name,
                        "input": node.input[0] if node.input else "",
                        "producer": producer_summary(
                            model, node.input[0] if node.input else ""
                        ),
                        "baseline": baseline,
                        "candidate": cand,
                        "strict_lower": strict,
                        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
                    }
                )
    OUT.write_text(json.dumps({"authority": str(AUTHORITY), "rows": rows}, indent=2) + "\n")
    strict_rows = [row for row in rows if row["strict_lower"]]
    print(json.dumps({"profiles": len(rows), "strict_lower": strict_rows}, indent=2))


if __name__ == "__main__":
    main()
