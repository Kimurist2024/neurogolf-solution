#!/usr/bin/env python3
"""Find the best one-value_info annotation shave per task."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of


def parse_tasks(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.update(range(lo, hi + 1))
        elif part.strip():
            result.add(int(part))
    return sorted(task for task in result if 1 <= task <= 400)


def vi_bytes(value: onnx.ValueInfoProto) -> int:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return 0
    dims = [dim.dim_value for dim in tensor.shape.dim]
    if not dims or any(dim <= 0 for dim in dims):
        return 0
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
    return int(math.prod(dims) * dtype.itemsize)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(3)
    costs = json.loads(args.base_costs.read_text())["costs"]
    winners: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in parse_tasks(args.tasks):
            original = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            ranked = sorted(
                enumerate(original.graph.value_info),
                key=lambda item: -vi_bytes(item[1]),
            )[: args.top]
            best: tuple[int, int, list[int], onnx.ModelProto] | None = None
            probe = args.out_dir / f".task{task:03d}_probe.onnx"
            for index, value in ranked:
                tensor = value.type.tensor_type
                if not tensor.HasField("shape"):
                    continue
                before = [dim.dim_value for dim in tensor.shape.dim]
                if not any(dim > 1 for dim in before):
                    continue
                candidate = copy.deepcopy(original)
                for dim in candidate.graph.value_info[index].type.tensor_type.shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value > 1:
                        dim.dim_value = 1
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
                    onnx.save(candidate, probe)
                    candidate_cost = int(cost_of(str(probe))[2])
                    if 0 < candidate_cost < int(costs[str(task)]) and (
                        best is None or candidate_cost < best[0]
                    ):
                        best = (candidate_cost, index, before, candidate)
                except Exception:
                    continue
            probe.unlink(missing_ok=True)
            if best is None:
                continue
            candidate_cost, index, before, candidate = best
            output = args.out_dir / f"task{task:03d}_single_vi1.onnx"
            onnx.save(candidate, output)
            base_cost = int(costs[str(task)])
            item = {
                "task": task,
                "path": str(output),
                "baseline_cost": base_cost,
                "candidate_cost": candidate_cost,
                "projected_gain": math.log(base_cost / candidate_cost),
                "value_info_index": index,
                "value_info_name": original.graph.value_info[index].name,
                "before": before,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            winners.append(item)
            print(f"task{task:03d}: {base_cost}->{candidate_cost} {item['value_info_name']}")

    winners.sort(key=lambda item: -float(item["projected_gain"]))
    payload = {
        "baseline": str(args.baseline),
        "tasks": args.tasks,
        "top_per_task": args.top,
        "winners": winners,
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
    }
    (args.out_dir / "manifest_pre_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
