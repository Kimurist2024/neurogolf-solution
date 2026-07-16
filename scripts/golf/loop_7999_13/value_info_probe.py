#!/usr/bin/env python3
"""Probe value-info-only cost shaves; never promotes candidates.

The computation graph and initializers remain byte-identical after removing
value_info.  Every resulting model still needs strict execution/gold/fresh
validation because incorrect annotations can influence ORT buffer reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.rank_dir import cost_of


def parse_tasks(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.update(range(start, end + 1))
        elif part.strip():
            result.add(int(part))
    return sorted(task for task in result if 1 <= task <= 400)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.base_costs.read_text())["costs"]
    ort.set_default_logger_severity(3)
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in parse_tasks(args.tasks):
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            changed: list[dict[str, object]] = []
            for value in model.graph.value_info:
                tensor = value.type.tensor_type
                if not tensor.HasField("shape"):
                    continue
                before = [dim.dim_value for dim in tensor.shape.dim]
                if not any(dim > 1 for dim in before):
                    continue
                for dim in tensor.shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value > 1:
                        dim.dim_value = 1
                changed.append({"name": value.name, "before": before})
            if not changed:
                continue
            try:
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(model, strict_mode=True)
                path = args.out_dir / f"task{task:03d}_all_vi1.onnx"
                onnx.save(model, path)
                candidate_cost = cost_of(str(path))[2]
                base_cost = int(costs[str(task)])
                if candidate_cost < 0 or candidate_cost >= base_cost:
                    path.unlink(missing_ok=True)
                    continue
                item = {
                    "task": task,
                    "path": str(path),
                    "baseline_cost": base_cost,
                    "candidate_cost": candidate_cost,
                    "projected_gain": math.log(base_cost / candidate_cost),
                    "changed_value_info": changed,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                rows.append(item)
                print(f"task{task:03d}: {base_cost}->{candidate_cost} gain={item['projected_gain']:.6f}")
            except Exception as exc:
                rows.append({"task": task, "error": repr(exc)})
    rows.sort(key=lambda item: -float(item.get("projected_gain", -1.0)))
    payload = {
        "baseline": str(args.baseline),
        "tasks": args.tasks,
        "candidates": rows,
        "candidate_gain": sum(float(item.get("projected_gain", 0.0)) for item in rows),
    }
    (args.out_dir / "manifest_pre_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"candidates": len(rows), "candidate_gain": payload["candidate_gain"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
