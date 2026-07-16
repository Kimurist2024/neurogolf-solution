#!/usr/bin/env python3
"""Probe one-dimension-at-a-time value_info reductions.

Unlike ``value_info_single_sweep.py``, this pass does not collapse every
non-unit dimension of a tensor together.  It tests individual dimensions and
intermediate sizes, which can preserve ORT buffer safety while still lowering
the measured allocation.  Produced models remain discovery candidates only.
"""

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

from scripts.golf.rank_dir import cost_of  # noqa: E402


def parse_tasks(text: str) -> list[int]:
    result: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.update(range(lo, hi + 1))
        elif part.strip():
            result.add(int(part))
    return sorted(task for task in result if 1 <= task <= 400)


def value_bytes(value: onnx.ValueInfoProto) -> int:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return 0
    dims = [int(dim.dim_value) for dim in tensor.shape.dim]
    if not dims or any(dim <= 0 for dim in dims):
        return 0
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
    return int(math.prod(dims) * dtype.itemsize)


def alternatives(original: int) -> list[int]:
    values = {1, original - 1, max(1, original // 2)}
    power = 1
    while power < original:
        values.add(power)
        power *= 2
    return sorted(value for value in values if 0 < value < original)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--top", type=int, default=32)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(3)
    costs = json.loads(args.base_costs.read_text())["costs"]
    winners: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in parse_tasks(args.tasks):
            original = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            base_entry = costs.get(str(task))
            if base_entry is None:
                continue
            base_cost = int(base_entry["cost"] if isinstance(base_entry, dict) else base_entry)
            ranked = sorted(
                enumerate(original.graph.value_info),
                key=lambda item: -value_bytes(item[1]),
            )[: args.top]
            best: tuple[int, int, int, int, onnx.ModelProto] | None = None
            probe = args.out_dir / f".task{task:03d}_probe.onnx"
            for value_index, value in ranked:
                tensor = value.type.tensor_type
                if not tensor.HasField("shape"):
                    continue
                for dim_index, dim in enumerate(tensor.shape.dim):
                    if not dim.HasField("dim_value") or dim.dim_value <= 1:
                        continue
                    before = int(dim.dim_value)
                    for after in alternatives(before):
                        candidate = copy.deepcopy(original)
                        candidate.graph.value_info[value_index].type.tensor_type.shape.dim[
                            dim_index
                        ].dim_value = after
                        try:
                            onnx.checker.check_model(candidate, full_check=True)
                            onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
                            onnx.save(candidate, probe)
                            candidate_cost = int(cost_of(str(probe))[2])
                            if 0 < candidate_cost < base_cost and (
                                best is None or candidate_cost < best[0]
                            ):
                                best = (
                                    candidate_cost,
                                    value_index,
                                    dim_index,
                                    after,
                                    candidate,
                                )
                        except Exception:  # noqa: BLE001
                            continue
            probe.unlink(missing_ok=True)
            if best is None:
                continue
            candidate_cost, value_index, dim_index, after, candidate = best
            output = args.out_dir / f"task{task:03d}_single_dimension.onnx"
            onnx.save(candidate, output)
            source_value = original.graph.value_info[value_index]
            before = int(source_value.type.tensor_type.shape.dim[dim_index].dim_value)
            item = {
                "task": task,
                "path": str(output),
                "baseline_cost": base_cost,
                "candidate_cost": candidate_cost,
                "projected_gain": math.log(base_cost / candidate_cost),
                "value_info_index": value_index,
                "value_info_name": source_value.name,
                "dimension_index": dim_index,
                "before": before,
                "after": after,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            winners.append(item)
            print(
                f"task{task:03d}: {base_cost}->{candidate_cost} "
                f"{source_value.name}[{dim_index}] {before}->{after}"
            )

    payload = {
        "baseline": str(args.baseline),
        "tasks": args.tasks,
        "top_per_task": args.top,
        "winners": winners,
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
    }
    (args.out_dir / "manifest_pre_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
