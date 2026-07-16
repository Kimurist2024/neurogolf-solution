#!/usr/bin/env python3
"""Apply multiple independent initializer-contraction rewrites to one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import onnx

from einsum_reuse_initializer_contraction import build, measure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--ordinals", required=True, help="Comma-separated rNNN ordinals")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ordinals = [int(value) for value in args.ordinals.split(",")]
    manifest = json.loads(args.manifest.read_text())
    selected: list[dict[str, object]] = []
    for ordinal in ordinals:
        matches = [
            row
            for row in manifest["rows"]
            if int(row["task"]) == args.task
            and re.search(rf"_r{ordinal:03d}\.onnx$", str(row["path"]))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one task{args.task:03d} r{ordinal:03d} plan, got {len(matches)}")
        selected.append(matches[0])

    with zipfile.ZipFile(args.zip) as archive:
        model = onnx.load_model_from_string(archive.read(f"task{args.task:03d}.onnx"))
    base_memory, base_params, base_cost = measure(model, args.task)
    for plan in selected:
        model = build(model, plan)
    memory, params, cost = measure(model, args.task)
    if cost >= base_cost:
        raise ValueError(f"combined candidate is not cheaper: {base_cost}->{cost}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    report = {
        "task": args.task,
        "path": str(args.output),
        "ordinals": ordinals,
        "plans": [
            {
                key: plan[key]
                for key in ("target", "source", "assignment", "parameter_saving")
            }
            for plan in selected
        ],
        "baseline_memory": base_memory,
        "baseline_params": base_params,
        "baseline_cost": base_cost,
        "candidate_memory": memory,
        "candidate_params": params,
        "candidate_cost": cost,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
