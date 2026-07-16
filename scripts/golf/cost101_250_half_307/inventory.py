#!/usr/bin/env python3
"""Inventory authority members and compare cost<=10 structural fingerprints."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def main():
    rows = {}
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            rows[task] = {"cost": int(row["cost"]), "score": float(row["score"])}
    groups = {"reference_cost_le10": [], "targets_101_250": []}
    with zipfile.ZipFile(ROOT / "submission_base_8011.05.zip") as archive:
        for task, score in sorted(rows.items()):
            if not (score["cost"] <= 10 or 101 <= score["cost"] <= 250):
                continue
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            initializers = []
            for tensor in model.graph.initializer:
                initializers.append({
                    "name": tensor.name, "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                    "dims": list(tensor.dims), "elements": int(__import__('math').prod(tensor.dims)),
                })
            item = {
                "task": task, **score, "nodes": len(model.graph.node),
                "ops": [node.op_type for node in model.graph.node],
                "op_hist": dict(Counter(node.op_type for node in model.graph.node)),
                "params": int(scoring.calculate_params(model)),
                "initializers": initializers,
                "outputs": [value.name for value in model.graph.output],
                "output_producer": next((node.op_type for node in model.graph.node
                    if any(name in {v.name for v in model.graph.output} for name in node.output)), None),
            }
            key = "reference_cost_le10" if score["cost"] <= 10 else "targets_101_250"
            groups[key].append(item)
    (HERE / "inventory.json").write_text(json.dumps(groups, indent=2) + "\n")
    for key, items in groups.items():
        print(key, len(items))
        for x in items:
            print(f"{x['task']:03d} c={x['cost']:3d} p={x['params']:3d} n={x['nodes']:2d} "
                  f"out={x['output_producer']} ops={'/'.join(x['ops'])}")


if __name__ == "__main__":
    main()
