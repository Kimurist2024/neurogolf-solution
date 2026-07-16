#!/usr/bin/env python3
"""Build exact fixed-Shape probes from current task284 only."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high134_fixed_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)


def replace(model: onnx.ModelProto, output: str, value: np.ndarray) -> None:
    producers = [node for node in model.graph.node if output in node.output]
    if len(producers) != 1:
        raise RuntimeError(f"producer count for {output}: {len(producers)}")
    model.graph.node.remove(producers[0])
    model.graph.initializer.append(numpy_helper.from_array(value, name=output))


def main() -> int:
    base = onnx.load(HERE / "current/task284.onnx")
    specs = (
        (
            "shape_input_batch_exact1", "xs0", np.asarray([1], dtype=np.int64),
            "Shape(input,start=0,end=1)=[1] from canonical input [1,10,30,30]",
        ),
        (
            "shape_x70_rows_exact56", "x73", np.asarray([56], dtype=np.int64),
            "x70 concatenates 56 scalar terms, so Shape(x70,start=0,end=1)=[56]",
        ),
    )
    outdir = HERE / "rejected_probes"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, output, value, proof in specs:
        model = copy.deepcopy(base)
        replace(model, output, value)
        data = model.SerializeToString()
        digest = hashlib.sha256(data).hexdigest()
        path = outdir / f"task284_{label}_{digest[:12]}.onnx"
        path.write_bytes(data)
        structural = SCAN.structural(copy.deepcopy(model))
        rows.append({
            "task": 284,
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "proof": proof,
            "reachable_value_range": [int(value[0]), int(value[0])],
            "overflow": "none: exact int64 value",
            "rounding": "none: integer Shape",
            "structural": structural,
            "decision": "REJECT",
        })
        print(f"{label} pass={structural.get('pass')} reasons={structural.get('reasons')}")
    (HERE / "fixed_integer_probes.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
