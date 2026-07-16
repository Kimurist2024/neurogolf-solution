#!/usr/bin/env python3
"""Fold provably constant Size nodes in the immutable 8004.50 baseline.

This lane is intentionally narrow: task177's Size(spw) is the element count of
the fixed [27] initializer, and task387's Size(input) is the element count of
the canonical fixed input [1,10,30,30].  Replacing either scalar node output by
an int64 scalar initializer is an ONNX-semantics-preserving constant fold.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
CANDIDATES = HERE / "candidates"

FOLDS = {
    177: {"node_output": "spatial_sizes", "value": 27, "source": "spw", "source_shape": [27]},
    387: {
        "node_output": "shape_dyn",
        "value": 9000,
        "source": "input",
        "source_shape": [1, 10, 30, 30],
    },
    # Secondary checks requested for spare capacity.  These intentionally fold
    # only the Size result whose sole CenterCropPad consumer has one axis; the
    # other Size result in each graph feeds multi-axis shape-cloak paths and is
    # left untouched.
    367: {"node_output": "n10", "value": 10, "source": "CBi", "source_shape": [1, 10]},
    69: {
        "node_output": "c10_dyn",
        "value": 10,
        "source": "codes_i8",
        "source_shape": [1, 10, 1, 1],
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fold(task: int, model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    spec = FOLDS[task]
    target = str(spec["node_output"])
    hits = [node for node in model.graph.node if node.op_type == "Size" and list(node.output) == [target]]
    if len(hits) != 1:
        raise RuntimeError(f"task{task:03d}: expected one Size -> {target}, found {len(hits)}")
    node = hits[0]
    if list(node.input) != [spec["source"]]:
        raise RuntimeError(f"task{task:03d}: unexpected Size source {list(node.input)}")

    kept = [candidate for candidate in model.graph.node if candidate is not node]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    if any(initializer.name == target for initializer in model.graph.initializer):
        raise RuntimeError(f"task{task:03d}: initializer {target} already exists")
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(spec["value"], dtype=np.int64), name=target)
    )
    model.graph.name = f"{model.graph.name}_exact_size_fold"
    return model, {
        "task": task,
        "rewrite": f"Size({spec['source']}) -> scalar int64 initializer {spec['value']}",
        "source_shape": spec["source_shape"],
        "removed_node": "Size",
        "added_initializer": target,
        "initializer_value": spec["value"],
    }


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in sorted(FOLDS):
            name = f"task{task:03d}.onnx"
            baseline_bytes = archive.read(name)
            baseline = onnx.load_model_from_string(baseline_bytes)
            candidate, row = fold(task, baseline)
            candidate_bytes = candidate.SerializeToString()
            output = CANDIDATES / name
            output.write_bytes(candidate_bytes)
            row.update(
                {
                    "baseline_zip": BASE_ZIP.name,
                    "baseline_sha256": sha256(baseline_bytes),
                    "candidate": str(output.relative_to(ROOT)),
                    "candidate_sha256": sha256(candidate_bytes),
                }
            )
            rows.append(row)
    (HERE / "build_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
