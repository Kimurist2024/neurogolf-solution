#!/usr/bin/env python3
"""Extract and structurally inventory exact 7999.13 B9 task members."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import score_and_verify  # noqa: E402


TASKS = {
    156: "694f12f3",
    182: "776ffc46",
    216: "8efcae92",
    237: "99fa7670",
    238: "9aec4887",
    284: "b7249182",
    379: "ecdecbb3",
}
ZIP = ROOT / "submission_base_7999.13.zip"


def tensor_key(tensor: onnx.TensorProto) -> tuple[object, ...]:
    array = numpy_helper.to_array(tensor)
    return (str(array.dtype), tuple(array.shape), array.tobytes())


def attr_key(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.input[:]
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def main() -> int:
    base_dir = HERE / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, object] = {
        "baseline_zip": str(ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        "tasks": {},
    }
    with zipfile.ZipFile(ZIP) as archive:
        for task, generator_hash in TASKS.items():
            raw = archive.read(f"task{task:03d}.onnx")
            path = base_dir / f"task{task:03d}.onnx"
            path.write_bytes(raw)
            model = onnx.load_from_string(raw)
            onnx.checker.check_model(model, full_check=True)
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)

            uses: dict[str, int] = defaultdict(int)
            for node in model.graph.node:
                for name in node.input:
                    uses[name] += 1
            unused_initializers = [
                initializer.name
                for initializer in model.graph.initializer
                if uses[initializer.name] == 0
            ]
            by_value: dict[tuple[object, ...], list[str]] = defaultdict(list)
            for initializer in model.graph.initializer:
                by_value[tensor_key(initializer)].append(initializer.name)
            duplicate_initializers = [names for names in by_value.values() if len(names) > 1]

            by_expr: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
            for index, node in enumerate(model.graph.node):
                key = (node.op_type, node.domain, tuple(node.input), attr_key(node))
                by_expr[key].append({"index": index, "outputs": list(node.output)})
            duplicate_expressions = [group for group in by_expr.values() if len(group) > 1]

            score = score_and_verify(
                copy.deepcopy(model), task, str(HERE / "work"), label=f"base{task}", require_correct=True
            )
            if score is None:
                raise RuntimeError(f"exact baseline failed official-like scoring: task{task}")
            inventory["tasks"][str(task)] = {
                "generator_hash": generator_hash,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
                "initializer_elements": sum(
                    int(numpy_helper.to_array(initializer).size)
                    for initializer in model.graph.initializer
                ),
                "score": score,
                "unused_initializers": unused_initializers,
                "duplicate_initializers": duplicate_initializers,
                "duplicate_expressions": duplicate_expressions,
                "structure": {
                    "checker_full": True,
                    "strict_shape_inference": True,
                    "domains": sorted({op.domain for op in inferred.opset_import}),
                    "functions": len(inferred.functions),
                    "sparse_initializers": len(inferred.graph.sparse_initializer),
                    "inputs": len(inferred.graph.input),
                    "outputs": len(inferred.graph.output),
                },
            }
    out = HERE / "baseline_inventory.json"
    out.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
