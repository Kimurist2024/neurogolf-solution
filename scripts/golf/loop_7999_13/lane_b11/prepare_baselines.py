#!/usr/bin/env python3
"""Extract and structurally inventory exact 7999.13 B11 task members."""

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
    264: "a8c38be5",
    281: "b548a754",
    300: "be94b721",
    358: "e21d9049",
    376: "eb281b96",
    387: "f35d900a",
    392: "f8c80d96",
}
ZIP = ROOT / "submission_base_7999.13.zip"


def tensor_key(tensor: onnx.TensorProto) -> tuple[object, ...]:
    array = numpy_helper.to_array(tensor)
    return str(array.dtype), tuple(array.shape), array.tobytes()


def attr_key(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.input[:]
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def main() -> None:
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
    with zipfile.ZipFile(ZIP) as archive:
        for task, generator_hash in TASKS.items():
            raw = archive.read(f"task{task:03d}.onnx")
            path = base_dir / f"task{task:03d}.onnx"
            model = onnx.load_from_string(raw)
            onnx.checker.check_model(model, full_check=True)
            inferred = onnx.shape_inference.infer_shapes(
                model, strict_mode=True, data_prop=True
            )
            uses: dict[str, int] = defaultdict(int)
            for node in model.graph.node:
                for name in node.input:
                    uses[name] += 1
            unused = [item.name for item in model.graph.initializer if uses[item.name] == 0]
            by_value: dict[tuple[object, ...], list[str]] = defaultdict(list)
            for item in model.graph.initializer:
                by_value[tensor_key(item)].append(item.name)
            duplicates = [names for names in by_value.values() if len(names) > 1]
            by_expr: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
            for index, node in enumerate(model.graph.node):
                key = (node.op_type, node.domain, tuple(node.input), attr_key(node))
                by_expr[key].append({"index": index, "outputs": list(node.output)})
            duplicate_expressions = [group for group in by_expr.values() if len(group) > 1]
            try:
                scored = score_and_verify(
                    copy.deepcopy(model), task, str(HERE / "work"),
                    label=f"base{task}", require_correct=True,
                )
            except Exception as exc:  # noqa: BLE001
                scored = {"error": f"{type(exc).__name__}: {exc}"}
            inventory["tasks"][str(task)] = {
                "generator_hash": generator_hash,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
                "initializer_elements": sum(
                    int(numpy_helper.to_array(item).size) for item in model.graph.initializer
                ),
                "score": scored,
                "unused_initializers": unused,
                "duplicate_initializers": duplicates,
                "duplicate_expressions": duplicate_expressions,
                "structure": {
                    "checker_full": True,
                    "strict_shape_inference": True,
                    "domains": sorted({item.domain for item in inferred.opset_import}),
                    "functions": len(inferred.functions),
                    "sparse_initializers": len(inferred.graph.sparse_initializer),
                    "inputs": len(inferred.graph.input),
                    "outputs": len(inferred.graph.output),
                },
            }
            (HERE / "baseline_inventory.json").write_text(
                json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
            )
    (HERE / "baseline_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
