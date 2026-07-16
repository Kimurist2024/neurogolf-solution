#!/usr/bin/env python3
"""Build lane-local standard task345 controls and sparse-storage probes."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ZIP = ROOT / "submission_base_8000.46.zip"
BASELINE = HERE / "baseline_task345.onnx"
LEGAL = HERE / "task345_legal_swapped_prescaled_cost389.onnx"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def build_legal(source: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        old_pads = list(
            helper.get_attribute_value(
                next(attribute for attribute in node.attribute if attribute.name == "pads")
            )
        )
        if len(node.input) != 2 or node.input[0] != "input" or node.input[1] != "Wpack":
            raise AssertionError(list(node.input))
        node.input[0], node.input[1] = node.input[1], node.input[0]
        replace_attribute(node, "kernel_shape", [30, 30])
        replace_attribute(node, "pads", [abs(value) for value in old_pads])
    model.graph.name = "task345_legal_swapped_prescaled_cost389"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def make_sparse(model: onnx.ModelProto, names: tuple[str, ...]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in result.graph.initializer
    }
    kept = [item for item in result.graph.initializer if item.name not in names]
    del result.graph.initializer[:]
    result.graph.initializer.extend(kept)
    for name in names:
        array = arrays[name]
        flat_positions = np.flatnonzero(array.reshape(-1)).astype(np.int64)
        coordinates = np.stack(
            np.unravel_index(flat_positions, array.shape), axis=1
        ).astype(np.int64)
        values = numpy_helper.from_array(array.reshape(-1)[flat_positions], name=name)
        indices = numpy_helper.from_array(coordinates, name=f"{name}_indices")
        result.graph.sparse_initializer.append(
            helper.make_sparse_tensor(values, indices, list(array.shape))
        )
    return result


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        BASELINE.write_bytes(archive.read("task345.onnx"))
    baseline = onnx.load(BASELINE)
    legal = build_legal(baseline)
    onnx.save(legal, LEGAL)

    sparse_rows: list[dict[str, object]] = []
    for names in (("Wpack",), ("cfac",), ("wfac",), ("cfac", "wfac")):
        probe = make_sparse(legal, names)
        row: dict[str, object] = {
            "initializers": list(names),
            "dense_elements_removed": sum(
                numpy_helper.to_array(item).size
                for item in legal.graph.initializer
                if item.name in names
            ),
            "stored_nonzero_values": sum(
                sparse.values.dims[0] if sparse.values.dims else 1
                for sparse in probe.graph.sparse_initializer
            ),
        }
        try:
            onnx.checker.check_model(probe, full_check=True)
            onnx.shape_inference.infer_shapes(probe, strict_mode=True, data_prop=True)
            row["full_checker"] = True
        except Exception as exc:  # expected schema rejection, retained as evidence
            row.update(full_checker=False, error=f"{type(exc).__name__}: {exc}")
        sparse_rows.append(row)

    manifest = {
        "baseline_zip": str(ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": sha256(ZIP),
        "baseline_model_sha256": sha256(BASELINE),
        "legal_swapped_prescaled": {
            "path": str(LEGAL.relative_to(ROOT)),
            "sha256": sha256(LEGAL),
            "full_checker": True,
            "strict_shape_data_prop": True,
            "change": "swap Conv data/weight and replace negative crop pads with positive standard pads; retain identical Wpack values and recurrence",
        },
        "sparse_probes": sparse_rows,
    }
    path = HERE / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
