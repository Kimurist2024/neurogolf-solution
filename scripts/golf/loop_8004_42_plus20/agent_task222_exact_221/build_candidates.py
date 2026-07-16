#!/usr/bin/env python3
"""Build isolated task222 algebra probes; never promotes or mutates shared state."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "authority" / "task222.onnx"
OUT = HERE / "candidates"


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {item.name: item for item in model.graph.initializer}


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> None:
    item = initializer_map(model)[name]
    item.CopyFrom(numpy_helper.from_array(np.asarray(array), name=name))


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(item for item in node.attribute if item.name == "equation")


def save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    onnx.save(model, OUT / name)


def build_no_projection(base: onnx.ModelProto) -> None:
    model = copy.deepcopy(base)
    node = model.graph.node[0]
    attr = equation_attribute(node)
    equation = attr.s.decode("ascii")
    suffix = ",ok->borc"
    assert equation.endswith(suffix), equation
    attr.s = (equation[: -len(suffix)] + "->bkrc").encode("ascii")
    del node.input[-1]
    kept = [item for item in model.graph.initializer if item.name != "P"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    save(model, "task222_no_P_cost280.onnx")


def build_rank_drops(base: onnx.ModelProto) -> None:
    for component in range(8):
        model = copy.deepcopy(base)
        arrays = {
            name: numpy_helper.to_array(item)
            for name, item in initializer_map(model).items()
        }
        replace_initializer(model, "V", np.delete(arrays["V"], component, axis=1))
        replace_initializer(model, "U", np.delete(arrays["U"], component, axis=1))
        save(model, f"task222_drop_rank_{component}_cost348.onnx")


def sparsify(model: onnx.ModelProto, names: set[str]) -> None:
    dense = initializer_map(model)
    sparse_items = []
    for name in sorted(names):
        array = numpy_helper.to_array(dense[name])
        flat = np.flatnonzero(array).astype(np.int64)
        coordinates = np.stack(np.unravel_index(flat, array.shape), axis=1).astype(
            np.int64
        )
        sparse_items.append(
            helper.make_sparse_tensor(
                numpy_helper.from_array(array.reshape(-1)[flat], name=name),
                numpy_helper.from_array(coordinates),
                list(array.shape),
            )
        )
    kept = [item for item in model.graph.initializer if item.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.extend(sparse_items)


def build_sparse_probe(base: onnx.ModelProto) -> None:
    # Exact dense values, but the official sanitizer does not rename sparse
    # initializers. This probe documents the runtime/strict-inference outcome.
    model = copy.deepcopy(base)
    sparsify(model, {"V", "S", "P"})
    # Plain checker accepts SparseTensorProto, but full_check/shape inference
    # rejects it as rank 0 at the Einsum boundary on the target ONNX build.
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, OUT / "task222_sparse_VSP_probe.onnx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = onnx.load(SOURCE)
    assert len(base.graph.node) == 1
    build_no_projection(base)
    build_rank_drops(base)
    build_sparse_probe(base)


if __name__ == "__main__":
    main()
