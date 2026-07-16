#!/usr/bin/env python3
"""Convert an exactly-zero-heavy Conv initializer to standard ONNX COO form."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="W")
    args = parser.parse_args()
    model = onnx.load(args.input)
    position = next(i for i, tensor in enumerate(model.graph.initializer) if tensor.name == args.name)
    tensor = model.graph.initializer[position]
    array = numpy_helper.to_array(tensor)
    linear = np.flatnonzero(array.reshape(-1) != 0).astype(np.int64)
    values = array.reshape(-1)[linear]
    sparse = onnx.SparseTensorProto()
    sparse.values.CopyFrom(numpy_helper.from_array(values, name=args.name))
    sparse.indices.CopyFrom(numpy_helper.from_array(linear, name=args.name + "_indices"))
    sparse.dims.extend(array.shape)
    del model.graph.initializer[position]
    model.graph.sparse_initializer.append(sparse)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"{args.name}: dense={array.size} sparse={values.size}")


if __name__ == "__main__":
    main()
