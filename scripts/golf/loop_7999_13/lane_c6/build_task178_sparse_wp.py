#!/usr/bin/env python3
"""Encode task178's genuinely sparse convolution kernel as an ONNX sparse initializer.

The dense incumbent kernel has shape ``[1, 10, 2, 2]``.  For colour channel
``c`` its 2x2 slice is ``c * [[0, 1], [1, 16]]``.  Thirteen of the forty
entries are exactly zero, so a standard COO sparse initializer represents the
identical tensor with 27 counted values.  No graph computation or arithmetic
is changed.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base/task178.onnx"
OUTPUT = HERE / "task178_sparse_wp.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    dense = next(item for item in model.graph.initializer if item.name == "WP")
    array = numpy_helper.to_array(dense)
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat).astype(np.int64)
    values = flat[indices].astype(array.dtype, copy=False)
    assert values.size == 27

    sparse = onnx.helper.make_sparse_tensor(
        numpy_helper.from_array(values, name="WP"),
        numpy_helper.from_array(indices),
        list(array.shape),
    )
    model.graph.initializer.remove(dense)
    model.graph.sparse_initializer.append(sparse)
    model.producer_name = "task178-standard-sparse-kernel"

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
