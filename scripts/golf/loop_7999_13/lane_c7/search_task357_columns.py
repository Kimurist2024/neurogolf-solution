#!/usr/bin/env python3
"""Search whether task357's right guard columns can be replaced by Conv padding."""

from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def main() -> None:
    source = onnx.load(HERE / "base/task357.onnx")
    for drop in range(1, 8):
        model = onnx.ModelProto()
        model.CopyFrom(source)
        columns = next(item for item in model.graph.initializer if item.name == "columns")
        array = numpy_helper.to_array(columns)
        assert array.shape == (1, 1, 1, 16)
        columns.CopyFrom(numpy_helper.from_array(array[..., :-drop], name="columns"))
        final = model.graph.node[-1]
        assert final.op_type == "QLinearConv"
        pads = next(attr for attr in final.attribute if attr.name == "pads")
        assert list(pads.ints) == [0, 0, 20, 16]
        del pads.ints[:]
        pads.ints.extend([0, 0, 20, 16 + drop])
        model.producer_name = f"task357-drop-right-guard-{drop}"
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        output = HERE / f"task357_drop_guard_{drop}.onnx"
        onnx.save(inferred, output)
        print(output)


if __name__ == "__main__":
    main()
