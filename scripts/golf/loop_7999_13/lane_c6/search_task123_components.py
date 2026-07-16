#!/usr/bin/env python3
"""Probe whether task123's six shared CP components contain a removable lane."""

from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def main() -> None:
    source = onnx.load(HERE / "base/task123.onnx")
    for drop in range(6):
        model = onnx.ModelProto()
        model.CopyFrom(source)
        v = next(item for item in model.graph.initializer if item.name == "V")
        f = next(item for item in model.graph.initializer if item.name == "F")
        va = numpy_helper.to_array(v)
        fa = numpy_helper.to_array(f)
        keep = [index for index in range(6) if index != drop]
        v.CopyFrom(numpy_helper.from_array(va[keep], name="V"))
        f.CopyFrom(numpy_helper.from_array(fa[:, keep][:, :, keep], name="F"))
        model.producer_name = f"task123-drop-shared-component-{drop}"
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        output = HERE / f"task123_drop{drop}.onnx"
        onnx.save(inferred, output)
        print(output)


if __name__ == "__main__":
    main()
