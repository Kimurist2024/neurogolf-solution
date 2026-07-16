#!/usr/bin/env python3
"""Build the exact-support 3x3 crop of task193's 4x4 grouped Conv.

Channels 1..9 are unchanged because their discarded top row and left column
are exactly zero.  Channel 0 becomes a policy95 approximation and therefore
must pass the independent generator gate before it can be adopted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.source, load_external_data=False)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Conv":
        raise RuntimeError("expected the one-node task193 Conv incumbent")
    initializers = {item.name: item for item in model.graph.initializer}
    weights = np.asarray(numpy_helper.to_array(initializers["W"]))
    if weights.shape != (10, 1, 4, 4):
        raise RuntimeError(f"unexpected weight shape: {weights.shape}")
    if not np.all(weights[1:, :, 0, :] == 0) or not np.all(
        weights[1:, :, :, 0] == 0
    ):
        raise RuntimeError("channels 1..9 do not have the expected exact zero border")

    cropped = np.ascontiguousarray(weights[:, :, 1:4, 1:4])
    replacement = numpy_helper.from_array(cropped, name="W")
    kept = [item for item in model.graph.initializer if item.name != "W"]
    kept.insert(0, replacement)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    conv = model.graph.node[0]
    for attribute in conv.attribute:
        if attribute.name == "pads":
            del attribute.ints[:]
            attribute.ints.extend([1, 1, 1, 1])
            break
    else:
        raise RuntimeError("Conv pads attribute missing")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    report = {
        "source": str(args.source),
        "output": str(args.output),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "weight_shape_before": list(weights.shape),
        "weight_shape_after": list(cropped.shape),
        "discarded_elements": int(weights.size - cropped.size),
        "channels_1_to_9_exact": True,
        "channel_0_policy95": True,
        "strict_checker": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
