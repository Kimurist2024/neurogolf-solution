#!/usr/bin/env python3
"""Audit exact optional-input/output removals in the 7999.13 baseline."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/optional_exact_audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            initializers = {item.name: item for item in model.graph.initializer}
            uses = Counter(name for node in model.graph.node for name in node.input if name)
            uses.update(item.name for item in model.graph.output)
            findings: list[dict[str, object]] = []
            for index, node in enumerate(model.graph.node):
                if node.op_type in {"Conv", "ConvTranspose", "Gemm"} and len(node.input) >= 3:
                    bias = initializers.get(node.input[2])
                    if bias is not None and np.all(numpy_helper.to_array(bias) == 0):
                        findings.append(
                            {
                                "kind": "zero_optional_bias",
                                "node_index": index,
                                "op": node.op_type,
                                "input": bias.name,
                                "elements": int(np.prod(bias.dims)) if bias.dims else 1,
                                "unique_use": uses[bias.name] == 1,
                            }
                        )
                if node.op_type == "QLinearConv" and len(node.input) >= 9 and node.input[8]:
                    bias = initializers.get(node.input[8])
                    if bias is not None and np.all(numpy_helper.to_array(bias) == 0):
                        findings.append(
                            {
                                "kind": "zero_optional_qlinearconv_bias",
                                "node_index": index,
                                "op": node.op_type,
                                "input": bias.name,
                                "elements": int(np.prod(bias.dims)) if bias.dims else 1,
                                "unique_use": uses[bias.name] == 1,
                            }
                        )
                if node.op_type in {"ConvInteger", "MatMulInteger"}:
                    for input_index in range(2, min(len(node.input), 4)):
                        if not node.input[input_index]:
                            continue
                        zero_point = initializers.get(node.input[input_index])
                        if zero_point is not None and np.all(numpy_helper.to_array(zero_point) == 0):
                            findings.append(
                                {
                                    "kind": "default_integer_zero_point",
                                    "node_index": index,
                                    "op": node.op_type,
                                    "input_index": input_index,
                                    "input": zero_point.name,
                                    "elements": int(np.prod(zero_point.dims)) if zero_point.dims else 1,
                                    "unique_use": uses[zero_point.name] == 1,
                                }
                            )
                if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and len(node.input) >= 3:
                    zero_point = initializers.get(node.input[2])
                    if zero_point is not None and np.all(numpy_helper.to_array(zero_point) == 0):
                        findings.append(
                            {
                                "kind": "default_quantize_zero_point",
                                "node_index": index,
                                "op": node.op_type,
                                "input": zero_point.name,
                                "elements": int(np.prod(zero_point.dims)) if zero_point.dims else 1,
                                "unique_use": uses[zero_point.name] == 1,
                            }
                        )
                if node.op_type == "Pad" and len(node.input) >= 3 and node.input[2]:
                    value = initializers.get(node.input[2])
                    if value is not None and np.all(numpy_helper.to_array(value) == 0):
                        findings.append(
                            {
                                "kind": "zero_pad_value",
                                "node_index": index,
                                "op": node.op_type,
                                "input": value.name,
                                "elements": int(np.prod(value.dims)) if value.dims else 1,
                                "unique_use": uses[value.name] == 1,
                            }
                        )
                if node.op_type == "Slice" and len(node.input) >= 5 and node.input[4]:
                    steps = initializers.get(node.input[4])
                    if steps is not None and np.all(numpy_helper.to_array(steps) == 1):
                        findings.append(
                            {
                                "kind": "default_slice_steps",
                                "node_index": index,
                                "op": node.op_type,
                                "input": steps.name,
                                "elements": int(np.prod(steps.dims)) if steps.dims else 1,
                                "unique_use": uses[steps.name] == 1,
                            }
                        )
                optional_outputs = {
                    "MaxPool": 1,
                    "Dropout": 1,
                    "LayerNormalization": 1,
                }
                first_optional = optional_outputs.get(node.op_type)
                if first_optional is not None and len(node.output) > first_optional:
                    for output_index in range(first_optional, len(node.output)):
                        name = node.output[output_index]
                        if name and uses[name] == 0:
                            findings.append(
                                {
                                    "kind": "unused_optional_output",
                                    "node_index": index,
                                    "op": node.op_type,
                                    "output_index": output_index,
                                    "output": name,
                                }
                            )
            if findings:
                rows.append({"task": task, "findings": findings})
    result = {"source_zip": str(args.zip), "task_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
