#!/usr/bin/env python3
"""Build a truthful, shape-static equivalent of the task243 authority.

The authority builds two fixed relation matrices through Shape/Reshape plus
broadcasting while declaring all of those tensors as 1-element values.  The
runtime tensors are actually 30x30 and 10x10.  This script materializes those
input-independent matrices as initializers and keeps the original terminal
Einsum byte-for-byte.  The resulting graph has no shape cloak and its declared
output is the real 1x10x30x30 runtime shape.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
OUTPUT = HERE / "candidates/task243_truthful_constant.onnx"
EVIDENCE = HERE / "build.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> None:
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task243.onnx")
    authority = onnx.load_model_from_string(authority_data)

    expected_prefix = [
        "Shape",
        "Reshape",
        "Flatten",
        "Cast",
        "Cast",
        "LeakyRelu",
        "LessOrEqual",
        "CastLike",
        "Slice",
        "Flatten",
        "Cast",
        "LeakyRelu",
        "Cast",
        "LessOrEqual",
        "CastLike",
        "CastLike",
    ]
    actual_prefix = [node.op_type for node in authority.graph.node[:-1]]
    if actual_prefix != expected_prefix or authority.graph.node[-1].op_type != "Einsum":
        raise RuntimeError("task243 authority topology changed; refusing stale rewrite")

    r0_tensor = next(init for init in authority.graph.initializer if init.name == "r0")
    r0 = numpy_helper.to_array(r0_tensor)
    if r0.dtype != np.int8 or r0.shape != (30,):
        raise RuntimeError(f"unexpected r0: dtype={r0.dtype}, shape={r0.shape}")

    # Reproduce the fixed prelude exactly.  Comparisons only depend on ordering;
    # float16 is retained here so boundary behavior matches the ONNX nodes.
    row = r0.reshape(30, 1).astype(np.float16)
    rng_raw = r0.astype(np.float16)
    rngp = np.where(rng_raw >= 0, rng_raw, rng_raw * np.float16(0.985)).astype(
        np.float16
    )
    relation_30 = (row <= rngp).astype(np.float32)

    colors = r0[20:30]
    color_row = colors.reshape(10, 1).astype(np.float16)
    color_raw = colors.astype(np.float16)
    color_key = np.where(
        color_raw >= 0,
        color_raw,
        color_raw * np.float16(-0.0078125),
    ).astype(np.float16)
    relation_10 = (color_row <= color_key).astype(np.float32)
    weights = colors.astype(np.float32)

    terminal = onnx.NodeProto()
    terminal.CopyFrom(authority.graph.node[-1])
    terminal.name = "truthful_terminal_einsum"

    graph = helper.make_graph(
        nodes=[terminal],
        name="task243_truthful_constant_relations",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 10, 30, 30]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, 10, 30, 30]
            )
        ],
        initializer=[
            numpy_helper.from_array(relation_10, "CB"),
            numpy_helper.from_array(weights, "w_dyn"),
            numpy_helper.from_array(relation_30, "L"),
        ],
    )
    candidate = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(opset.domain, opset.version)
            for opset in authority.opset_import
        ],
        producer_name="codex-task243-truthful-repair",
        ir_version=authority.ir_version,
    )
    candidate.domain = authority.domain
    candidate.model_version = authority.model_version

    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        candidate, strict_mode=True, data_prop=True
    )
    if dims(inferred.graph.output[0]) != [1, 10, 30, 30]:
        raise RuntimeError(
            f"truthful output inference failed: {dims(inferred.graph.output[0])}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUTPUT)
    output_data = OUTPUT.read_bytes()
    EVIDENCE.write_text(
        json.dumps(
            {
                "task": 243,
                "authority": str(AUTHORITY.relative_to(ROOT)),
                "authority_zip_sha256": sha256(AUTHORITY.read_bytes()),
                "authority_member_sha256": sha256(authority_data),
                "candidate": str(OUTPUT.relative_to(ROOT)),
                "candidate_sha256": sha256(output_data),
                "authority_declared_shapes": {
                    value.name: dims(value)
                    for value in [
                        *authority.graph.input,
                        *authority.graph.value_info,
                        *authority.graph.output,
                    ]
                },
                "truthful_runtime_shapes": {
                    "CB": list(relation_10.shape),
                    "w_dyn": list(weights.shape),
                    "L": list(relation_30.shape),
                    "output": [1, 10, 30, 30],
                },
                "relation_sums": {
                    "CB": int(relation_10.sum()),
                    "L": int(relation_30.sum()),
                },
                "static_checks": {
                    "full_check": True,
                    "strict_shape_inference_data_prop": True,
                },
                "nodes": len(candidate.graph.node),
                "params_by_shape": 10 * 10 + 10 + 30 * 30,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
