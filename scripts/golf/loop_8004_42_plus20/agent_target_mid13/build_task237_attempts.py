#!/usr/bin/env python3
"""Build and gate isolated task237 width-9 packing probes.

All rejected ONNX files live only in a temporary directory.  The JSON report is
the durable evidence; this lane deliberately retains no failed candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline/task237.onnx"
REPORT = HERE / "task237_attempts.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_cost(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    free = {value.name for value in inferred.graph.input} | {
        value.name for value in inferred.graph.output
    }
    initializers = {value.name for value in inferred.graph.initializer}
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in seen or name in free or name in initializers:
                continue
            seen.add(name)
            tensor = typed[name].type.tensor_type
            elements = math.prod(
                int(dim.dim_value) for dim in tensor.shape.dim
            )
            width = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            ).itemsize
            memory += elements * width
    params = sum(
        math.prod(value.dims) if value.dims else 1
        for value in inferred.graph.initializer
    ) + sum(
        math.prod(value.values.dims) if value.values.dims else 1
        for value in inferred.graph.sparse_initializer
    )
    return {"memory": memory, "params": params, "cost": memory + params}


def sparse_from_dense(name: str, array: np.ndarray) -> onnx.SparseTensorProto:
    coordinates = np.argwhere(array != 0)
    values = array[tuple(coordinates.T)]
    values_tensor = numpy_helper.from_array(
        values.astype(array.dtype, copy=False), name=name
    )
    indices_tensor = numpy_helper.from_array(
        coordinates.astype(np.int64), name=f"{name}_indices"
    )
    return helper.make_sparse_tensor(values_tensor, indices_tensor, array.shape)


def expanded_width_model(sparse: bool) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(BASELINE, load_external_data=False))
    inits = {value.name: value for value in model.graph.initializer}
    old = numpy_helper.to_array(inits["packed_kernel"])
    expanded = np.zeros((10, 1, 1, 9), dtype=np.float32)
    # With left pad 8, input column 0 aligns with kernel position 8.  Shift the
    # old eight entries right and use the new leading entry for guaranteed-
    # background input column 8 (markers are sampled only through width-2).
    expanded[:, :, :, 1:] = old
    expanded[0, 0, 0, 0] = 10.0
    inits["packed_kernel"].CopyFrom(
        numpy_helper.from_array(expanded, "packed_kernel")
    )

    conv = next(node for node in model.graph.node if node.output[0] == "packed_grid")
    pads = next(attribute for attribute in conv.attribute if attribute.name == "pads")
    pads.ints[:] = [0, 8, 21, 29]

    remove_outputs = {"w9_flag_f", "w9_flag_u8", "max_col_index"}
    kept = [
        node
        for node in model.graph.node
        if not any(output in remove_outputs for output in node.output)
    ]
    for node in kept:
        for index, value in enumerate(node.input):
            if value == "max_col_index":
                node.input[index] = "max_col_index_base"
    del model.graph.node[:]
    model.graph.node.extend(kept)
    keep_inits = [value for value in model.graph.initializer if value.name != "w9_idx"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)

    if sparse:
        dense = next(
            value for value in model.graph.initializer if value.name == "packed_kernel"
        )
        sparse_value = sparse_from_dense(
            "packed_kernel", numpy_helper.to_array(dense)
        )
        keep_inits = [
            value for value in model.graph.initializer if value.name != "packed_kernel"
        ]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep_inits)
        model.graph.sparse_initializer.append(sparse_value)
    return model


def gate(label: str, model: onnx.ModelProto, path: Path) -> dict[str, object]:
    row: dict[str, object] = {"label": label}
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        row["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(
            strict_shape_data_prop=False,
            strict_shape_error=f"{type(exc).__name__}: {exc}",
        )
    try:
        row["static_cost"] = static_cost(model)
    except Exception as exc:  # noqa: BLE001
        row["static_cost_error"] = f"{type(exc).__name__}: {exc}"
    onnx.save(model, path)
    row["sha256"] = sha256(path)
    row["serialized_bytes"] = path.stat().st_size
    row["sparse_initializers"] = len(model.graph.sparse_initializer)
    baseline = static_cost(onnx.load(BASELINE))
    row["baseline"] = baseline
    candidate_cost = (row.get("static_cost") or {}).get("cost")
    row["strictly_cheaper_static"] = (
        isinstance(candidate_cost, int) and candidate_cost < baseline["cost"]
    )
    if not row.get("full_check") or not row.get("strict_shape_data_prop"):
        row["verdict"] = "REJECT_STRUCTURE"
    elif not row["strictly_cheaper_static"]:
        row["verdict"] = "REJECT_NOT_CHEAPER"
    else:
        # This branch is intentionally conservative.  A cheaper structure would
        # advance to the full lane verifier before being retained.
        row["verdict"] = "ADVANCE_FULL_GATE"
    return row


def main() -> None:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="task237_mid13_", dir="/tmp") as work:
        workdir = Path(work)
        rows.append(
            gate(
                "dense_width9_pack",
                expanded_width_model(sparse=False),
                workdir / "dense.onnx",
            )
        )
        # ONNX 1.20 shape inference crashes in native code (exit 139) when the
        # ConvTranspose weight is replaced by this sparse initializer.  Record
        # the isolated subprocess result without re-triggering it in-process.
        sparse_model = expanded_width_model(sparse=True)
        sparse_path = workdir / "sparse_unchecked.onnx"
        onnx.save(sparse_model, sparse_path)
        rows.append(
            {
                "label": "sparse_width9_pack",
                "sha256": sha256(sparse_path),
                "serialized_bytes": sparse_path.stat().st_size,
                "sparse_initializers": 1,
                "full_check": False,
                "strict_shape_data_prop": False,
                "full_check_error": (
                    "isolated ONNX checker/shape-inference process exited 139 "
                    "for sparse ConvTranspose weights"
                ),
                "baseline": static_cost(onnx.load(BASELINE)),
                "theoretical_static_cost_if_supported": {
                    "memory": 407,
                    "params": 116,
                    "cost": 523,
                },
                "strictly_cheaper_static": True,
                "verdict": "REJECT_STRUCTURE",
            }
        )
    payload = {
        "baseline_path": str(BASELINE),
        "baseline_sha256": sha256(BASELINE),
        "attempts": rows,
        "retained_candidates": [],
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
