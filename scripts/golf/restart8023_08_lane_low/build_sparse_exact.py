#!/usr/bin/env python3
"""Try standards-compliant exact sparse initializers on the 8023.08 authority.

This is fail-closed: an artifact is retained only when ONNX full checking,
strict data-propagating shape inference, and ORT session construction all pass.
The dense tensor is reproduced exactly; this changes representation only.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8023.08.zip"
AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
TASKS = (156, 238, 297, 324, 341, 398)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dense_to_sparse(model: onnx.ModelProto, names: set[str]) -> list[dict[str, object]]:
    dense = {item.name: item for item in model.graph.initializer}
    rows: list[dict[str, object]] = []
    for name in sorted(names):
        item = dense[name]
        array = np.asarray(numpy_helper.to_array(item))
        flat = array.reshape(-1)
        indices = np.flatnonzero(flat != 0).astype(np.int64)
        values = np.ascontiguousarray(flat[indices])
        sparse = helper.make_sparse_tensor(
            numpy_helper.from_array(values, name=name),
            numpy_helper.from_array(indices, name=name + "__indices"),
            list(array.shape),
        )
        model.graph.sparse_initializer.append(sparse)

        # Represent the initializer truthfully as sparse data.  ONNX's strict
        # inference support for sparse operator inputs is incomplete; the
        # subsequent gates decide whether a particular consumer is supported.
        # Do not add a conflicting tensor/sparse-tensor ValueInfo.  If the
        # ONNX implementation cannot infer the initializer's dense consumer
        # rank directly, strict inference below rejects the variant.
        rows.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dense_params": int(array.size),
                "sparse_params": int(values.size),
                "saving": int(array.size - values.size),
            }
        )
    kept = [item for item in model.graph.initializer if item.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return rows


def params(model: onnx.ModelProto) -> int:
    return int(
        sum(math.prod(item.dims) for item in model.graph.initializer)
        + sum(math.prod(item.values.dims) for item in model.graph.sparse_initializer)
    )


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    out_dir = HERE / "sparse_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
    }
    zero = np.zeros((1, 10, 30, 30), dtype=np.float32)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            selected = set()
            for item in model.graph.initializer:
                array = np.asarray(numpy_helper.to_array(item))
                dense_count = int(array.size)
                nnz = int(np.count_nonzero(array))
                if array.ndim > 0 and dense_count > nnz:
                    selected.add(item.name)
            row: dict[str, object] = {
                "source_sha256": sha256(data),
                "source_params": params(model),
                "selected": sorted(selected),
            }
            try:
                changes = dense_to_sparse(model, selected)
                row["changes"] = changes
                row["candidate_params"] = params(model)
                onnx.checker.check_model(model, full_check=True)
                row["full_check"] = True
                onnx.shape_inference.infer_shapes(
                    model, strict_mode=True, data_prop=True
                )
                row["strict_data_prop"] = True
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    model.SerializeToString(), options,
                    providers=["CPUExecutionProvider"],
                )
                row["session_constructed"] = True
                try:
                    output = session.run(["output"], {"input": zero})[0]
                    row["zero_runtime"] = True
                    row["zero_output_shape"] = list(output.shape)
                except Exception as exc:  # data-dependent execution can fail
                    row["zero_runtime"] = False
                    row["zero_runtime_error"] = f"{type(exc).__name__}: {exc}"
                model.producer_name = "codex-exact-sparse-initializer"
                blob = model.SerializeToString()
                path = out_dir / f"task{task:03d}_sparse_exact.onnx"
                path.write_bytes(blob)
                row["candidate"] = str(path.relative_to(ROOT))
                row["candidate_sha256"] = sha256(blob)
            except Exception as exc:
                row["accepted_preflight"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["accepted_preflight"] = True
            report["tasks"][str(task)] = row
            print(json.dumps({"task": task, **row}), flush=True)
    (HERE / "sparse_exact_build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
