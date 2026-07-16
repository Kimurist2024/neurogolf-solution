#!/usr/bin/env python3
"""Try exact SparseTensorProto storage for zero-heavy non-Einsum initializers."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
MIN_SAVING = 10


def dense_from_sparse(item: onnx.SparseTensorProto) -> np.ndarray:
    values = np.asarray(numpy_helper.to_array(item.values))
    indices = np.asarray(numpy_helper.to_array(item.indices), dtype=np.int64)
    dense = np.zeros(tuple(item.dims), dtype=values.dtype)
    dense.reshape(-1)[indices.reshape(-1)] = values.reshape(-1)
    return dense


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"sparse149_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def build(source: onnx.ModelProto, name: str) -> tuple[onnx.ModelProto, dict]:
    model = copy.deepcopy(source)
    position = next(index for index, item in enumerate(model.graph.initializer) if item.name == name)
    original = model.graph.initializer[position]
    array = np.asarray(numpy_helper.to_array(original))
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat != 0).astype(np.int64)
    values = flat[indices]
    if values.size == 0:
        # A zero-length values tensor is rejected by the scorer's positive-dim
        # gate.  Storing one explicit zero still reproduces the dense tensor.
        indices = np.asarray([0], dtype=np.int64)
        values = np.asarray([flat[0]], dtype=array.dtype)
    kept = [item for item in model.graph.initializer if item.name != name]
    # Match scoring.sanitize_model: sparse initializers are not renamed, while
    # node inputs are.  The next canonical safe name keeps the binding intact.
    sparse_name = f"safe_name_{len(kept)}"
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(values, name=sparse_name),
        numpy_helper.from_array(indices, name=""),
        list(array.shape),
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.append(sparse)
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == name:
                node.input[index] = sparse_name
    rebuilt = dense_from_sparse(sparse)
    if not np.array_equal(array, rebuilt, equal_nan=True):
        raise AssertionError("sparse dense reconstruction differs")
    return model, {
        "initializer": name,
        "shape": list(array.shape),
        "dense_elements": int(flat.size),
        "stored_values": int(values.size),
        "parameter_saving": int(flat.size - values.size),
        "sparse_name": sparse_name,
    }


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            uses: dict[str, set[str]] = {}
            for node in model.graph.node:
                for input_name in node.input:
                    uses.setdefault(input_name, set()).add(node.op_type)
            baseline = None
            for init in model.graph.initializer:
                array = np.asarray(numpy_helper.to_array(init))
                nnz = int(np.count_nonzero(array))
                stored = max(1, nnz)
                saving = int(array.size - stored)
                op_uses = sorted(uses.get(init.name, set()))
                if saving < MIN_SAVING or not op_uses or "Einsum" in op_uses:
                    continue
                row = {
                    "task": task, "initializer": init.name,
                    "uses": op_uses, "dense_elements": int(array.size),
                    "nonzero": nnz, "expected_parameter_saving": saving,
                }
                try:
                    candidate, detail = build(model, init.name)
                    row.update(detail)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row.update({
                        "checker_full": True, "strict_data_prop": True,
                        "dense_reconstruction": "BIT_IDENTICAL",
                        "baseline": baseline, "candidate": current,
                        "strict_lower": current["cost"] < baseline["cost"],
                    })
                    if row["strict_lower"]:
                        safe_init = "".join(ch if ch.isalnum() else "_" for ch in init.name)[:40]
                        path = CANDIDATES / f"task{task:03d}_{safe_init}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path)
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row.update({"strict_lower": False, "error": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
    result = {"authority": str(AUTHORITY), "min_saving": MIN_SAVING, "rows": rows}
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "attempts": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "error_count": sum("error" in row for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
