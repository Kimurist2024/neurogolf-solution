#!/usr/bin/env python3
"""Store zero-heavy dense initializers as Constant(sparse_value=...)."""

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
    with tempfile.TemporaryDirectory(prefix=f"sparseconst151_{task:03d}_") as wd:
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
        indices = np.asarray([0], dtype=np.int64)
        values = np.asarray([flat[0]], dtype=array.dtype)
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(values, name=f"{name}__values"),
        numpy_helper.from_array(indices, name=f"{name}__indices"),
        list(array.shape),
    )
    constant = helper.make_node(
        "Constant", [], [name], name=f"{name}__sparse_constant",
        sparse_value=sparse,
    )
    del model.graph.initializer[position]
    old_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([constant, *old_nodes])
    if not np.array_equal(array, dense_from_sparse(sparse), equal_nan=True):
        raise AssertionError("sparse dense reconstruction differs")
    return model, {
        "initializer": name,
        "shape": list(array.shape),
        "dense_elements": int(flat.size),
        "stored_values": int(values.size),
        "parameter_saving": int(flat.size - values.size),
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
                if saving < MIN_SAVING or init.name not in uses:
                    continue
                row = {
                    "task": task, "initializer": init.name,
                    "uses": sorted(uses[init.name]),
                    "dense_elements": int(array.size), "nonzero": nnz,
                    "expected_parameter_saving": saving,
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
