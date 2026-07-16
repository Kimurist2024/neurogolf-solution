#!/usr/bin/env python3
"""Profile all-input exact DequantizeLinear(scale=1, zero_point=0) -> Cast."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"dequant168_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def drop_dead(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    census = {"nodes": 0, "scale_one": 0, "scale_one_zero_point": 0}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type != "DequantizeLinear" or len(node.input) < 2:
                    continue
                census["nodes"] += 1
                scale = values.get(node.input[1])
                if scale is None or scale.size == 0 or not np.all(scale == 1):
                    continue
                census["scale_one"] += 1
                if len(node.input) >= 3 and node.input[2]:
                    zero = values.get(node.input[2])
                    if zero is None or zero.size == 0 or not np.all(zero == 0):
                        continue
                census["scale_one_zero_point"] += 1
                candidate = copy.deepcopy(model)
                target = candidate.graph.node[index]
                to_type = helper.np_dtype_to_tensor_dtype(scale.dtype)
                replacement = helper.make_node(
                    "Cast", [target.input[0]], list(target.output), name=target.name, to=to_type
                )
                target.CopyFrom(replacement)
                dropped = drop_dead(candidate)
                row: dict[str, Any] = {
                    "task": task,
                    "node_index": index,
                    "scale_shape": list(scale.shape),
                    "scale_dtype": str(scale.dtype),
                    "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row["baseline"] = baseline
                    row["candidate"] = current
                    row["strict_lower"] = current["cost"] < baseline["cost"]
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "census": census,
        "profiles": len(rows),
        "strict_lower_count": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "census": census,
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": len([row for row in rows if "error" in row]),
    }, indent=2))


if __name__ == "__main__":
    main()
