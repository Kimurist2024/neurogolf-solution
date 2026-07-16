#!/usr/bin/env python3
"""Scan exact omission of all-zero optional Conv-family/Gemm biases."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
BIAS_INDEX = {"Conv": 2, "ConvTranspose": 2, "QLinearConv": 8, "Gemm": 2}


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"zerobias162_{task:03d}_") as wd:
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
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            baseline = None
            for index, node in enumerate(model.graph.node):
                bias_index = BIAS_INDEX.get(node.op_type)
                if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
                    continue
                bias = values.get(node.input[bias_index])
                if bias is None or bias.size == 0 or not np.all(bias == 0):
                    continue
                candidate = copy.deepcopy(model)
                target = candidate.graph.node[index]
                inputs = list(target.input)
                inputs[bias_index] = ""
                while inputs and inputs[-1] == "":
                    inputs.pop()
                del target.input[:]
                target.input.extend(inputs)
                dropped = drop_dead(candidate)
                row = {"task": task, "node_index": index, "op": node.op_type,
                       "bias": node.input[bias_index], "bias_shape": list(bias.shape),
                       "bias_elements": int(bias.size), "dropped_initializers": dropped}
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row.update(baseline=baseline, candidate=current, strict_lower=current["cost"] < baseline["cost"])
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}_{node.op_type}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {"authority": str(AUTHORITY.relative_to(ROOT)), "profiles": len(rows),
               "strict_lower_count": sum(bool(r.get("strict_lower")) for r in rows), "rows": rows}
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "strict_lower": [r for r in rows if r.get("strict_lower")],
                      "errors": len([r for r in rows if "error" in r])}, indent=2))


if __name__ == "__main__":
    main()
