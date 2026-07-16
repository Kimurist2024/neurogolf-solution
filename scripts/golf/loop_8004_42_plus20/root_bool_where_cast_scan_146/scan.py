#!/usr/bin/env python3
"""Profile exact Where(bool,1,0) -> Cast and inverse -> Not+Cast rewrites."""

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
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"boolwhere146_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def drop_dead_initializers(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input)
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Where" or len(node.input) != 3:
                    continue
                yes, no = values.get(node.input[1]), values.get(node.input[2])
                if yes is None or no is None or yes.size != 1 or no.size != 1:
                    continue
                y = float(yes.reshape(-1)[0])
                n = float(no.reshape(-1)[0])
                if (y, n) not in {(1.0, 0.0), (0.0, 1.0)}:
                    continue
                if yes.dtype != no.dtype:
                    continue
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                to_type = helper.np_dtype_to_tensor_dtype(yes.dtype)
                if (y, n) == (1.0, 0.0):
                    replacement = helper.make_node(
                        "Cast", [original.input[0]], list(original.output),
                        name=original.name, to=to_type,
                    )
                    candidate.graph.node[index].CopyFrom(replacement)
                    rewrite = "Where(cond,1,0)->Cast(cond)"
                else:
                    not_output = f"{original.output[0]}__inverse_bool"
                    not_node = helper.make_node("Not", [original.input[0]], [not_output])
                    cast_node = helper.make_node(
                        "Cast", [not_output], list(original.output),
                        name=original.name, to=to_type,
                    )
                    nodes = list(candidate.graph.node)
                    nodes[index:index + 1] = [not_node, cast_node]
                    del candidate.graph.node[:]
                    candidate.graph.node.extend(nodes)
                    rewrite = "Where(cond,0,1)->Cast(Not(cond))"
                dropped = drop_dead_initializers(candidate)
                record = {
                    "task": task, "node_index": index, "rewrite": rewrite,
                    "branch_dtype": str(yes.dtype), "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    baseline = profile(model, task)
                    current = profile(candidate, task)
                    record.update({
                        "baseline": baseline, "candidate": current,
                        "strict_lower": current["cost"] < baseline["cost"],
                    })
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:03d}.onnx"
                        onnx.save(candidate, path)
                        record["path"] = str(path)
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    (HERE / "scan.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": [row for row in rows if "error" in row],
    }, indent=2))


if __name__ == "__main__":
    main()
