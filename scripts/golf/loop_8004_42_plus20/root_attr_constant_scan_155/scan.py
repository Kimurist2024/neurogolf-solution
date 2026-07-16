#!/usr/bin/env python3
"""Scan exact attribute-carrier rewrites on immutable 8009.46 payloads.

Families:
* scalar float32 PRelu(x, slope) -> LeakyRelu(x, alpha=slope)
* CastLike(x, initializer) -> Cast(x, to=initializer_dtype)

Dead initializers are removed before official-like profiling.  Strict-lower
models are retained only after full checker and strict data propagation pass.
Runtime/semantic admission is deliberately a later gate.
"""

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
    with tempfile.TemporaryDirectory(prefix=f"attr155_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def drop_dead_initializers(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input)
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def make_rewrite(
    model: onnx.ModelProto,
    index: int,
    values: dict[str, np.ndarray],
) -> tuple[onnx.ModelProto, str] | None:
    node = model.graph.node[index]
    if node.op_type == "PRelu" and len(node.input) == 2:
        slope = values.get(node.input[1])
        if slope is None or slope.size != 1 or slope.dtype != np.dtype("float32"):
            return None
        alpha = float(slope.reshape(-1)[0])
        if not np.isfinite(alpha):
            return None
        candidate = copy.deepcopy(model)
        original = candidate.graph.node[index]
        replacement = helper.make_node(
            "LeakyRelu",
            [original.input[0]],
            list(original.output),
            name=original.name,
            alpha=alpha,
        )
        candidate.graph.node[index].CopyFrom(replacement)
        return candidate, f"PRelu(scalar={alpha!r})->LeakyRelu(alpha)"

    if node.op_type == "CastLike" and len(node.input) == 2:
        target = values.get(node.input[1])
        if target is None:
            return None
        candidate = copy.deepcopy(model)
        original = candidate.graph.node[index]
        to_type = helper.np_dtype_to_tensor_dtype(target.dtype)
        replacement = helper.make_node(
            "Cast",
            [original.input[0]],
            list(original.output),
            name=original.name,
            to=to_type,
        )
        candidate.graph.node[index].CopyFrom(replacement)
        return candidate, f"CastLike(init:{target.dtype})->Cast(to={to_type})"
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            baseline: dict | None = None
            for index, node in enumerate(model.graph.node):
                if node.op_type not in {"PRelu", "CastLike"}:
                    continue
                built = make_rewrite(model, index, values)
                if built is None:
                    continue
                candidate, rewrite = built
                dropped = drop_dead_initializers(candidate)
                row = {
                    "task": task,
                    "node_index": index,
                    "source_op": node.op_type,
                    "rewrite": rewrite,
                    "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(
                        candidate, strict_mode=True, data_prop=True
                    )
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row.update(
                        baseline=baseline,
                        candidate=current,
                        strict_lower=current["cost"] < baseline["cost"],
                    )
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}_{node.op_type}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(REPO))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

    payload = {
        "authority": str(AUTHORITY.relative_to(REPO)),
        "profiles": len(rows),
        "strict_lower_count": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": [row for row in rows if "error" in row],
    }, indent=2))


if __name__ == "__main__":
    main()
