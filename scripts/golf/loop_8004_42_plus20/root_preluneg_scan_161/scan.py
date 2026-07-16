#!/usr/bin/env python3
"""Exhaust task086 shared PRelu slope=-1 replacements with exact Abs."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"preluneg161_{label}_") as wd:
        path = Path(wd) / "task086.onnx"
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
    with zipfile.ZipFile(AUTHORITY) as archive:
        model = onnx.load_model_from_string(archive.read("task086.onnx"))
    values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    targets = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "PRelu" or len(node.input) != 2:
            continue
        slope = values.get(node.input[1])
        if slope is not None and slope.size == 1 and float(slope.reshape(-1)[0]) == -1.0:
            targets.append(index)
    baseline = profile(model, "base")
    rows = []
    for size in range(1, len(targets) + 1):
        for subset in combinations(targets, size):
            for op in ("Abs", "LeakyRelu"):
                candidate = copy.deepcopy(model)
                for index in subset:
                    original = candidate.graph.node[index]
                    kwargs = {"alpha": -1.0} if op == "LeakyRelu" else {}
                    replacement = helper.make_node(
                        op, [original.input[0]], list(original.output),
                        name=original.name, **kwargs,
                    )
                    candidate.graph.node[index].CopyFrom(replacement)
                dropped = drop_dead(candidate)
                row = {"subset": list(subset), "op": op, "dropped_initializers": dropped}
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    current = profile(candidate, f"{op}_{'_'.join(map(str, subset))}")
                    row.update(baseline=baseline, candidate=current, strict_lower=current["cost"] < baseline["cost"])
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task086_{op}_{'_'.join(map(str, subset))}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {"authority": str(AUTHORITY.relative_to(ROOT)), "targets": targets,
               "baseline": baseline, "profiles": len(rows),
               "strict_lower_count": sum(bool(r.get("strict_lower")) for r in rows), "rows": rows}
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"targets": targets, "profiles": len(rows),
                      "strict_lower": [r for r in rows if r.get("strict_lower")],
                      "errors": len([r for r in rows if "error" in r])}, indent=2))


if __name__ == "__main__":
    main()
