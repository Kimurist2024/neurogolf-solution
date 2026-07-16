#!/usr/bin/env python3
"""Build isolated exact probes that omit one unused trailing Split output."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "current/task286.onnx"
OUT = HERE / "exact_probes"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(target: str) -> dict[str, object]:
    model = onnx.load(BASE)
    changed = 0
    for node in model.graph.node:
        for index, name in enumerate(node.output):
            if name == target:
                node.output[index] = ""
                changed += 1
    if changed != 1:
        raise RuntimeError(f"expected one output named {target}, found {changed}")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    data = model.SerializeToString()
    path = OUT / f"task286_omit_{target}.onnx"
    path.write_bytes(data)
    return {
        "target": target,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "serialized_bytes": len(data),
        "full_check": True,
        "strict_data_prop": True,
        "proof": f"{target} has no graph consumer and is not a graph output; omitting only this optional Split output leaves every reachable tensor unchanged",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_sha256": digest(BASE.read_bytes()),
        "probes": [build("V_12"), build("S_12")],
    }
    (HERE / "optional_probe_build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
