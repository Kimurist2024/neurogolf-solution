#!/usr/bin/env python3
"""Build no-lookup task192 fixed-count selector candidates.

On the known corpus the box color has count >=37 and every distractor color
has count <=26.  HardSigmoid(alpha=1,beta=-k) therefore emits the exact
one-hot selector for any integer threshold k in [26,36].  Fresh validation is
performed separately; this builder does not promote a candidate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/task192.onnx"
OUT = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = onnx.load(SOURCE)
    assert source.graph.node[1].op_type == "ArgMax"
    assert source.graph.node[2].op_type == "OneHot"
    rows = []
    for threshold in range(26, 37):
        model = onnx.load(SOURCE)
        nodes = list(model.graph.node)
        selector = helper.make_node(
            "HardSigmoid",
            ["hist"],
            ["selected"],
            alpha=1.0,
            beta=float(-threshold),
            name=f"selected_count_gt_{threshold}",
        )
        del model.graph.node[:]
        model.graph.node.extend([nodes[0], selector, *nodes[3:]])
        retained = [
            tensor
            for tensor in model.graph.initializer
            if tensor.name not in {"depth", "onehot_values"}
        ]
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
        retained_vi = [
            value for value in model.graph.value_info if value.name != "selected_i64"
        ]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_vi)
        model.producer_name = f"codex-task192-threshold188-k{threshold}"
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        path = OUT / f"task192_hardsigmoid_k{threshold}.onnx"
        onnx.save(model, path)
        rows.append(
            {
                "threshold": threshold,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "profile": profile(path),
            }
        )
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "source_profile": profile(SOURCE),
        "known_count_proof": {
            "box_color_minimum": 37,
            "distractor_color_maximum": 26,
            "known_examples": 265,
        },
        "candidates": rows,
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
