#!/usr/bin/env python3
"""Build task133's exact discrete-domain Mul-to-Selu parameter shave."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank/task133.onnx")
CANDIDATE = HERE / "candidate/task133.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    findings = check_conv_bias(model)
    row["conv_bias_findings"] = findings
    row["conv_bias_ub0"] = not findings
    row["pass"] = bool(row.get("full_check") and row.get("strict_data_prop") and row["conv_bias_ub0"])
    return row


def main() -> int:
    model = onnx.load(AUTHORITY)
    result = copy.deepcopy(model)
    half = next(item for item in result.graph.initializer if item.name == "half_f")
    half_value = float(np.asarray(numpy_helper.to_array(half)).reshape(-1)[0])
    if half_value != 0.5:
        raise ValueError(f"unexpected half_f={half_value}")
    uses = []
    for index, node in enumerate(result.graph.node):
        for input_index, name in enumerate(node.input):
            if name == "half_f":
                uses.append((index, input_index, node))
    if len(uses) != 1:
        raise ValueError(f"half_f use count={len(uses)}")
    index, _input_index, node = uses[0]
    if node.op_type != "Mul" or set(node.input) != {"scale_m1", "half_f"}:
        raise ValueError(f"unexpected half_f consumer {node}")
    # For x>=0, Selu(x, gamma=.5)=x*.5. The only negative reachable value is
    # x=-1; alpha is chosen so .5*alpha*(exp(-1)-1)=-.5 exactly before the
    # final binary16 cast.
    alpha = 1.0 / (1.0 - math.exp(-1.0))
    replacement = helper.make_node(
        "Selu",
        ["scale_m1"],
        list(node.output),
        name="exact_discrete_scale_minus_one_half",
        alpha=alpha,
        gamma=0.5,
    )
    nodes = list(result.graph.node)
    nodes[index] = replacement
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    kept = [item for item in result.graph.initializer if item.name != "half_f"]
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(kept)
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, CANDIDATE)
    bm, bp, bc = cost_of(str(AUTHORITY))
    cm, cp, cc = cost_of(str(CANDIDATE))
    output = {
        "task": 133,
        "authority_sha256": digest(AUTHORITY),
        "candidate_sha256": digest(CANDIDATE),
        "alpha": alpha,
        "gamma": 0.5,
        "reachable_scale_m1": [-1, 0, 1, 2, 3],
        "authority_profile": {"memory": bm, "params": bp, "cost": bc},
        "candidate_profile": {"memory": cm, "params": cp, "cost": cc},
        "strict_lower": cc < bc,
        "authority_structure": structure(model),
        "candidate_structure": structure(result),
    }
    (HERE / "build.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["strict_lower"] and output["candidate_structure"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
