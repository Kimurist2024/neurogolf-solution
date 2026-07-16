#!/usr/bin/env python3
"""Replace task377's nonnegative ReduceL1 with exact shape-matched Einsum."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank/task377.onnx")
CANDIDATE = HERE / "candidate/task377.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure(model: onnx.ModelProto) -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    findings = check_conv_bias(model)
    result["conv_bias_findings"] = findings
    result["conv_bias_ub0"] = not findings
    result["pass"] = bool(result.get("full_check") and result.get("strict_data_prop") and result["conv_bias_ub0"])
    return result


def main() -> int:
    base = onnx.load(AUTHORITY)
    candidate = copy.deepcopy(base)
    targets = [(i, n) for i, n in enumerate(candidate.graph.node) if n.name == "areas"]
    if len(targets) != 1:
        raise ValueError(f"areas node count={len(targets)}")
    index, node = targets[0]
    if node.op_type != "ReduceL1" or list(node.input) != ["x", "axes_hw"]:
        raise ValueError(f"unexpected areas node {node}")
    nodes = list(candidate.graph.node)
    nodes[index] = helper.make_node("Einsum", ["x"], list(node.output), name="areas_exact_sum", equation="abcd->ab")
    candidate.graph.ClearField("node")
    candidate.graph.node.extend(nodes)
    kept = [item for item in candidate.graph.initializer if item.name != "axes_hw"]
    if len(kept) + 1 != len(candidate.graph.initializer):
        raise ValueError("axes_hw removal failed")
    candidate.graph.ClearField("initializer")
    candidate.graph.initializer.extend(kept)
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, CANDIDATE)
    bm, bp, bc = cost_of(str(AUTHORITY))
    cm, cp, cc = cost_of(str(CANDIDATE))
    result = {
        "task": 377,
        "authority_sha256": sha(AUTHORITY),
        "candidate_sha256": sha(CANDIDATE),
        "rewrite": "ReduceL1(x, axes=[2,3], keepdims=0) -> Einsum('abcd->ab', x)",
        "valid_domain": "x is binary one-hot input converted to float16; abs(x)=x and each reduced sum is an exactly representable integer in [0,900]",
        "authority_profile": {"memory": bm, "params": bp, "cost": bc},
        "candidate_profile": {"memory": cm, "params": cp, "cost": cc},
        "authority_structure": structure(base),
        "candidate_structure": structure(candidate),
        "strict_lower": cc < bc,
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["strict_lower"] and result["candidate_structure"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
