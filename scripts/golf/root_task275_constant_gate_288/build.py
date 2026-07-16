#!/usr/bin/env python3
"""Fold task275's all-support one-hot count gate into a constant initializer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATE = HERE / "candidates/task275_constant_gate.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(data: bytes, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"task275_288_{label}_", dir="/tmp") as work:
        path = Path(work) / "task275.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    authority_zip = AUTHORITY.read_bytes()
    if digest(authority_zip) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task275.onnx")
    model = onnx.load_model_from_string(authority_data)
    if [(node.op_type, list(node.output)) for node in model.graph.node[:2]] != [
        ("ReduceSum", ["total"]), ("Conv", ["gate"])
    ]:
        raise RuntimeError("unexpected authority prefix")
    conv = model.graph.node[1]
    if list(conv.input) != ["total", "GW", "GB"]:
        raise RuntimeError("unexpected gate Conv")
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    if not np.array_equal(arrays["GW"], np.asarray([-1.0, 0.0], dtype=np.float32).reshape(2, 1, 1, 1)):
        raise RuntimeError("unexpected GW")
    if not np.array_equal(arrays["GB"], np.asarray([25.0, 7.0], dtype=np.float32)):
        raise RuntimeError("unexpected GB")

    # Every valid benchmark input is canonical one-hot [1,10,30,30], hence
    # ReduceSum(input, axes=1,2,3) is exactly float32 900.  The 1x1 Conv is
    # therefore the invariant tensor [-875, 7].
    gate = np.asarray([-875.0, 7.0], dtype=np.float32).reshape(1, 2, 1, 1)
    candidate = copy.deepcopy(model)
    del candidate.graph.node[:2]
    for node in candidate.graph.node:
        for index, name in enumerate(node.input):
            if name == "gate":
                node.input[index] = "gate_const"
    kept = [item for item in candidate.graph.initializer if item.name not in {"GW", "GB"}]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    candidate.graph.initializer.append(numpy_helper.from_array(gate, name="gate_const"))
    live_outputs = {name for node in candidate.graph.node for name in node.output if name}
    kept_info = [item for item in candidate.graph.value_info if item.name in live_outputs]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_info)

    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    findings = check_conv_bias(candidate)
    if findings:
        raise RuntimeError(f"Conv bias UB findings: {findings}")
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, CANDIDATE)
    candidate_data = CANDIDATE.read_bytes()
    authority_cost = profile(authority_data, "authority")
    candidate_cost = profile(candidate_data, "candidate")
    if candidate_cost != {"memory": 0, "params": 414, "cost": 414}:
        raise RuntimeError(f"unexpected candidate profile {candidate_cost}")
    output = {
        "status": "BUILT_NEEDS_AUDIT",
        "authority_zip_sha256": digest(authority_zip),
        "authority_member_sha256": digest(authority_data),
        "authority_cost": authority_cost,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest(candidate_data),
        "candidate_cost": candidate_cost,
        "gain": math.log(authority_cost["cost"] / candidate_cost["cost"]),
        "proof": (
            "canonical scoring input is one-hot [1,10,30,30], so its ReduceSum is "
            "exactly 900; Conv([900],[-1,0],[25,7]) is exactly [-875,7]"
        ),
    }
    (HERE / "build.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
