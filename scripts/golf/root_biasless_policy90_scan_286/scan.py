#!/usr/bin/env python3
"""Build and preliminary-audit clean POLICY90 Conv-family bias omissions."""

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TARGETS = {160: "b1m", 193: "B", 275: "GB"}
BIAS_INDEX = {"Conv": 2, "ConvTranspose": 2, "QLinearConv": 8, "Gemm": 2}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from audit_candidates import runtime_shape_trace  # noqa: E402
import screen_all  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(data: bytes, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"biasless286_{task:03d}_", dir="/tmp") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def remove_all_bias_uses(source: onnx.ModelProto, bias_name: str) -> tuple[onnx.ModelProto, list[dict]]:
    model = copy.deepcopy(source)
    changes = []
    for index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name != bias_name:
                continue
            if BIAS_INDEX.get(node.op_type) != input_index:
                raise RuntimeError(f"unsupported non-bias use at node {index} input {input_index}")
            before = list(node.input)
            inputs = list(node.input)
            inputs[input_index] = ""
            while inputs and inputs[-1] == "":
                inputs.pop()
            del node.input[:]
            node.input.extend(inputs)
            changes.append({"node_index": index, "op": node.op_type, "before": before, "after": list(node.input)})
    if not changes:
        raise RuntimeError("bias has no uses")
    live = Counter(name for node in model.graph.node for name in node.input if name)
    kept = [item for item in model.graph.initializer if live[item.name] > 0]
    dropped = {item.name for item in model.graph.initializer if live[item.name] == 0}
    if bias_name not in dropped:
        raise RuntimeError("bias initializer remains live")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model, changes


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    candidate_dir = HERE / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    authority_data = AUTHORITY.read_bytes()
    if digest(authority_data) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, bias_name in sorted(TARGETS.items()):
            member = archive.read(f"task{task:03d}.onnx")
            source = onnx.load_model_from_string(member)
            source_ub = check_conv_bias(source)
            row = {
                "task": task,
                "bias": bias_name,
                "authority_sha256": digest(member),
                "authority_cost": profile(member, task),
                "authority_conv_bias_findings": source_ub,
            }
            try:
                candidate, changes = remove_all_bias_uses(source, bias_name)
                onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
                onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
                data = candidate.SerializeToString()
                candidate_cost = profile(data, task)
                row.update(
                    changes=changes,
                    candidate_sha256=digest(data),
                    candidate_cost=candidate_cost,
                    strict_lower=candidate_cost["cost"] < row["authority_cost"]["cost"],
                    candidate_conv_bias_findings=check_conv_bias(candidate),
                )
                if source_ub or row["candidate_conv_bias_findings"] or not row["strict_lower"]:
                    row.update(status="REJECT_STATIC")
                    rows.append(row)
                    continue
                path = candidate_dir / f"task{task:03d}_drop_{bias_name}.onnx"
                path.write_bytes(data)
                row["path"] = str(path.relative_to(ROOT))
                dual = screen_all.known_dual(task, data)
                row["known_dual"] = dual
                total_ok = all(
                    mode.get("right", 0) + mode.get("wrong", 0) > 0
                    and mode.get("right", 0) / (mode.get("right", 0) + mode.get("wrong", 0)) >= 0.90
                    and mode.get("errors", 0) == 0
                    and not mode.get("session_error")
                    for mode in dual.values()
                )
                if not total_ok:
                    row.update(status="REJECT_KNOWN_POLICY90")
                    rows.append(row)
                    continue
                trace = runtime_shape_trace(task, candidate)
                row["runtime_shape_trace"] = trace
                if trace.get("declared_actual_mismatches"):
                    row.update(status="REJECT_SHAPE_CLOAK")
                else:
                    row.update(status="PRELIMINARY_KNOWN_PASS_NEEDS_FRESH")
            except Exception as exc:  # noqa: BLE001
                row.update(status="REJECT_EXCEPTION", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    output = {
        "status": "PRELIMINARY_ONLY_DO_NOT_STAGE",
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": digest(authority_data),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([{key: row.get(key) for key in (
        "task", "authority_cost", "candidate_cost", "known_dual", "status", "error"
    )} for row in rows], indent=2))


if __name__ == "__main__":
    main()
