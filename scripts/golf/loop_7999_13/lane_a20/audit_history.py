#!/usr/bin/env python3
"""Audit every retained task191/task216 history model against A20 gates."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from harvest import structure_gate  # noqa: E402
from lib import scoring  # noqa: E402


BASE_COST = {191: 3444, 216: 1511}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result = []
    for dim in value.type.tensor_type.shape.dim:
        result.append(int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?")
    return result


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(task: int, session: ort.InferenceSession) -> dict[str, int]:
    right = wrong = errors = 0
    for examples in scoring.load_examples(task).values():
        for example in examples:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                output = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
                if np.array_equal(output > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
    return {"right": right, "wrong": wrong, "errors": errors}


def audit(task: int, label: str, path: Path, inventory_static: int | None, sources: list[str], baseline: bool = False) -> dict[str, object]:
    model = onnx.load(path)
    ops = Counter(node.op_type for node in model.graph.node)
    row: dict[str, object] = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "inventory_static_cost": inventory_static,
        "baseline_actual_cost": BASE_COST[task],
        "sources": sources,
        "nodes": len(model.graph.node),
        "params": scoring.calculate_params(model),
        "value_info": len(model.graph.value_info),
        "declared_output_shapes": [dims(value) for value in model.graph.output],
        "ops": dict(ops),
        "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
        "opsets": [[item.domain, int(item.version)] for item in model.opset_import],
        "nonstandard_domains": [item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")] + [node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")],
        "banned_ops": [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()],
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": [item.name for item in model.graph.initializer if item.data_location == onnx.TensorProto.EXTERNAL or item.external_data],
        "conv_bias_issues": check_conv_bias(model),
        "lookup_red_flags": {
            "tfidf": int(ops.get("TfIdfVectorizer", 0)),
            "hardmax": int(ops.get("Hardmax", 0)),
            "giant_initializer": sum(int(numpy_helper.to_array(item).size) >= 10_000 for item in model.graph.initializer),
        },
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
        row["nonstatic_inferred"] = [
            value.name
            for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
            if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in value.type.tensor_type.shape.dim)
        ]
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}", nonstatic_inferred=[])
    _, gate_reason, static_floor = structure_gate(path.read_bytes())
    row["shared_structure_gate"] = gate_reason
    row["recomputed_static_floor"] = static_floor
    try:
        trace = runtime_shape_trace(task, model)
        row["runtime_shape_trace"] = {
            "declared_actual_mismatch_count": len(trace["declared_actual_mismatches"]),
            "first_mismatches": trace["declared_actual_mismatches"][:12],
            "runtime_intermediate_bytes": trace["single_example_intermediate_bytes"],
        }
        row["shape_value_cloak_free"] = not trace["declared_actual_mismatches"]
    except Exception as exc:  # noqa: BLE001
        row["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
        row["shape_value_cloak_free"] = False
    for disabled, key in ((True, "known_disable_all"), (False, "known_default")):
        try:
            row[key] = known(task, make_session(model, disabled))
        except Exception as exc:  # noqa: BLE001
            row[key] = {"session_error": f"{type(exc).__name__}: {exc}"}
    with tempfile.TemporaryDirectory(prefix=f"a20_{label}_", dir="/tmp") as workdir:
        try:
            row["official_like_score"] = scoring.score_and_verify(copy.deepcopy(model), task, workdir, label, require_correct=False)
        except Exception as exc:  # noqa: BLE001
            row["official_like_score_error"] = f"{type(exc).__name__}: {exc}"
    profile = row.get("official_like_score")
    row["actual_static_agreement"] = bool(profile and inventory_static is not None and profile["cost"] == inventory_static)
    reasons = []
    if not profile:
        reasons.append("profile_error")
    elif not baseline and profile["cost"] >= BASE_COST[task]:
        reasons.append("not_strictly_cheaper_actual")
    for field in ("full_check", "strict_data_prop", "shape_value_cloak_free"):
        if not row.get(field):
            reasons.append(field)
    for field in ("nonstatic_inferred", "nonstandard_domains", "banned_ops", "functions", "sparse_initializers", "external_initializers", "conv_bias_issues"):
        if row.get(field):
            reasons.append(field)
    if row["max_einsum_inputs"] >= 15:
        reasons.append("giant_einsum")
    if any(row["lookup_red_flags"].values()):
        reasons.append("lookup")
    for field in ("known_disable_all", "known_default"):
        result = row.get(field, {})
        if "session_error" in result or result.get("wrong") != 0 or result.get("errors") != 0 or not result.get("right"):
            reasons.append(field)
    row["pre_fresh_pass"] = not reasons
    row["pre_fresh_reasons"] = sorted(set(reasons))
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows = []
    for task in (191, 216):
        label = f"task{task:03d}_base"
        row = audit(task, label, HERE / "baseline" / f"task{task:03d}.onnx", BASE_COST[task], ["submission_base_7999.13.zip"], baseline=True)
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
    for label, item in manifest["inventory_entries"].items():
        task = int(item["task"])
        row = audit(task, label, HERE / "candidates" / f"{label}.onnx", int(item["static_cost"]), list(item["sources"]))
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "history_audit.json").write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    pending = [row for row in rows if row["pre_fresh_pass"] and not row["label"].endswith("_base")]
    (HERE / "history_audit.json").write_text(json.dumps({"rows": rows, "pending": pending, "complete": True}, indent=2) + "\n")
    print(f"DONE rows={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
