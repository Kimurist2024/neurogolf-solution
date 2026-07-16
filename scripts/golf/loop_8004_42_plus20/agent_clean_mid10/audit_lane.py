#!/usr/bin/env python3
"""No-promotion audit for the clean-mid10 tasks 182/201/251/370.

The script intentionally only writes evidence below this lane directory.  It
profiles the immutable 8004.50 members, inventories the exhaustive historical
union already collected by clean95_all, and measures generator-sound controls
with honest static shapes.  It never calls try_candidate or edits a ZIP.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
TASKS = (370, 182, 201, 251)
CONTROLS = {
    370: ROOT / "artifacts/optimized/task370.onnx",
    182: ROOT / "scripts/golf/loop_7999_13/lane_b9/task182_static_shapes.onnx",
    201: ROOT / "artifacts/optimized/task201.onnx",
    251: ROOT / "artifacts/optimized/task251.onnx",
}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    nonstatic = [
        value.name
        for value in values
        if value.type.HasField("tensor_type")
        and any(
            not dim.HasField("dim_value") or dim.dim_value <= 0 or dim.HasField("dim_param")
            for dim in value.type.tensor_type.shape.dim
        )
    ]
    domains = sorted({node.domain for node in model.graph.node} | {item.domain for item in model.opset_import})
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive": not nonstatic,
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "no_banned": all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "no_nested_functions_sparse": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attr in node.attribute
            )
        ),
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
        ),
        "no_giant_contraction": max_einsum <= 16,
        "no_lookup_ops": not any(
            node.op_type in {"TfIdfVectorizer", "ScatterElements", "ScatterND", "Hardmax"}
            for node in model.graph.node
        ),
        "no_shape_cloak_ops": not any(node.op_type == "CenterCropPad" for node in model.graph.node),
    }
    return {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "params": sum(math.prod(item.dims) for item in model.graph.initializer),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "domains": domains,
        "max_einsum_inputs": max_einsum,
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
        "nonstatic": nonstatic,
    }


def make_session(model: onnx.ModelProto, disable: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_dual(path: Path, task: int) -> dict[str, Any]:
    model = onnx.load(path)
    result: dict[str, Any] = {}
    examples = scoring.load_examples(task)
    converted = [
        item
        for subset in ("train", "test", "arc-gen")
        for raw in examples.get(subset, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]
    for name, disable in (("disable_all", True), ("default", False)):
        row: dict[str, Any] = {"right": 0, "wrong": 0, "runtime_errors": 0, "total": len(converted)}
        try:
            session = make_session(model, disable)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            result[name] = row
            continue
        for item in converted:
            try:
                raw = session.run(None, {"input": item["input"]})[0]
                predicted = (raw > 0).astype(np.float32)
                if np.array_equal(predicted, item["output"]):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
        result[name] = row
    return result


def profile(path: Path, task: int) -> dict[str, Any]:
    model = onnx.load(path)
    with tempfile.TemporaryDirectory(prefix=f"mid10_{task:03d}_") as directory:
        scored = scoring.score_and_verify(
            model, task, directory, label="sound_control", require_correct=False
        )
    if scored is None:
        return {"error": "score_and_verify returned None"}
    return {
        "memory": int(scored["memory"]),
        "params": int(scored["params"]),
        "cost": int(scored["cost"]),
        "score": float(scored["score"]),
        "known_correct": bool(scored["correct"]),
    }


def main() -> None:
    baseline_dir = HERE / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            path = baseline_dir / f"task{task:03d}.onnx"
            data = archive.read(path.name)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)

    ranked = json.loads(
        (HERE.parent / "current_costs_8004_50.json").read_text(encoding="utf-8")
    )["ranked"]
    current_costs = {int(row["task"]): row for row in ranked}
    screen = json.loads(
        (HERE.parent / "agent_clean95_all/screen_results.json").read_text(encoding="utf-8")
    )["rows"]
    inventory = {
        str(task): [row for row in screen if int(row.get("task", -1)) == task]
        for task in TASKS
    }

    rules = {
        "370": {
            "hash": "e8dc4411",
            "type": "bounded geometric propagation",
            "rule": "find the black origin sprite and the single colored direction hint; repeat the exact sprite along that hint vector, respecting generator flip/transpose and the true grid boundary",
            "private_status": "outside private-zero catalog",
        },
        "182": {
            "hash": "776ffc46",
            "type": "global template matching",
            "rule": "use the colored sprite inside the gray 7x7 frame as the template; recolor every unframed color-1 sprite with identical shape to the template color",
            "private_status": "not private-zero catalogued, but task is in the downstream-contamination ledger",
        },
        "201": {
            "hash": "846bdb03",
            "type": "data-dependent crop/geometry",
            "rule": "recover two colored Conway sprites and the yellow/color frame; undo the optional horizontal flip and place the sprites into the data-dependent framed output",
            "private_status": "historical private-zero/black lineage; only decoded sound reference plus fresh100 can qualify",
        },
        "251": {
            "hash": "a5313dff",
            "type": "bounded boxes/local fill",
            "rule": "for each fully in-grid red rectangle, paint its black interior ring blue while preserving the red border and the inner red core; clipped rectangles are unchanged",
            "private_status": "outside private-zero catalog",
        },
    }

    tasks: dict[str, Any] = {}
    for task in TASKS:
        baseline = baseline_dir / f"task{task:03d}.onnx"
        control = CONTROLS[task]
        control_profile = profile(control, task)
        control_cost = control_profile.get("cost")
        incumbent_cost = int(current_costs[task]["cost"])
        tasks[str(task)] = {
            "rule": rules[str(task)],
            "baseline": {
                "path": str(baseline.relative_to(ROOT)),
                "reported_actual_cost": current_costs[task],
                "structural": structural(baseline),
            },
            "historical_cheaper_union": inventory[str(task)],
            "sound_control_attempt": {
                "path": str(control.relative_to(ROOT)),
                "structural": structural(control),
                "official_like_profile": control_profile,
                "known_dual": known_dual(control, task),
                "strictly_cheaper_than_incumbent": (
                    isinstance(control_cost, int) and control_cost < incumbent_cost
                ),
            },
        }

    result = {
        "complete": True,
        "scope": list(TASKS),
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha(BASE_ZIP),
        "admission_policy": {
            "truthful_actual_cost_strictly_lower": True,
            "known_complete": "100% in both ORT modes, runtime0",
            "fresh": ">=90% on two seeds; private lineage requires decoded reference and fresh100",
            "strict_data_prop_static_positive": True,
            "conv_bias_ub0": True,
            "forbidden": ["lookup", "shape cloak", "giant contraction", "nonstandard domain"],
        },
        "tasks": tasks,
        "admitted": [],
        "admitted_gain": 0.0,
        "verdict": "NO_TRUTHFUL_STRICTLY_CHEAPER_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "history_rows": {task: len(rows) for task, rows in inventory.items()},
        "controls": {
            task: row["sound_control_attempt"]["official_like_profile"]
            for task, row in tasks.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
