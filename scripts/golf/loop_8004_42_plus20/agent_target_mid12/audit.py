#!/usr/bin/env python3
"""Read-only audit for the four target tasks.

All persistent output is written beside this script.  Baselines are read
directly from submission_base_8004.50.zip and are never extracted in place.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ort.set_default_logger_severity(3)


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

TASKS = {
    330: {
        "hash": "d2abd087",
        "control": ROOT / "scripts/golf/loop_7999_13/lane_a26/rule_references/task330_truthful_component_rect.onnx",
        "control_cost": 5525,
    },
    280: {
        "hash": "b527c5c6",
        "control": ROOT / "scripts/golf/loop_7999_13/lane_b17/candidate_task280_truthful.onnx",
        "control_cost": 2161,
    },
    364: {
        "hash": "e509e548",
        "control": ROOT / "scripts/golf/scratch/task364/cand8.onnx",
        "control_cost": 46741,
    },
    310: {
        "hash": "c909285e",
        "control": ROOT / "scripts/golf/loop_7999_13/lane_c9/task310_safe_linear_selector.onnx",
        "control_cost": 633,
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value) -> list[int]:
    return [int(d.dim_value) for d in value.type.tensor_type.shape.dim]


def model_structure(model: onnx.ModelProto) -> dict:
    checker = strict = False
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        strict = True
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"
    hist = Counter(n.op_type for n in model.graph.node)
    giant = [
        {"name": n.name, "inputs": len([x for x in n.input if x])}
        for n in model.graph.node
        if n.op_type == "Einsum" and len([x for x in n.input if x]) >= 16
    ]
    banned = sorted(
        {
            n.op_type
            for n in model.graph.node
            if n.op_type.upper() in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
            or "SEQUENCE" in n.op_type.upper()
        }
    )
    bias = []
    init = {x.name: x for x in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) >= 3 and n.input[2]:
            w, b = init.get(n.input[1]), init.get(n.input[2])
        elif n.op_type == "QLinearConv" and len(n.input) >= 9 and n.input[8]:
            w, b = init.get(n.input[3]), init.get(n.input[8])
        else:
            continue
        out_ch = int(w.dims[0]) if w is not None and w.dims else None
        b_len = math.prod(b.dims) if b is not None else None
        bias.append({"node": n.name, "op": n.op_type, "out_channels": out_ch, "bias_len": b_len, "ub": b_len != out_ch})
    return {
        "nodes": len(model.graph.node),
        "params": sum(math.prod(x.dims) for x in model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "op_histogram": dict(sorted(hist.items())),
        "input_shapes": {x.name: dims(x) for x in model.graph.input},
        "output_shapes": {x.name: dims(x) for x in model.graph.output},
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_data_prop": strict,
        "strict_error": strict_error,
        "banned_ops": banned,
        "giant_einsums": giant,
        "has_tfidf": bool(hist.get("TfIdfVectorizer")),
        "has_hardmax": bool(hist.get("Hardmax")),
        "has_center_crop_pad": bool(hist.get("CenterCropPad")),
        "has_group_norm": bool(hist.get("GroupNormalization")),
        "conv_bias": bias,
        "conv_bias_ub_count": sum(int(x["ub"]) for x in bias),
    }


def session(model: onnx.ModelProto, mode: str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    return ort.InferenceSession(sanitized.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def known(model: onnx.ModelProto, task: int, mode: str) -> dict:
    examples = scoring.load_examples(task)
    rows = examples["train"] + examples["test"] + examples["arc-gen"]
    result = {"mode": mode, "total": len(rows), "right": 0, "wrong": 0, "runtime_errors": 0, "session_error": None}
    try:
        sess = session(model, mode)
    except Exception as exc:  # noqa: BLE001
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        result["runtime_errors"] = len(rows)
        return result
    min_pos = None
    mid = 0
    actual_shapes: set[tuple[int, ...]] = set()
    for ex in rows:
        benchmark = scoring.convert_to_numpy(ex)
        try:
            raw = sess.run(["output"], {"input": benchmark["input"]})[0]
            actual_shapes.add(tuple(int(x) for x in raw.shape))
            pos = raw[raw > 0]
            if pos.size:
                cur = float(np.min(pos))
                min_pos = cur if min_pos is None else min(min_pos, cur)
                mid += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
            if np.array_equal(raw > 0, benchmark["output"] > 0):
                result["right"] += 1
            else:
                result["wrong"] += 1
        except Exception:  # noqa: BLE001
            result["runtime_errors"] += 1
    result["margin_min_positive"] = min_pos
    result["margin_mid_count"] = mid
    result["runtime_output_shapes"] = [list(x) for x in sorted(actual_shapes)]
    return result


def raw_rule_known(task: int, task_hash: str) -> dict:
    spec = importlib.util.spec_from_file_location(f"sakana_task_{task}", ROOT / f"inputs/sakana-gcg-2025/raw/task{task}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    data = scoring.load_examples(task)
    rows = data["train"] + data["test"] + data["arc-gen"]
    right = 0
    first_wrong = None
    for idx, ex in enumerate(rows):
        try:
            got = mod.p(copy.deepcopy(ex["input"]))
            got = [list(x) for x in got]
            ok = got == ex["output"]
        except Exception as exc:  # noqa: BLE001
            ok = False
            if first_wrong is None:
                first_wrong = {"index": idx, "error": f"{type(exc).__name__}: {exc}"}
        if ok:
            right += 1
        elif first_wrong is None:
            first_wrong = {"index": idx}
    return {"source": f"inputs/sakana-gcg-2025/raw/task{task}.py", "total": len(rows), "right": right, "wrong": len(rows) - right, "first_wrong": first_wrong}


def main() -> None:
    costs = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json").read_text())
    cost_map = {int(x["task"]): x for x in costs["ranked"]}
    zpath = ROOT / "submission_base_8004.50.zip"
    result = {"baseline_zip": str(zpath.relative_to(ROOT)), "baseline_zip_sha256": sha(zpath.read_bytes()), "tasks": {}}
    with zipfile.ZipFile(zpath) as zf:
        for task, meta in TASKS.items():
            data = zf.read(f"task{task}.onnx")
            model = onnx.load_model_from_string(data)
            control_data = meta["control"].read_bytes()
            control = onnx.load_model_from_string(control_data)
            result["tasks"][str(task)] = {
                "hash": meta["hash"],
                "true_rule_known": raw_rule_known(task, meta["hash"]),
                "baseline": {
                    "source": f"submission_base_8004.50.zip::task{task}.onnx",
                    "sha256": sha(data),
                    "serialized_bytes": len(data),
                    "measured_memory": cost_map[task]["memory"],
                    "measured_params": cost_map[task]["params"],
                    "measured_cost": cost_map[task]["cost"],
                    "structure": model_structure(model),
                    "known": [known(model, task, "disable_all"), known(model, task, "default")],
                },
                "truthful_control": {
                    "source": str(meta["control"].relative_to(ROOT)),
                    "sha256": sha(control_data),
                    "serialized_bytes": len(control_data),
                    "measured_cost": meta["control_cost"],
                    "structure": model_structure(control),
                    "known": [known(control, task, "disable_all"), known(control, task, "default")],
                    "strictly_cheaper_than_baseline": meta["control_cost"] < cost_map[task]["cost"],
                },
            }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
