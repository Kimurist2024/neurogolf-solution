#!/usr/bin/env python3
"""Four-configuration known audit and direct runtime-shape trace.

Only models already proven actually cheaper by rescreen.json are executed here.
The lane remains fail-closed and non-promoting.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    102, 25, 324, 308, 338, 134, 268, 184, 377, 170,
    239, 222, 48, 234, 264, 200, 387, 132, 388, 228,
)
BASE_ZIP = ROOT / "submission_base_8005.17.zip"
COSTS_PATH = HERE / "baseline_costs_8005_17.json"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "mid20b_audit_scanner",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
SWEEP = load_module(
    "mid20b_audit_static",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_config(task: int, data: bytes, disable: bool, threads: int) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    total = sum(len(examples[split]) for split in ("train", "test", "arc-gen"))
    row: dict[str, Any] = {
        "total": total,
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "min_positive": None,
        "max_abs_raw": 0.0,
        "output_shapes": [],
        "first_failure": None,
    }
    try:
        session = make_session(data, disable, threads)
    except Exception as exc:  # noqa: BLE001
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_errors"] = total
        row["perfect"] = False
        return row
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
                shape = list(raw.shape)
                if shape not in row["output_shapes"]:
                    row["output_shapes"].append(shape)
                finite = np.isfinite(raw)
                row["nonfinite_values"] += int(raw.size - np.count_nonzero(finite))
                safe = raw[finite]
                if safe.size:
                    positive = safe[safe > 0]
                    row["near_positive_values"] += int(np.count_nonzero((safe > 0) & (safe < 0.25)))
                    if positive.size:
                        value = float(positive.min())
                        row["min_positive"] = value if row["min_positive"] is None else min(row["min_positive"], value)
                    row["max_abs_raw"] = max(row["max_abs_raw"], float(np.abs(safe).max(initial=0.0)))
                expected = benchmark["output"].astype(bool)
                if np.array_equal(raw > 0, expected):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "split": split,
                        "index": index,
                        "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                    }
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "split": split,
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    row["perfect"] = (
        row["right"] == total
        and row["wrong"] == 0
        and row["runtime_errors"] == 0
        and row["nonfinite_values"] == 0
    )
    return row


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def direct_runtime_shape_trace(task: int, data: bytes) -> dict[str, Any]:
    """Trace the raw model without sanitize_model dropping added outputs."""
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    existing = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name in names or name not in typed:
                continue
            names.append(name)
            if name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    benchmark = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    if benchmark is None:
        raise RuntimeError("first train example is not convertible")
    arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = 0
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        nonfinite += int(value.size - np.count_nonzero(np.isfinite(value))) if value.dtype.kind in "fc" else 0
        declared = dims(typed[name])
        actual = list(value.shape)
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def main() -> int:
    report = json.loads((HERE / "rescreen.json").read_text())
    wanted = {
        row["sha256"]: row
        for row in report["rows"]
        if row.get("actual_screen_cost") is not None
        and int(row["actual_screen_cost"]) < int(row["current_actual_cost"])
    }
    SCANNER.HERE = HERE
    SCANNER.TARGETS = TARGETS
    SCANNER.BASE_ZIP = BASE_ZIP
    SCANNER.CURRENT_COSTS_JSON = COSTS_PATH
    inventory, inventory_report = SCANNER.inventory()
    data_by_sha = {
        digest: item["data"]
        for per_task in inventory.values()
        for digest, item in per_task.items()
        if digest in wanted
    }
    rows = []
    for index, (digest, source_row) in enumerate(sorted(wanted.items(), key=lambda item: (item[1]["task"], item[0])), 1):
        task = int(source_row["task"])
        data = data_by_sha[digest]
        static = SWEEP.static_audit(data)
        configs = {
            label: known_config(task, data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        known_perfect = all(item.get("perfect", False) for item in configs.values())
        try:
            trace = direct_runtime_shape_trace(task, data) if known_perfect else None
            trace_error = None
        except Exception as exc:  # noqa: BLE001
            trace = None
            trace_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "task": task,
                "sha256": digest,
                "sources": source_row["sources"],
                "baseline_cost": source_row["current_actual_cost"],
                "actual_cost": source_row["actual_screen_cost"],
                "gain": float(source_row.get("gain", 0.0)),
                "source_stage": source_row["stage"],
                "static": static,
                "known_four_configs": configs,
                "known_perfect_all_configs": known_perfect,
                "runtime_shape_trace": trace,
                "runtime_shape_trace_error": trace_error,
            }
        )
        print(
            f"AUDIT {index}/{len(wanted)} task{task:03d} cost={source_row['actual_screen_cost']} "
            f"known4={known_perfect} shape={None if trace is None else trace['truthful']}",
            flush=True,
        )
    output = {
        "baseline_zip": report["baseline_zip"],
        "baseline_zip_sha256": report["baseline_zip_sha256"],
        "inventory_counts": inventory_report["counts"],
        "candidate_count": len(rows),
        "known_perfect_four_configs_count": sum(row["known_perfect_all_configs"] for row in rows),
        "truthful_count": sum(
            bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in rows
        ),
        "rows": rows,
    }
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "actual_lower_four_config.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
