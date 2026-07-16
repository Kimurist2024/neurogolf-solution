#!/usr/bin/env python3
"""Fail-closed runtime audit for strict-lower candidates emitted by this lane."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (178, 228, 234, 264, 325, 344, 357, 387, 388, 392, 397, 398)
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module("exactB116_scan", HERE / "scan_candidates.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


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


def run(session: ort.InferenceSession, benchmark: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
    )


def known_config(task: int, baseline: bytes, candidate: bytes, disable: bool, threads: int) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    ordered = [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[split])
    ]
    row: dict[str, Any] = {
        "total": len(ordered),
        "candidate_right": 0,
        "baseline_right": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "runtime_errors": {"candidate": 0, "baseline": 0},
        "nonfinite_values": {"candidate": 0, "baseline": 0},
        "candidate_shapes": [],
        "baseline_shapes": [],
        "first_failure": None,
    }
    try:
        sessions = {
            "candidate": make_session(candidate, disable, threads),
            "baseline": make_session(baseline, disable, threads),
        }
    except Exception as exc:  # noqa: BLE001
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["perfect"] = False
        return row
    for split, index, example in ordered:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            row["first_failure"] = row["first_failure"] or {
                "split": split,
                "index": index,
                "error": "convert_to_numpy returned None",
            }
            continue
        outputs: dict[str, np.ndarray] = {}
        for name, session in sessions.items():
            try:
                value = run(session, benchmark)
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"][name] += 1
                row["first_failure"] = row["first_failure"] or {
                    "split": split,
                    "index": index,
                    "model": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            outputs[name] = value
            shape = list(value.shape)
            key = f"{name}_shapes"
            if shape not in row[key]:
                row[key].append(shape)
            row["nonfinite_values"][name] += int(value.size - np.count_nonzero(np.isfinite(value)))
            row[f"{name}_right"] += int(np.array_equal(value > 0, benchmark["output"].astype(bool)))
        if len(outputs) == 2:
            raw_equal = np.array_equal(outputs["candidate"], outputs["baseline"])
            threshold_equal = np.array_equal(outputs["candidate"] > 0, outputs["baseline"] > 0)
            row["raw_equal"] += int(raw_equal)
            row["threshold_equal"] += int(threshold_equal)
            if not (raw_equal and threshold_equal):
                row["first_failure"] = row["first_failure"] or {
                    "split": split,
                    "index": index,
                    "comparison": "candidate_vs_immutable",
                    "raw_equal": bool(raw_equal),
                    "threshold_equal": bool(threshold_equal),
                }
    total = row["total"]
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())
    row["perfect"] = (
        row["candidate_right"] == total
        and row["baseline_right"] == total
        and row["raw_equal"] == total
        and row["threshold_equal"] == total
        and row["runtime_errors_total"] == 0
        and row["nonfinite_values_total"] == 0
    )
    return row


def direct_trace(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {value.name for value in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    benchmark = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    if benchmark is None:
        raise RuntimeError("first train example not convertible")
    try:
        session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
        arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    mismatches = []
    nonfinite = 0
    actual_shapes: dict[str, list[int]] = {}
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        actual = list(value.shape)
        actual_shapes[name] = actual
        declared = dims(typed[name])
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "actual_shapes": actual_shapes,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def alias_proof(baseline: bytes, candidate: bytes) -> dict[str, Any]:
    base = onnx.load_model_from_string(baseline)
    cand = onnx.load_model_from_string(candidate)
    base_init = {item.name: np.asarray(numpy_helper.to_array(item)) for item in base.graph.initializer}
    cand_init = {item.name: np.asarray(numpy_helper.to_array(item)) for item in cand.graph.initializer}
    equal = "s1" in base_init and "axes1" in base_init and np.array_equal(base_init["s1"], base_init["axes1"])
    axes_removed = "axes1" not in cand_init
    candidate_uses_s1 = sum(list(node.input).count("s1") for node in cand.graph.node)
    candidate_uses_axes1 = sum(list(node.input).count("axes1") for node in cand.graph.node)
    return {
        "baseline_s1": base_init.get("s1", np.asarray([])).tolist(),
        "baseline_axes1": base_init.get("axes1", np.asarray([])).tolist(),
        "value_dtype_shape_equal": bool(equal),
        "axes1_removed": axes_removed,
        "candidate_s1_use_count": candidate_uses_s1,
        "candidate_axes1_use_count": candidate_uses_axes1,
        "pass": bool(equal and axes_removed and candidate_uses_s1 >= 2 and candidate_uses_axes1 == 0),
    }


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority hash changed")
    scan = json.loads((HERE / "candidate_scan.json").read_text(encoding="utf-8"))
    inventory = json.loads((HERE / "inventory.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
        "winners": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        baseline = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}

    for task in TASKS:
        inv = inventory["tasks"][str(task)]
        current_data = baseline[task]
        current_sha = digest(current_data)
        current_cost = SCAN.official_cost(current_data, f"audit_task{task:03d}_base")
        task_row: dict[str, Any] = {
            "current_sha256": current_sha,
            "inventory_sha256_match": current_sha == inv["sha256"],
            "current_cost": current_cost,
            "inventory_cost_match": current_cost == inv["official_cost"],
            "graph_inventory": {
                "nodes": len(inv["graph"]["nodes"]),
                "dead_nodes": inv["graph"]["dead_node_indices"],
                "unused_initializers": inv["graph"]["unused_initializers"],
                "initializer_alias_groups": inv["graph"]["initializer_alias_groups"],
                "duplicate_node_groups": inv["graph"]["duplicate_node_groups"],
                "unused_optional_outputs": inv["graph"]["unused_optional_outputs"],
            },
            "unique_variants_scanned": scan["tasks"][str(task)]["variants_unique"],
            "strict_lower_candidates": [],
            "verdict": "NO_STRICT_LOWER_CANDIDATE",
        }
        for candidate_row in scan["tasks"][str(task)]["strict_lower"]:
            path = ROOT / candidate_row["path"]
            candidate_data = path.read_bytes()
            candidate_cost = SCAN.official_cost(candidate_data, f"audit_task{task:03d}_{candidate_row['label']}")
            static = SCAN.structural(onnx.load_model_from_string(candidate_data))
            trace = direct_trace(task, candidate_data)
            row: dict[str, Any] = {
                "label": candidate_row["label"],
                "path": candidate_row["path"],
                "sha256": digest(candidate_data),
                "scan_sha256_match": digest(candidate_data) == candidate_row["sha256"],
                "official_cost": candidate_cost,
                "strict_lower": candidate_cost["cost"] < current_cost["cost"],
                "static": static,
                "runtime_shape_trace": trace,
                "alias_proof": alias_proof(current_data, candidate_data) if task == 264 else None,
                "known_four_configs": {},
                "fresh": {"status": "not_run_before_shape_truth_gate"},
                "reasons": [],
                "accepted": False,
            }
            if not row["scan_sha256_match"]:
                row["reasons"].append("candidate_sha_changed")
            if not row["strict_lower"]:
                row["reasons"].append("not_strict_lower")
            if not static.get("pass", False):
                row["reasons"].append("static_gate")
            if task == 264 and not row["alias_proof"]["pass"]:
                row["reasons"].append("alias_proof_failed")
            # Known is still measured to document that the rewrite preserves
            # the LB-white payload; shape truth remains a mandatory admission gate.
            for disable, threads, label in CONFIGS:
                row["known_four_configs"][label] = known_config(
                    task, current_data, candidate_data, disable, threads
                )
            if not all(item.get("perfect", False) for item in row["known_four_configs"].values()):
                row["reasons"].append("known_or_raw_equivalence_failed")
            if not trace.get("truthful", False):
                row["reasons"].append("runtime_shapes_not_truthful")
            # No candidate in this lane reaches the shape gate. If one ever
            # does, fail closed until a dedicated 2x2000 dual-ORT fresh audit is added.
            if not row["reasons"]:
                row["reasons"].append("fresh_two_seed_dual_not_run")
                row["fresh"] = {"status": "required_not_run_fail_closed"}
            task_row["strict_lower_candidates"].append(row)
        if task_row["strict_lower_candidates"]:
            task_row["verdict"] = "REJECT_ALL_STRICT_LOWER"
        report["tasks"][str(task)] = task_row
        print(
            f"task{task:03d} cost={current_cost['cost']} lower={len(task_row['strict_lower_candidates'])} "
            f"verdict={task_row['verdict']}",
            flush=True,
        )
    report["summary"] = {
        "tasks": len(TASKS),
        "unique_variants_scanned": sum(row["unique_variants_scanned"] for row in report["tasks"].values()),
        "strict_lower_candidates": sum(len(row["strict_lower_candidates"]) for row in report["tasks"].values()),
        "winners": len(report["winners"]),
        "verdict": "NO_SAFE_EXACT_CANDIDATE",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "authority": report["authority"],
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": list(TASKS),
        "winners": [],
        "winner_count": 0,
        "promotion_performed": False,
        "verdict": "NO_SAFE_EXACT_CANDIDATE",
        "audit": str((HERE / "audit.json").relative_to(ROOT)),
        "inventory": str((HERE / "inventory.json").relative_to(ROOT)),
        "candidate_scan": str((HERE / "candidate_scan.json").relative_to(ROOT)),
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
