#!/usr/bin/env python3
"""Deep audit strict-lower historical leads against immutable authority."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATES = HERE / "history_candidates"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = load_module(
    "lane157_deep_auditor",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disable_all: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def mode_audit(
    task: int,
    authority: onnx.ModelProto,
    candidate: onnx.ModelProto,
    disable_all: bool,
    threads: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "optimization": "ORT_DISABLE_ALL" if disable_all else "ORT_DEFAULT",
        "intra_op_threads": threads,
        "inter_op_threads": 1,
    }
    try:
        reference = session(authority, disable_all, threads)
        trial = session(candidate, disable_all, threads)
    except Exception as exc:  # noqa: BLE001
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        return result
    reference_input = reference.get_inputs()[0].name
    reference_output = reference.get_outputs()[0].name
    trial_input = trial.get_inputs()[0].name
    trial_output = trial.get_outputs()[0].name
    rows: dict[str, dict[str, int]] = {}
    for subset in ("train", "test", "arc-gen"):
        counters = {
            "examples": 0,
            "candidate_correct": 0,
            "candidate_wrong": 0,
            "candidate_errors": 0,
            "authority_errors": 0,
            "raw_equal_authority": 0,
            "raw_different_authority": 0,
        }
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            counters["examples"] += 1
            try:
                reference_value = np.asarray(
                    reference.run([reference_output], {reference_input: benchmark["input"]})[0]
                )
            except Exception:  # noqa: BLE001
                counters["authority_errors"] += 1
                continue
            try:
                trial_value = np.asarray(
                    trial.run([trial_output], {trial_input: benchmark["input"]})[0]
                )
            except Exception:  # noqa: BLE001
                counters["candidate_errors"] += 1
                continue
            if np.array_equal(trial_value > 0, benchmark["output"] > 0):
                counters["candidate_correct"] += 1
            else:
                counters["candidate_wrong"] += 1
            if raw_equal(reference_value, trial_value):
                counters["raw_equal_authority"] += 1
            else:
                counters["raw_different_authority"] += 1
        rows[subset] = counters
    totals = {
        key: sum(row[key] for row in rows.values())
        for key in next(iter(rows.values()))
    }
    rows["total"] = totals
    result["known"] = rows
    return result


def main() -> None:
    ort.set_default_logger_severity(4)
    history = json.loads((HERE / "history_screen.json").read_text())
    output: dict[str, object] = {
        "authority": history["authority"],
        "authority_sha256": history["authority_sha256"],
        "policy": {
            "acceptance": "strict-lower actual competition profile, full checker, strict data_prop, standard/no banned/nested/functions/sparse, Conv UB0, truthful runtime shapes, default+disable threads1/4 known-complete, and all-input semantic proof or exact raw authority pass-through",
            "history_behavior_change": "A noncanonical historical graph has no all-input proof; known/fresh accuracy cannot establish a universal rule and is rejected even if known-complete.",
        },
        "candidates": [],
    }
    for path in sorted(CANDIDATES.glob("task*.onnx")):
        match = re.search(r"task(\d{3})", path.name)
        if not match:
            continue
        task = int(match.group(1))
        candidate = onnx.load(path)
        authority_path = HERE / "baseline" / f"task{task:03d}.onnx"
        authority = onnx.load(authority_path)
        record = AUDITOR.audit(path.stem, task, path)
        source_row = next(
            row
            for row in history["tasks"][str(task)]["rows"]
            if row["sha256"] == record["sha256"]
        )
        modes = [
            mode_audit(task, authority, candidate, disable_all, threads)
            for disable_all in (True, False)
            for threads in (1, 4)
        ]
        trace = record.get("runtime_shape_trace", {})
        mismatches = trace.get("declared_actual_mismatches", []) if isinstance(trace, dict) else []
        mode_complete = all(
            "session_error" not in mode
            and mode.get("known", {}).get("total", {}).get("candidate_wrong") == 0
            and mode.get("known", {}).get("total", {}).get("candidate_errors") == 0
            and mode.get("known", {}).get("total", {}).get("authority_errors") == 0
            for mode in modes
        )
        raw_all = all(
            mode.get("known", {}).get("total", {}).get("raw_different_authority") == 0
            and mode.get("known", {}).get("total", {}).get("raw_equal_authority")
            == mode.get("known", {}).get("total", {}).get("examples")
            for mode in modes
            if "session_error" not in mode
        )
        reasons: list[str] = []
        if not source_row.get("same_canonical_graph_as_authority"):
            reasons.append("NO_ALL_INPUT_EQUIVALENCE_PROOF_NONCANONICAL_HISTORY")
        if mismatches:
            reasons.append(f"RUNTIME_SHAPE_NOT_TRUTHFUL_{len(mismatches)}_MISMATCHES")
        if not mode_complete:
            reasons.append("DEFAULT_DISABLE_THREADS1_4_NOT_COMPLETE")
        if not raw_all:
            reasons.append("RAW_OUTPUT_DIFFERS_FROM_AUTHORITY")
        deep = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": record["sha256"],
            "sources": source_row["sources"],
            "same_canonical_graph_as_authority": source_row["same_canonical_graph_as_authority"],
            "actual_profile": record.get("official_like_score"),
            "full_check": record.get("full_check"),
            "strict_data_prop": record.get("strict_shape_data_prop"),
            "nonstandard_domains": record.get("nonstandard_domains"),
            "banned_ops": record.get("banned_ops"),
            "nested_graph_attributes": record.get("nested_graph_attributes"),
            "function_count": record.get("function_count"),
            "sparse_initializer_count": record.get("sparse_initializer_count"),
            "conv_bias_findings": record.get("conv_bias_findings"),
            "runtime_shape_trace": trace,
            "modes": modes,
            "raw_equal_all_available_modes": raw_all,
            "mode_complete": mode_complete,
            "decision": "REJECT" if reasons else "ADMIT",
            "reasons": reasons,
        }
        output["candidates"].append(deep)
        (HERE / "deep_audit.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "task": task,
                    "cost": deep["actual_profile"].get("cost") if deep["actual_profile"] else None,
                    "correct": deep["actual_profile"].get("correct") if deep["actual_profile"] else None,
                    "shape_mismatches": len(mismatches),
                    "mode_complete": mode_complete,
                    "raw_all": raw_all,
                    "decision": deep["decision"],
                }
            ),
            flush=True,
        )
    output["admitted_count"] = sum(row["decision"] == "ADMIT" for row in output["candidates"])
    (HERE / "deep_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
