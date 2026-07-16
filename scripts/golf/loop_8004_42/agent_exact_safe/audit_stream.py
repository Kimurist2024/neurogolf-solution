#!/usr/bin/env python3
"""Memory-bounded known/fresh differential audit for one ONNX candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402


TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def session(model: onnx.ModelProto, disable_all: bool) -> onnxruntime.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Several incumbent models deliberately carry conservative static shape
    # annotations; ORT warns on every fresh example.  Suppress those warnings
    # so a long streaming audit cannot overflow its process output channel.
    options.log_severity_level = 4
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return onnxruntime.InferenceSession(sanitized.SerializeToString(), options)


def structural_gate(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {
        "checker": "FAIL",
        "strict_shape_inference": "FAIL",
        "static_shapes": False,
        "banned_ops": [],
        "nested_graphs": [],
        "conv_bias_ub": [],
        "errors": [],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["checker"] = "PASS"
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        row["strict_shape_inference"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["pass"] = False
        return row

    bad_shapes = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(
        inferred.graph.output
    ):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            bad_shapes.append(value.name)
            continue
        if any(
            (not dim.HasField("dim_value"))
            or dim.HasField("dim_param")
            or dim.dim_value <= 0
            for dim in tensor_type.shape.dim
        ):
            bad_shapes.append(value.name)
    row["static_shapes"] = not bad_shapes
    if bad_shapes:
        row["errors"].append(f"dynamic_or_missing_shapes: {bad_shapes[:20]}")

    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            row["banned_ops"].append(node.op_type)
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                row["nested_graphs"].append(node.op_type)

    spec = importlib.util.spec_from_file_location(
        "check_conv_bias", ROOT / "scripts/golf/check_conv_bias.py"
    )
    if spec is None or spec.loader is None:
        row["errors"].append("cannot import check_conv_bias")
    else:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row["conv_bias_ub"] = [list(item) for item in module.check_model(model)]
    row["pass"] = bool(
        row["checker"] == "PASS"
        and row["strict_shape_inference"] == "PASS"
        and row["static_shapes"]
        and not row["banned_ops"]
        and not row["nested_graphs"]
        and not row["conv_bias_ub"]
    )
    return row


def counters() -> dict[str, int | float | None]:
    return {
        "total": 0,
        "baseline_right": 0,
        "candidate_right": 0,
        "baseline_runtime_errors": 0,
        "candidate_runtime_errors": 0,
        "one_sided_runtime_errors": 0,
        "raw_bitwise_equal": 0,
        "decoded_equal": 0,
        "max_abs_raw_difference": 0.0,
    }


def update(
    row: dict[str, int | float | None],
    baseline_session: onnxruntime.InferenceSession,
    candidate_session: onnxruntime.InferenceSession,
    example: dict[str, np.ndarray],
) -> None:
    row["total"] = int(row["total"]) + 1
    expected = example["output"] > 0
    baseline_raw = candidate_raw = None
    try:
        baseline_raw = baseline_session.run(["output"], {"input": example["input"]})[0]
    except Exception:  # noqa: BLE001
        row["baseline_runtime_errors"] = int(row["baseline_runtime_errors"]) + 1
    try:
        candidate_raw = candidate_session.run(["output"], {"input": example["input"]})[0]
    except Exception:  # noqa: BLE001
        row["candidate_runtime_errors"] = int(row["candidate_runtime_errors"]) + 1
    if (baseline_raw is None) != (candidate_raw is None):
        row["one_sided_runtime_errors"] = int(row["one_sided_runtime_errors"]) + 1
    if baseline_raw is None or candidate_raw is None:
        return
    baseline_decoded = baseline_raw > 0
    candidate_decoded = candidate_raw > 0
    row["baseline_right"] = int(row["baseline_right"]) + int(
        np.array_equal(baseline_decoded, expected)
    )
    row["candidate_right"] = int(row["candidate_right"]) + int(
        np.array_equal(candidate_decoded, expected)
    )
    row["raw_bitwise_equal"] = int(row["raw_bitwise_equal"]) + int(
        np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
    )
    row["decoded_equal"] = int(row["decoded_equal"]) + int(
        np.array_equal(baseline_decoded, candidate_decoded)
    )
    delta = np.abs(
        np.nan_to_num(baseline_raw).astype(np.float64, copy=False)
        - np.nan_to_num(candidate_raw).astype(np.float64, copy=False)
    )
    row["max_abs_raw_difference"] = max(
        float(row["max_abs_raw_difference"]), float(delta.max(initial=0.0))
    )


def finalize(row: dict[str, int | float | None]) -> None:
    executable = int(row["total"]) - int(row["candidate_runtime_errors"])
    row["candidate_accuracy"] = (
        int(row["candidate_right"]) / executable if executable else None
    )


def score(model: onnx.ModelProto, task: int, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix=f"exact_safe_{task}_{label}_") as directory:
        return scoring.score_and_verify(
            model, task, directory, label=label, require_correct=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fresh", type=int, default=3000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-known", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--session-refresh", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_bytes = archive.read(f"task{args.task:03d}.onnx")
    candidate_bytes = args.candidate.read_bytes()
    baseline = onnx.load_model_from_string(baseline_bytes)
    candidate = onnx.load_model_from_string(candidate_bytes)
    def make_sessions() -> dict[str, tuple[onnxruntime.InferenceSession, onnxruntime.InferenceSession]]:
        return {
            "disable_all": (session(baseline, True), session(candidate, True)),
            "default": (session(baseline, False), session(candidate, False)),
        }

    sessions = make_sessions()
    rows = {
        mode: {"known": counters(), "fresh": counters()} for mode in sessions
    }

    if not args.skip_known:
        known = scoring.load_examples(args.task)
        for subset in ("train", "test", "arc-gen"):
            for raw in known.get(subset, []):
                converted = scoring.convert_to_numpy(raw)
                if converted is None:
                    continue
                for mode, (base_session, candidate_session) in sessions.items():
                    update(rows[mode]["known"], base_session, candidate_session, converted)
        print(f"task{args.task:03d}: known complete", flush=True)

    generator = importlib.import_module(f"task_{TASK_MAP[f'{args.task:03d}']}")
    random.seed(args.seed if args.seed is not None else 8_004_420 + args.task)
    generated = 0
    generation_errors = 0
    attempts = 0
    while generated < args.fresh and attempts < args.fresh * 10:
        attempts += 1
        try:
            converted = scoring.convert_to_numpy(generator.generate())
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if converted is None:
            continue
        generated += 1
        for mode, (base_session, candidate_session) in sessions.items():
            update(rows[mode]["fresh"], base_session, candidate_session, converted)
        # task233's deliberately conservative value_info causes ORT to retain
        # alternate-shape arenas across random grid sizes.  Recreate sessions
        # periodically to keep the long audit bounded without changing the
        # tested model bytes or execution modes.
        if (
            args.session_refresh > 0
            and generated % args.session_refresh == 0
            and generated < args.fresh
        ):
            del sessions
            gc.collect()
            sessions = make_sessions()
        if generated % args.progress_every == 0:
            print(f"task{args.task:03d}: fresh {generated}/{args.fresh}", flush=True)

    for mode in rows.values():
        finalize(mode["known"])
        finalize(mode["fresh"])

    base_score = None if args.skip_score else score(baseline, args.task, "baseline")
    candidate_score = None if args.skip_score else score(candidate, args.task, "candidate")
    cost_reduction = None
    gain = None
    if base_score and candidate_score:
        cost_reduction = int(base_score["cost"]) - int(candidate_score["cost"])
        gain = math.log(int(base_score["cost"]) / int(candidate_score["cost"]))

    exact = bool(structural_gate(candidate).get("pass"))
    for mode in rows.values():
        for subset in ("known", "fresh"):
            row = mode[subset]
            exact &= bool(
                row["total"] > 0
                and row["candidate_runtime_errors"] == 0
                and row["one_sided_runtime_errors"] == 0
                and row["raw_bitwise_equal"] == row["total"]
                and row["decoded_equal"] == row["total"]
                and row["candidate_right"] == row["total"]
            )
    exact &= cost_reduction is not None and cost_reduction > 0

    report = {
        "task": args.task,
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_sha256": sha256(baseline_bytes),
        "candidate": str(args.candidate.resolve().relative_to(ROOT)),
        "candidate_sha256": sha256(candidate_bytes),
        "structural_gate": structural_gate(candidate),
        "fresh_requested": args.fresh,
        "fresh_generated": generated,
        "generation_errors": generation_errors,
        "generation_attempts": attempts,
        "known_skipped": args.skip_known,
        "score_skipped": args.skip_score,
        "modes": rows,
        "baseline_score": base_score,
        "candidate_score": candidate_score,
        "cost_reduction": cost_reduction,
        "projected_gain": gain,
        "verdict": "ACCEPT_EXACT" if exact else "REJECT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": args.task,
        "verdict": report["verdict"],
        "cost_reduction": cost_reduction,
        "projected_gain": gain,
        "fresh": {mode: value["fresh"] for mode, value in rows.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
