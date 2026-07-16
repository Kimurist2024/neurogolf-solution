#!/usr/bin/env python3
"""Fail-closed structural/cost/runtime screen for lane 138 candidates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
BASE_COSTS = {157: 849, 368: 521, 370: 944}
MEMBER_SHA256 = {
    157: "a1254f2619406b8db5d3fe5fdd1c42c917820fa51b91faef0f3ceed5d8b3662f",
    368: "0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811",
    370: "513c0b40056f0ef9ee30cffe32a940571a0e977bf467d1c90096425c68e682d9",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATIC = load_module(
    "lane138_static",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "lane138_trace",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
RANK = load_module("lane138_rank", ROOT / "scripts/golf/rank_dir.py")
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def cost(path: Path) -> dict[str, int]:
    memory, params, total = RANK.cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(total)}


def official_profile(task: int, path: Path, label: str) -> dict[str, Any]:
    model = onnx.load(path)
    with tempfile.TemporaryDirectory(prefix=f"lane138_{task}_", dir="/tmp") as workdir:
        try:
            result = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label, require_correct=False
            )
            return {"result": result}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def raw_known_equivalence(task: int, base: bytes, candidate: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    rows = []
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            item: dict[str, Any] = {
                "mode": mode,
                "threads": threads,
                "total": 0,
                "bitwise_equal": 0,
                "threshold_equal": 0,
                "errors": 0,
            }
            try:
                base_session = make_session(base, disabled, threads)
                candidate_session = make_session(candidate, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                item["session_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(item)
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    item["total"] += 1
                    try:
                        a = np.asarray(
                            base_session.run(
                                [base_session.get_outputs()[0].name],
                                {base_session.get_inputs()[0].name: benchmark["input"]},
                            )[0]
                        )
                        b = np.asarray(
                            candidate_session.run(
                                [candidate_session.get_outputs()[0].name],
                                {candidate_session.get_inputs()[0].name: benchmark["input"]},
                            )[0]
                        )
                        item["bitwise_equal"] += int(
                            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                        )
                        item["threshold_equal"] += int(np.array_equal(a > 0, b > 0))
                    except Exception:  # noqa: BLE001
                        item["errors"] += 1
            rows.append(item)
    return {
        "configs": rows,
        "pass": all(
            row["total"] > 0
            and row["bitwise_equal"] == row["total"]
            and row["threshold_equal"] == row["total"]
            and row["errors"] == 0
            and "session_error" not in row
            for row in rows
        ),
    }


def baseline_audit() -> dict[str, Any]:
    result = {}
    for task in BASE_COSTS:
        path = HERE / f"baseline/task{task:03d}.onnx"
        data = path.read_bytes()
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} member drift")
        measured = cost(path)
        if measured["cost"] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {measured}")
        structural = STATIC.structural_audit(data)
        try:
            trace = TRACE.runtime_shape_trace(task, onnx.load(path))
        except Exception as exc:  # noqa: BLE001
            trace = {"error": f"{type(exc).__name__}: {exc}"}
        known = STATIC.known_four(task, data)
        result[str(task)] = {
            "task": task,
            "path": rel(path),
            "sha256": sha256(data),
            "cost": measured,
            "structural": structural,
            "runtime_shape_trace": trace,
            "known_four": known,
            "known_four_complete": STATIC.known_complete(known),
        }
        print(
            f"BASE task{task:03d} cost={measured['cost']} "
            f"mismatch={len(trace.get('declared_actual_mismatches', []))} "
            f"known4={result[str(task)]['known_four_complete']}",
            flush=True,
        )
    return result


def main() -> None:
    authority_before = sha256(AUTHORITY.read_bytes())
    if authority_before != AUTHORITY_SHA256:
        raise RuntimeError(f"authority drift: {authority_before}")
    build = json.loads((HERE / "audit/build_manifest.json").read_text())
    baselines = baseline_audit()
    rows = []
    for original in build["rows"]:
        row = dict(original)
        task = int(row["task"])
        path = ROOT / row["path"]
        data = path.read_bytes()
        structural = STATIC.structural_audit(data)
        row["structural"] = structural
        if not structural.get("pass"):
            row["stage"] = "REJECT_FULL_STRICT_SCHEMA_UB"
            rows.append(row)
            continue
        try:
            measured = cost(path)
            row["actual_official_like_cost"] = measured
        except Exception as exc:  # noqa: BLE001
            row["cost_error"] = f"{type(exc).__name__}: {exc}"
            row["stage"] = "REJECT_UNSCORABLE_OR_ORT_KERNEL"
            rows.append(row)
            continue
        if measured["cost"] < 0:
            row["stage"] = "REJECT_UNSCORABLE_OR_ORT_KERNEL"
            rows.append(row)
            continue
        if measured["cost"] >= BASE_COSTS[task]:
            row["stage"] = "REJECT_NOT_STRICTLY_LOWER"
            rows.append(row)
            continue
        row["official_profile"] = official_profile(task, path, f"lane138_{task}_{row['kind']}")
        known = STATIC.known_four(task, data)
        row["known_four"] = known
        row["known_four_complete"] = STATIC.known_complete(known)
        row["raw_known_equivalence"] = raw_known_equivalence(
            task,
            (HERE / f"baseline/task{task:03d}.onnx").read_bytes(),
            data,
        )
        try:
            trace = TRACE.runtime_shape_trace(task, onnx.load(path))
        except Exception as exc:  # noqa: BLE001
            trace = {"error": f"{type(exc).__name__}: {exc}"}
        row["runtime_shape_trace"] = trace
        truthful = (
            not trace.get("error")
            and not trace.get("declared_actual_mismatches")
        )
        row["truthful_runtime_shapes"] = truthful
        if not row["known_four_complete"]:
            row["stage"] = "REJECT_KNOWN4_OR_RUNTIME"
        elif not row["raw_known_equivalence"]["pass"]:
            row["stage"] = "REJECT_AUTHORITY_RAW_MISMATCH"
        elif not truthful:
            row["stage"] = "REJECT_SHAPE_CLOAK"
        else:
            # No candidate reached this point. Fresh is intentionally fail-closed and
            # is run only after all cheaper gates pass.
            row["stage"] = "FRESH_REQUIRED"
        rows.append(row)

    stage_counts = dict(Counter(row["stage"] for row in rows))
    fresh_required = [row for row in rows if row["stage"] == "FRESH_REQUIRED"]
    authority_after = sha256(AUTHORITY.read_bytes())
    if authority_after != authority_before:
        raise RuntimeError(f"authority changed: {authority_after}")
    result = {
        "lane": "agent_high368_157_370_138",
        "authority": {
            "score": 8009.46,
            "archive": "submission.zip",
            "sha256_before": authority_before,
            "sha256_after": authority_after,
            "member_sha256": MEMBER_SHA256,
            "costs": BASE_COSTS,
        },
        "baselines": baselines,
        "candidate_count": len(rows),
        "stage_counts": stage_counts,
        "fresh_required_count": len(fresh_required),
        "fresh_policy": "2 independent seeds x >=1500, >=90%, dual ORT, authority raw bitwise equality; only after all earlier gates",
        "rows": rows,
    }
    (HERE / "audit/screen_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "stage_counts": stage_counts, "fresh_required": len(fresh_required)}, indent=2))


if __name__ == "__main__":
    main()
