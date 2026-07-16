#!/usr/bin/env python3
"""Non-promoting POLICY90 residual audit for tasks 109/184/368/374.

This script reads the immutable 8009.46 authority, regenerates a small set of
exact local algebraic probes inside this lane, and fail-closes before fresh
testing unless cost, schema, truthful runtime shapes, and complete known data
all pass.  It deliberately does not call try_candidate.py.
"""

from __future__ import annotations

import copy
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
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_BASE = ROOT / "submission_base_8009.46.zip"
SCORES = ROOT / "all_scores.csv"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
SCORES_SHA = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
TASKS = (109, 184, 368, 374)
BASE_COSTS = {109: 405, 184: 421, 368: 521, 374: 481}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATIC = load_module(
    "residual_static",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "residual_trace",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_model(path: Path, model: onnx.ModelProto) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.SerializeToString()
    path.write_bytes(data)
    return data


def profile(task: int, data: bytes, label: str) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"residual_{task}_", dir="/tmp") as work:
        try:
            result = scoring.score_and_verify(
                copy.deepcopy(model), task, work, label=label, require_correct=False
            )
            return result or {"error": "score_and_verify returned None"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}


def runtime_trace(task: int, data: bytes) -> dict[str, Any]:
    try:
        return TRACE.runtime_shape_trace(task, onnx.load_model_from_string(data))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def converted_known(task: int) -> list[tuple[str, int, dict[str, np.ndarray]]]:
    rows: list[tuple[str, int, dict[str, np.ndarray]]] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(scoring.load_examples(task)[split]):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append((split, index, converted))
    return rows


def evaluate_cases(
    data: bytes,
    cases: list[tuple[str, int, dict[str, np.ndarray]]],
    disable_all: bool,
    threads: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(cases),
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite_cases": 0,
        "output_shape_mismatches": 0,
        "near_positive_values": 0,
        "min_positive": None,
        "first_failure": None,
    }
    try:
        session = make_session(data, disable_all, threads)
    except Exception as exc:  # noqa: BLE001
        result["errors"] = len(cases)
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        result["first_failure"] = {"kind": "session", "error": result["session_error"]}
        return result
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for split, index, benchmark in cases:
        try:
            raw = np.asarray(
                session.run([output_name], {input_name: benchmark["input"]})[0]
            )
        except Exception as exc:  # noqa: BLE001
            result["errors"] += 1
            if result["first_failure"] is None:
                result["first_failure"] = {
                    "kind": "runtime",
                    "split": split,
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            continue
        finite = bool(np.isfinite(raw).all())
        if not finite:
            result["nonfinite_cases"] += 1
        if raw.shape != benchmark["output"].shape:
            result["output_shape_mismatches"] += 1
            if result["first_failure"] is None:
                result["first_failure"] = {
                    "kind": "output_shape",
                    "split": split,
                    "index": index,
                    "actual": list(raw.shape),
                    "expected": list(benchmark["output"].shape),
                }
            continue
        positive = raw[raw > 0]
        if positive.size:
            minimum = float(np.min(positive))
            result["min_positive"] = (
                minimum
                if result["min_positive"] is None
                else min(result["min_positive"], minimum)
            )
            result["near_positive_values"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        actual = raw > 0
        expected = benchmark["output"] > 0
        if finite and np.array_equal(actual, expected):
            result["right"] += 1
        else:
            result["wrong"] += 1
            if result["first_failure"] is None:
                result["first_failure"] = {
                    "kind": "wrong",
                    "split": split,
                    "index": index,
                    "different_cells": int(np.count_nonzero(actual != expected)),
                }
    result["accuracy"] = result["right"] / result["total"] if result["total"] else 0.0
    return result


def known_four(task: int, data: bytes) -> dict[str, Any]:
    cases = converted_known(task)
    rows = {}
    for disable, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            rows[key] = evaluate_cases(data, cases, disable, threads)
    return rows


def known_perfect(rows: dict[str, Any]) -> bool:
    return len(rows) == 4 and all(
        item.get("right") == item.get("total")
        and item.get("wrong") == 0
        and item.get("errors") == 0
        and item.get("nonfinite_cases") == 0
        and item.get("output_shape_mismatches") == 0
        and item.get("near_positive_values") == 0
        for item in rows.values()
    )


def score_value(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def policy_markers(model: onnx.ModelProto) -> list[str]:
    ops = [node.op_type for node in model.graph.node]
    markers = []
    if any(op in {"TfIdfVectorizer", "Hardmax"} for op in ops):
        markers.append("lookup_op")
    if "ArgMax" in ops and "ScatterND" in ops and "Gather" in ops:
        markers.append("argmax_gather_scatter_lookup_chain")
    if any(node.op_type == "Einsum" and len(node.input) >= 15 for node in model.graph.node):
        markers.append("giant_einsum")
    return markers


def remove_value_infos(model: onnx.ModelProto, names: set[str]) -> None:
    keep = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)


def build_task109_shape_fold(base: bytes) -> onnx.ModelProto:
    model = onnx.load_model_from_string(base)
    kept = []
    removed = False
    for node in model.graph.node:
        if node.op_type == "Shape" and list(node.output) == ["shape7_dyn"]:
            removed = True
            continue
        kept.append(copy.deepcopy(node))
    if not removed:
        raise RuntimeError("task109 Shape node not found")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([12], dtype=np.int64), "shape7_dyn"))
    return model


def bypass_single_use(model: onnx.ModelProto, output: str, replacement: str) -> None:
    kept = []
    removed = False
    for node in model.graph.node:
        if output in node.output:
            removed = True
            continue
        clone = copy.deepcopy(node)
        for index, name in enumerate(clone.input):
            if name == output:
                clone.input[index] = replacement
        kept.append(clone)
    if not removed:
        raise RuntimeError(f"producer for {output} not found")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    remove_value_infos(model, {output})


def build_task184_batch_noop(base: bytes) -> onnx.ModelProto:
    model = onnx.load_model_from_string(base)
    # CenterCropPad(input, Shape(input)[0:1], axes=[0]) is identity because the
    # canonical batch dimension is statically one.
    bypass_single_use(model, "hid", "input")
    return model


def build_task374_batch_noops(base: bytes) -> onnx.ModelProto:
    model = onnx.load_model_from_string(base)
    # All three carriers crop/pad only batch axis 0 to the already-static size 1.
    bypass_single_use(model, "hmask_fake", "hmask")
    bypass_single_use(model, "color_fake", "color_any")
    bypass_single_use(model, "occ_fake", "occ")
    # __sp_shape is now dead.
    kept = [copy.deepcopy(node) for node in model.graph.node if "__sp_shape" not in node.output]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    remove_value_infos(model, {"__sp_shape"})
    return model


def fresh_two_seed(task: int, data: bytes, count: int = 1500) -> dict[str, Any]:
    mapping = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    tasks_dir = ROOT / "inputs/arc-gen-repo/tasks"
    sys.path.insert(0, str(tasks_dir))
    generator = importlib.import_module(f"task_{mapping[f'{task:03d}']}")
    reports = []
    for seed in (374_109_000 + task, 374_184_000 + task):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        cases = []
        attempts = generation_errors = conversion_skips = 0
        while len(cases) < count and attempts < count * 10:
            attempts += 1
            try:
                converted = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if converted is None:
                conversion_skips += 1
                continue
            cases.append(("fresh", len(cases), converted))
        configs = {}
        for disable, mode in ((True, "disable_all"), (False, "default")):
            for threads in (1, 4):
                key = f"{mode}_threads{threads}"
                configs[key] = evaluate_cases(data, cases, disable, threads)
        reports.append(
            {
                "seed": seed,
                "valid": len(cases),
                "attempts": attempts,
                "generation_errors": generation_errors,
                "conversion_skips": conversion_skips,
                "configs": configs,
            }
        )
    return {"count_per_seed": count, "runs": reports}


def candidate_record(task: int, label: str, path: Path, theorem: str) -> dict[str, Any]:
    data = path.read_bytes()
    model = onnx.load_model_from_string(data)
    structural = STATIC.structural_audit(data)
    trace = runtime_trace(task, data)
    prof = profile(task, data, label)
    cost = prof.get("cost") if isinstance(prof, dict) else None
    truthful = not trace.get("error") and not trace.get("declared_actual_mismatches")
    markers = policy_markers(model)
    record: dict[str, Any] = {
        "task": task,
        "label": label,
        "path": rel(path),
        "sha256": sha(data),
        "serialized_bytes": len(data),
        "theorem": theorem,
        "authority_cost": BASE_COSTS[task],
        "official_profile": prof,
        "score_gain": (
            score_value(int(cost)) - score_value(BASE_COSTS[task])
            if isinstance(cost, int) and cost > 0
            else None
        ),
        "strictly_lower": isinstance(cost, int) and 0 < cost < BASE_COSTS[task],
        "structural": structural,
        "runtime_shape_trace": trace,
        "truthful_all_traced_shapes": truthful,
        "policy_markers": markers,
    }
    if record["strictly_lower"]:
        record["known_four"] = known_four(task, data)
        record["known_complete"] = known_perfect(record["known_four"])
    else:
        record["known_four"] = "NOT_RUN_COST_GATE_FAILED"
        record["known_complete"] = False
    eligible = (
        record["strictly_lower"]
        and structural.get("pass")
        and truthful
        and not markers
        and record["known_complete"]
    )
    record["fresh_eligible"] = eligible
    if eligible:
        record["fresh_two_seed"] = fresh_two_seed(task, data)
        rates = [
            cfg["right"] / cfg["total"]
            for run in record["fresh_two_seed"]["runs"]
            for cfg in run["configs"].values()
            if cfg["total"]
        ]
        record["fresh_min_accuracy"] = min(rates) if rates else 0.0
        record["accepted"] = record["fresh_min_accuracy"] >= 0.9
    else:
        failed = []
        if not record["strictly_lower"]:
            failed.append("cost_not_strictly_lower")
        if not structural.get("pass"):
            failed.append("full_checker_or_strict_data_prop")
        if not truthful:
            failed.append("runtime_shape_cloak_or_trace_error")
        if markers:
            failed.append("forbidden_lookup_or_giant")
        if record["strictly_lower"] and not record["known_complete"]:
            failed.append("complete_known_four_config")
        record["fresh_two_seed"] = {"status": "NOT_RUN_PRE_FRESH_GATE_FAILED", "failed": failed}
        record["fresh_min_accuracy"] = None
        record["accepted"] = False
    return record


def main() -> int:
    ort.set_default_logger_severity(4)
    before = {
        "submission.zip": sha(AUTHORITY.read_bytes()),
        "submission_base_8009.46.zip": sha(AUTHORITY_BASE.read_bytes()),
        "all_scores.csv": sha(SCORES.read_bytes()),
    }
    if before["submission.zip"] != AUTHORITY_SHA or before["submission_base_8009.46.zip"] != AUTHORITY_SHA:
        raise RuntimeError(f"authority drift: {before}")
    if before["all_scores.csv"] != SCORES_SHA:
        raise RuntimeError(f"score ledger drift: {before['all_scores.csv']}")

    authority_dir = HERE / "authority"
    candidate_dir = HERE / "candidates"
    authority_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    authority_rows = {}
    for task, data in members.items():
        path = authority_dir / f"task{task:03d}.onnx"
        path.write_bytes(data)
        prof = profile(task, data, f"residual_authority_{task}")
        if prof.get("cost") != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} authority cost drift: {prof}")
        trace = runtime_trace(task, data)
        known = known_four(task, data)
        authority_rows[str(task)] = {
            "task": task,
            "path": rel(path),
            "sha256": sha(data),
            "serialized_bytes": len(data),
            "official_profile": prof,
            "structural": STATIC.structural_audit(data),
            "runtime_shape_trace": trace,
            "truthful_all_traced_shapes": not trace.get("error") and not trace.get("declared_actual_mismatches"),
            "known_four": known,
            "known_complete": known_perfect(known),
        }

    candidates: list[tuple[int, str, Path, str]] = []
    candidates.append(
        (109, "task109_constant_shape_fold", candidate_dir / "task109_constant_shape_fold.onnx",
         "Replace Shape(constant int8[12]) with the exact int64 initializer [12].")
    )
    write_model(candidates[-1][2], build_task109_shape_fold(members[109]))

    source109 = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task109_r02_static177.onnx"
    target109 = candidate_dir / "task109_history_global_lppool.onnx"
    target109.write_bytes(source109.read_bytes())
    candidates.append(
        (109, "task109_history_global_lppool", target109,
         "Retained global-pool parameter shave; screened because 8009.46 corrected the cost gate to 405.")
    )

    candidates.append(
        (184, "task184_batch_identity_bypass", candidate_dir / "task184_batch_identity_bypass.onnx",
         "Bypass the exact batch-axis CenterCropPad identity before CastLike.")
    )
    write_model(candidates[-1][2], build_task184_batch_noop(members[184]))

    source184 = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task184_r04_static170.onnx"
    target184 = candidate_dir / "task184_history_ground_u8.onnx"
    target184.write_bytes(source184.read_bytes())
    candidates.append(
        (184, "task184_history_ground_u8", target184,
         "Only repository history model previously measured below authority (420 versus 421).")
    )

    source368 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_high368_157_370_138/candidates/task368_cast_attribute_af7d8318545c.onnx"
    target368 = candidate_dir / "task368_cast_attribute.onnx"
    target368.write_bytes(source368.read_bytes())
    candidates.append(
        (368, "task368_cast_attribute", target368,
         "Exact CastLike(gn, uint8-zero) to Cast(to=UINT8) attributeization.")
    )

    candidates.append(
        (374, "task374_batch_identity_bypasses", candidate_dir / "task374_batch_identity_bypasses.onnx",
         "Bypass all three exact batch-axis CenterCropPad identities and remove their dead Shape producer.")
    )
    write_model(candidates[-1][2], build_task374_batch_noops(members[374]))

    source374 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_high182_365_374_130/candidates/task374_cast_i32_attribute_b82ae4b99fba.onnx"
    target374 = candidate_dir / "task374_cast_i32_attribute.onnx"
    target374.write_bytes(source374.read_bytes())
    candidates.append(
        (374, "task374_cast_i32_attribute", target374,
         "Exact CastLike(color_fake, int32 dummy) to Cast(to=INT32), deleting one parameter.")
    )

    candidate_rows = [candidate_record(*item) for item in candidates]
    after = {
        "submission.zip": sha(AUTHORITY.read_bytes()),
        "submission_base_8009.46.zip": sha(AUTHORITY_BASE.read_bytes()),
        "all_scores.csv": sha(SCORES.read_bytes()),
    }
    if after != before:
        raise RuntimeError(f"protected files changed: before={before}, after={after}")

    output = {
        "lane": "agent_tasks109_184_368_374",
        "policy": {
            "normal_fresh_min_accuracy": 0.9,
            "known_requirement": "complete known in disable/default ORT x threads 1/4",
            "fresh_requirement": "two disjoint seeds; runtime errors/nonfinite/output-shape mismatch forbidden",
            "private_zero_requirement": "complete pass-through guarantee (not applicable to these normal candidates)",
            "forbidden": ["lookup", "shape cloak", "runtime error", "nonfinite", "shape mismatch", "UB"],
        },
        "authority_archive": rel(AUTHORITY),
        "authority_sha256": AUTHORITY_SHA,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "authority": authority_rows,
        "candidates": candidate_rows,
        "accepted": [row for row in candidate_rows if row["accepted"]],
        "accepted_count": sum(row["accepted"] for row in candidate_rows),
        "projected_gain": sum(float(row["score_gain"] or 0.0) for row in candidate_rows if row["accepted"]),
        "verdict": "NO_PROMOTION" if not any(row["accepted"] for row in candidate_rows) else "PROMOTION_CANDIDATE",
    }
    (HERE / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "authority": {task: row["official_profile"]["cost"] for task, row in authority_rows.items()},
        "candidates": [
            {
                "label": row["label"],
                "cost": row["official_profile"].get("cost"),
                "lower": row["strictly_lower"],
                "truthful": row["truthful_all_traced_shapes"],
                "known": row["known_complete"],
                "accepted": row["accepted"],
            }
            for row in candidate_rows
        ],
        "verdict": output["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
