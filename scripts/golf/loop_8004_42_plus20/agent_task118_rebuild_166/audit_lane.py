#!/usr/bin/env python3
"""SOUND task118 true-rule rebuild audit against immutable LB 8009.46."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ROOT_SCORES = ROOT / "all_scores.csv"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
SCORES_SHA = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
BASELINE_SHA = "a7c763ba3468863d1cebdf97522fc613052ab5d435af51b8d9035d413c096ab8"
TASK = 118
FRESH_COUNT = 5000
FRESH_SEED = 166_118_001
MODE_LEVELS = {
    "default": None,
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "minimal": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
}
MODELS = {
    "authority": HERE / "baseline/task118.onnx",
    "observable_rule": HERE / "candidates/task118_observable_rule.onnx",
    "full_roi_control": HERE / "candidates/task118_full_roi_control.onnx",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "task118_166_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
DEEP = load_module(
    "task118_166_deep",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
TEAM = load_module(
    "task118_166_team",
    ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha(path.read_bytes())


def make_session(data: bytes, mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(onnx.load_model_from_string(data))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    level = MODE_LEVELS[mode]
    if level is not None:
        options.graph_optimization_level = level
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def run_case(session: ort.InferenceSession, benchmark: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(
        session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
    )
    positive = raw[raw > 0]
    return raw, {
        "correct": bool(np.array_equal(raw > 0, benchmark["output"].astype(bool))),
        "nonfinite": int(raw.size - np.count_nonzero(np.isfinite(raw))),
        "near_positive": int(np.count_nonzero((raw > 0) & (raw < 0.25))),
        "min_positive": float(positive.min()) if positive.size else None,
        "max_positive": float(positive.max()) if positive.size else None,
    }


def known_four(data: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(TASK)
    ordered = [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[split])
    ]
    result: dict[str, Any] = {}
    raw_by_mode: dict[str, list[np.ndarray | None]] = {}
    for mode in MODE_LEVELS:
        row: dict[str, Any] = {
            "total": len(ordered), "right": 0, "errors": 0, "nonfinite": 0,
            "near_positive": 0, "min_positive": None, "first_failure": None,
        }
        raws: list[np.ndarray | None] = []
        try:
            session = make_session(data, mode)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["pass"] = False
            result[mode] = row
            raw_by_mode[mode] = [None] * len(ordered)
            continue
        for split, index, example in ordered:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                row["errors"] += 1
                raws.append(None)
                row["first_failure"] = row["first_failure"] or {"split": split, "index": index, "error": "conversion"}
                continue
            try:
                raw, stats = run_case(session, benchmark)
                raws.append(raw)
            except Exception as exc:  # noqa: BLE001
                row["errors"] += 1
                raws.append(None)
                row["first_failure"] = row["first_failure"] or {"split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
                continue
            row["right"] += int(stats["correct"])
            row["nonfinite"] += stats["nonfinite"]
            row["near_positive"] += stats["near_positive"]
            value = stats["min_positive"]
            if value is not None:
                row["min_positive"] = value if row["min_positive"] is None else min(row["min_positive"], value)
            if not stats["correct"]:
                row["first_failure"] = row["first_failure"] or {"split": split, "index": index, "error": "threshold_mismatch"}
        row["pass"] = row["right"] == row["total"] and row["errors"] == row["nonfinite"] == row["near_positive"] == 0
        result[mode] = row
        raw_by_mode[mode] = raws
    reference = raw_by_mode["disabled"]
    for mode, raws in raw_by_mode.items():
        raw_equal = sign_equal = 0
        for left, right in zip(reference, raws):
            if left is None or right is None:
                continue
            raw_equal += int(np.array_equal(left, right))
            sign_equal += int(np.array_equal(left > 0, right > 0))
        result[mode]["raw_equal_to_disabled"] = raw_equal
        result[mode]["sign_equal_to_disabled"] = sign_equal
    return result


def fresh_four(data: bytes, seed: int = FRESH_SEED, count: int = FRESH_COUNT) -> dict[str, Any]:
    generator = importlib.import_module("task_50846271")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    sessions: dict[str, ort.InferenceSession | None] = {}
    result: dict[str, Any] = {"seed": seed, "count": count, "generation_errors": 0, "modes": {}}
    for mode in MODE_LEVELS:
        try:
            sessions[mode] = make_session(data, mode)
            result["modes"][mode] = {
                "right": 0, "errors": 0, "nonfinite": 0, "near_positive": 0,
                "raw_equal_to_disabled": 0, "sign_equal_to_disabled": 0,
                "first_failure": None,
            }
        except Exception as exc:  # noqa: BLE001
            sessions[mode] = None
            result["modes"][mode] = {"right": 0, "errors": count, "session_error": f"{type(exc).__name__}: {exc}", "first_failure": {"case": 0, "error": "session"}}
    index = 0
    while index < count:
        try:
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
        except Exception:  # noqa: BLE001
            result["generation_errors"] += 1
            continue
        if benchmark is None:
            result["generation_errors"] += 1
            continue
        case_raw: dict[str, np.ndarray] = {}
        for mode, session in sessions.items():
            row = result["modes"][mode]
            if session is None:
                continue
            try:
                raw, stats = run_case(session, benchmark)
                case_raw[mode] = raw
            except Exception as exc:  # noqa: BLE001
                row["errors"] += 1
                row["first_failure"] = row["first_failure"] or {"case": index, "error": f"{type(exc).__name__}: {exc}"}
                continue
            row["right"] += int(stats["correct"])
            row["nonfinite"] += stats["nonfinite"]
            row["near_positive"] += stats["near_positive"]
            if not stats["correct"]:
                row["first_failure"] = row["first_failure"] or {"case": index, "error": "threshold_mismatch"}
        disabled = case_raw.get("disabled")
        if disabled is not None:
            for mode, raw in case_raw.items():
                result["modes"][mode]["raw_equal_to_disabled"] += int(np.array_equal(disabled, raw))
                result["modes"][mode]["sign_equal_to_disabled"] += int(np.array_equal(disabled > 0, raw > 0))
        index += 1
    for mode in MODE_LEVELS:
        row = result["modes"][mode]
        row["rate"] = row["right"] / count
        row["pass90"] = row["rate"] >= 0.90 and row["errors"] == row.get("nonfinite", 0) == row.get("near_positive", 0) == 0
    return result


def official_profile(data: bytes, label: str) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"task118_166_{label}_", dir="/tmp") as work:
        value = scoring.score_and_verify(copy.deepcopy(model), TASK, work, label=label, require_correct=False)
    if value is None:
        raise RuntimeError(f"profile failed {label}")
    return value


def team_profile(data: bytes, label: str) -> dict[str, Any]:
    audit, failures = TEAM.audit_model_bytes(
        data, TASK, ROOT / "inputs/neurogolf-2026", source=label,
        keep_trace=False, trace_dir=HERE / "traces",
    )
    row = dataclasses.asdict(audit)
    row["failures"] = failures
    return row


def sakana_reference() -> dict[str, Any]:
    module = load_module("task118_166_sakana", ROOT / "inputs/sakana-gcg-2025/raw/task118.py")
    known = json.loads((ROOT / "inputs/neurogolf-2026/task118.json").read_text())
    right = total = 0
    failures = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(known[split]):
            total += 1
            got = module.p(copy.deepcopy(example["input"]))
            ok = got == example["output"]
            right += int(ok)
            if not ok and len(failures) < 10:
                failures.append({"split": split, "index": index})
    generator = importlib.import_module("task_50846271")
    streams = []
    for seed in (166_118_101, 166_118_102):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stream_right = 0
        for _ in range(FRESH_COUNT):
            example = generator.generate()
            stream_right += int(module.p(copy.deepcopy(example["input"])) == example["output"])
        streams.append({"seed": seed, "right": stream_right, "total": FRESH_COUNT, "rate": stream_right / FRESH_COUNT})
    return {"known": {"right": right, "total": total, "failures": failures}, "fresh": streams}


def noninjective_witness() -> dict[str, Any]:
    height = width = 10
    radius = 2
    centers = ((3, 3), (6, 6))
    input_grid = np.full((height, width), 5, dtype=np.uint8)
    outputs = []
    supports = []
    for row, col in centers:
        output = input_grid.copy()
        support = sorted({(row, col + d) for d in range(-radius, radius + 1)} | {(row + d, col) for d in range(-radius, radius + 1)})
        for rr, cc in support:
            output[rr, cc] = 8
        outputs.append(output)
        supports.append(support)
    return {
        "generator_reachable": True,
        "construction": "10x10 all-gray static, radius2, four identical center attempts; first accepted and remaining three rejected",
        "center_a": list(centers[0]),
        "center_b": list(centers[1]),
        "same_input": True,
        "different_outputs": bool(not np.array_equal(outputs[0], outputs[1])),
        "input_sha256": sha(input_grid.tobytes()),
        "output_a_sha256": sha(outputs[0].tobytes()),
        "output_b_sha256": sha(outputs[1].tobytes()),
        "positive_probability": "all static gray, selected radius, grid size, and repeated attempted center each have positive generator probability",
        "conclusion": "no deterministic input-only ONNX can be all-input equivalent",
    }


def main() -> int:
    before = {"submission": sha_file(ROOT_SUBMISSION), "all_scores": sha_file(ROOT_SCORES)}
    if sha_file(AUTHORITY_ZIP) != AUTHORITY_SHA or before["submission"] != AUTHORITY_SHA or before["all_scores"] != SCORES_SHA:
        raise RuntimeError("root authority changed")
    if sha_file(MODELS["authority"]) != BASELINE_SHA:
        raise RuntimeError("task118 authority member changed")
    result: dict[str, Any] = {
        "lane": "agent_task118_rebuild_166",
        "authority_zip_sha256": AUTHORITY_SHA,
        "authority_task118_sha256": BASELINE_SHA,
        "true_rule": {
            "classification": "global bounded set-cover / non-injective inverse",
            "pseudocode": [
                "Input contains 0/5 static plus radius-2 or radius-3 crosses; visible cross cells over black are red2, hidden cells over gray appear gray5.",
                "For radius2 then radius3, enumerate plus supports containing only red2/gray5 and use disjoint exact cover of every red2 cell.",
                "On the chosen supports, change gray5 to cyan8; leave all other cells unchanged.",
            ],
        },
        "noninjective_witness": noninjective_witness(),
        "sakana_reference": sakana_reference(),
        "models": {},
        "winners": [],
        "projected_gain": 0.0,
        "root_before": before,
    }
    for label, path in MODELS.items():
        print(f"AUDIT {label}", flush=True)
        data = path.read_bytes()
        structural = SCAN.structural(copy.deepcopy(onnx.load_model_from_string(data)))
        trace = DEEP.direct_trace(TASK, data)
        official = official_profile(data, label)
        team = team_profile(data, label)
        known = known_four(data)
        fresh = fresh_four(data)
        reasons = []
        if label != "authority" and official["cost"] >= 3665:
            reasons.append("not_strict_lower_than_3665")
        if label != "authority" and not structural.get("pass", False):
            reasons.append("structure_failed")
        if label != "authority" and not trace.get("truthful", False):
            reasons.append("runtime_shapes_not_truthful")
        if label != "authority" and not all(row.get("pass", False) for row in known.values()):
            reasons.append("known_four_modes_failed")
        if label != "authority" and not all(row.get("pass90", False) for row in fresh["modes"].values()):
            reasons.append("fresh90_failed")
        if label != "authority":
            reasons.append("private_zero_guarantee_not_closed_noninjective_generator")
        result["models"][label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(data),
            "file_size": len(data),
            "structural": structural,
            "runtime_node_shape_trace": trace,
            "competition_actual": {"scoring": official, "team_validator": team, "costs_agree": official["cost"] == team["cost"]},
            "known_four_modes": known,
            "fresh_5000_four_modes": fresh,
            "accepted": False,
            "reasons": reasons,
        }
        (HERE / "result.partial.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["architecture_floor"] = {
        "observable_rule_decode_f32": 25 * 28 * 4,
        "observable_rule_cast_u8": 25 * 28,
        "subtotal_before_detection": 25 * 28 * 5,
        "authority_cost": 3665,
        "explanation": "truthful Conv color-code plus required uint8 cast already consumes 3500 activation bytes; parameters and any center/detection tensor force cost above authority",
    }
    after = {"submission": sha_file(ROOT_SUBMISSION), "all_scores": sha_file(ROOT_SCORES)}
    result["root_after"] = after
    result["root_unchanged"] = before == after
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
