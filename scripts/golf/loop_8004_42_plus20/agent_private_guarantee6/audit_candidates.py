#!/usr/bin/env python3
"""Non-promoting guarantee audit for private-zero tasks 035/066/090/377."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
MAP = {35: "1f642eb9", 66: "2dd70a9a", 90: "3eda0437", 377: "eb5a1d5d"}
BASE_COST = {35: 545, 66: 677, 90: 1050, 377: 409}
SEED = 914_066_090
FRESH_COUNT = 1000

CANDIDATES = {
    35: [
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task035_r01_static493.onnx",
    ],
    66: [
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task066_r01_static368.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task066_r02_static582.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task066_r03_static583.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task066_r04_static636.onnx",
    ],
    90: [
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r01_static130.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r02_static174.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r03_static208.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r04_static226.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r05_static400.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r06_static418.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r07_static430.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task090_r08_static431.onnx",
    ],
    377: [
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r01_static235.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r02_static236.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r03_static242.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r04_static242.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r05_static247.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r06_static247.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r07_static248.onnx",
        "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task377_r08_static248.onnx",
    ],
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from harvest import actual_screen, exact_conv_bias_gate  # noqa: E402
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ref035(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int64)
    rr, cc = np.where(a == 8)
    if rr.size == 0:
        raise ValueError("cyan pool not found")
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    out = a.copy()
    h, w = a.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            coords = []
            if r == r0:
                coords.append((0, c))
            if r == r1:
                coords.append((h - 1, c))
            if c == c0:
                coords.append((r, 0))
            if c == c1:
                coords.append((r, w - 1))
            for er, ec in coords:
                if a[er, ec] not in (0, 8):
                    out[r, c] = a[er, ec]
    return out.tolist()


def ref090(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int64)
    h, w = a.shape
    best = -1
    winners: list[tuple[int, int, int, int]] = []
    for r2 in range(h):
        for r1 in range(r2):
            for c2 in range(w):
                for c1 in range(c2):
                    if np.any(a[r1 : r2 + 1, c1 : c2 + 1]):
                        continue
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area > best:
                        best, winners = area, []
                    if area == best:
                        winners.append((r1, r2, c1, c2))
    if len(winners) != 1:
        raise ValueError(f"largest zero rectangle not unique: {len(winners)}")
    r1, r2, c1, c2 = winners[0]
    out = a.copy()
    out[r1 : r2 + 1, c1 : c2 + 1] = 6
    return out.tolist()


def ref377(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int64)
    r0 = c0 = 0
    r1, c1 = a.shape[0] - 1, a.shape[1] - 1
    colors: list[int] = []
    while True:
        color = int(a[r0, c0])
        colors.append(color)
        sub = a[r0 : r1 + 1, c0 : c1 + 1]
        rr, cc = np.where(sub != color)
        if rr.size == 0:
            break
        r0, r1 = r0 + int(rr.min()), r0 + int(rr.max())
        c0, c1 = c0 + int(cc.min()), c0 + int(cc.max())
    size = 2 * len(colors) - 1
    out = np.zeros((size, size), dtype=np.int64)
    for i, color in enumerate(colors):
        out[i : size - i, i : size - i] = color
    return out.tolist()


def _untransform(a: np.ndarray, flip: int, hflip: int, xpose: int) -> np.ndarray:
    if xpose:
        a = a.T
    if hflip:
        a = a[:, ::-1]
    if flip:
        a = a[::-1]
    return a


def _transform(a: np.ndarray, flip: int, hflip: int, xpose: int) -> np.ndarray:
    if flip:
        a = a[::-1]
    if hflip:
        a = a[:, ::-1]
    if xpose:
        a = a.T
    return a


def ref066(grid: list[list[int]]) -> list[list[int]]:
    """Enumerate the generator's S/U latent geometry and require one output."""
    observed = np.asarray(grid, dtype=np.int64)
    outputs: dict[bytes, np.ndarray] = {}
    for flip in (0, 1):
        for hflip in (0, 1):
          for xpose in (0, 1):
            a = _untransform(observed, flip, hflip, xpose)
            red = np.argwhere(a == 2)
            green = np.argwhere(a == 3)
            if red.shape != (2, 2) or green.shape != (2, 2):
                continue
            if len(set(red[:, 1])) != 1 or len(set(green[:, 1])) != 1:
                continue
            if int(np.ptp(red[:, 0])) != 1 or int(np.ptp(green[:, 0])) != 1:
                continue
            cr, cl = int(red[0, 1]), int(green[0, 1])
            if cr <= cl:
                continue
            rr0, rr1 = int(red[:, 0].min()), int(red[:, 0].max())
            rg0, rg1 = int(green[:, 0].min()), int(green[:, 0].max())
            h, w = a.shape

            # S: red starts the right leg, green ends the left leg.
            for mid in range(rr1 + 1, rg0):
                manual_tight_turn = mid == rr1 + 1 and rg0 == mid + 1
                if cr + 1 >= w or (
                    not manual_tight_turn
                    and (a[mid - 1, cl] != 8 or a[mid, cr + 1] != 8)
                ):
                    continue
                path = (
                    [(r, cr) for r in range(rr0, mid)]
                    + [(mid, c) for c in range(cl, cr + 1)]
                    + [(r, cl) for r in range(mid + 1, rg1 + 1)]
                )
                if any(a[r, c] not in (0, 2, 3) for r, c in path):
                    continue
                out = a.copy()
                for r, c in path:
                    if out[r, c] == 0:
                        out[r, c] = 3
                out = _transform(out, flip, hflip, xpose)
                outputs[out.tobytes()] = out

            # U: two vertical legs meet on the bottom horizontal.
            for bottom in range(max(rr1, rg1), h - 1):
                if cr + 1 >= w or cl < 0 or cr - 1 < 0:
                    continue
                if a[bottom + 1, cl] != 8 or a[bottom, cr + 1] != 8:
                    continue
                if a[rr0, cr - 1] != 8:
                    continue
                path = (
                    [(r, cr) for r in range(rr0, bottom + 1)]
                    + [(bottom, c) for c in range(cl, cr)]
                    + [(r, cl) for r in range(bottom, rg0 - 1, -1)]
                )
                if any(a[r, c] not in (0, 2, 3) for r, c in path):
                    continue
                out = a.copy()
                for r, c in path:
                    if out[r, c] == 0:
                        out[r, c] = 3
                out = _transform(out, flip, hflip, xpose)
                outputs[out.tobytes()] = out
    if len(outputs) != 1:
        raise ValueError(f"S/U reconstruction not unique: {len(outputs)}")
    return next(iter(outputs.values())).tolist()


REFERENCES: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    35: ref035,
    66: ref066,
    90: ref090,
    377: ref377,
}


def reference_known(task: int) -> dict[str, Any]:
    stats: dict[str, Any] = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(scoring.load_examples(task)[subset]):
            try:
                actual = REFERENCES[task](example["input"])
                if actual == example["output"]:
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
                    stats["first_failure"] = stats["first_failure"] or {"subset": subset, "index": index}
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                stats["first_failure"] = stats["first_failure"] or {
                    "subset": subset,
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    stats["perfect"] = stats["wrong"] == stats["errors"] == 0
    return stats


def make_session(data: bytes, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_dual(task: int, data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    examples = scoring.load_examples(task)
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        row: dict[str, Any] = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
        try:
            session = make_session(data, disabled)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["errors"] = sum(len(examples[s]) for s in ("train", "test", "arc-gen"))
            result[mode] = row
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = session.run(["output"], {"input": benchmark["input"]})[0]
                    if np.array_equal(raw > 0, benchmark["output"] > 0):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        row["first_failure"] = row["first_failure"] or {"subset": subset, "index": index}
                except Exception as exc:  # noqa: BLE001
                    row["errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "subset": subset,
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        result[mode] = row
    return result


def fresh_probe(task: int, data: bytes) -> dict[str, Any]:
    module = importlib.import_module(f"task_{MAP[task]}")
    sessions = {
        "disable_all": make_session(data, True),
        "default": make_session(data, False),
    }
    rows = {mode: {"right": 0, "wrong": 0, "errors": 0, "first_failure": None} for mode in sessions}
    reference = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
    random.seed(SEED + task)
    for index in range(FRESH_COUNT):
        example = module.generate()
        try:
            decoded = REFERENCES[task](example["input"])
            if decoded == example["output"]:
                reference["right"] += 1
            else:
                reference["wrong"] += 1
                reference["first_failure"] = reference["first_failure"] or index
        except Exception as exc:  # noqa: BLE001
            reference["errors"] += 1
            reference["first_failure"] = reference["first_failure"] or f"{index}:{type(exc).__name__}:{exc}"
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        for mode, session in sessions.items():
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    rows[mode]["right"] += 1
                else:
                    rows[mode]["wrong"] += 1
                    rows[mode]["first_failure"] = rows[mode]["first_failure"] or index
            except Exception as exc:  # noqa: BLE001
                rows[mode]["errors"] += 1
                rows[mode]["first_failure"] = rows[mode]["first_failure"] or f"{index}:{type(exc).__name__}:{exc}"
    return {"count": FRESH_COUNT, "seed": SEED + task, "reference": reference, "modes": rows}


def dims_static_positive(inferred: onnx.ModelProto) -> bool:
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            return False
        if any(not d.HasField("dim_value") or d.dim_value <= 0 for d in value.type.tensor_type.shape.dim):
            return False
    return True


def structural_audit(task: int, data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checker_full": False,
        "strict_shape_data_prop": False,
        "all_inferred_dims_static_positive": False,
        "truthful_runtime_shapes": False,
        "conv_bias_ub0": False,
    }
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["checker_full"] = True
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row["strict_shape_data_prop"] = True
        row["all_inferred_dims_static_positive"] = dims_static_positive(inferred)
    except Exception as exc:  # noqa: BLE001
        row["checker_error"] = f"{type(exc).__name__}: {exc}"
        return row
    ops = Counter(node.op_type for node in model.graph.node)
    row["ops"] = dict(sorted(ops.items()))
    row["lookup"] = bool(ops.get("TfIdfVectorizer") or ops.get("Hardmax"))
    row["max_einsum_inputs"] = max((len(n.input) for n in model.graph.node if n.op_type == "Einsum"), default=0)
    row["giant_contraction"] = row["max_einsum_inputs"] >= 15
    bias_ok, bias_reason, bias_findings = exact_conv_bias_gate(model)
    row["conv_bias_ub0"] = bias_ok
    row["conv_bias_reason"] = bias_reason
    row["conv_bias_findings"] = bias_findings
    try:
        trace = runtime_shape_trace(task, model)
        row["runtime_shape_trace"] = trace
        row["truthful_runtime_shapes"] = not trace["declared_actual_mismatches"]
    except Exception as exc:  # noqa: BLE001
        row["runtime_shape_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_sha = {task: digest(archive.read(f"task{task:03d}.onnx")) for task in MAP}
    report: dict[str, Any] = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASE_ZIP.read_bytes()),
        "fresh_reject_probe": {"count": FRESH_COUNT, "seed_base": SEED},
        "tasks": {},
        "accepted": [],
        "verified_gain": 0.0,
    }
    for task in MAP:
        task_row: dict[str, Any] = {
            "rule": {
                35: "Project every non-cyan border marker onto the nearest cell of the solid cyan rectangle.",
                66: "Recover the hidden blue S/U orthogonal path between the red start pair and green end pair; recolor path cells green.",
                90: "Find the unique maximum-area all-black axis-aligned rectangle and paint it pink (6).",
                377: "Read colors of nested filled rectangles and render them as an odd concentric square.",
            }[task],
            "generator_module": f"inputs/arc-gen-repo/tasks/task_{MAP[task]}.py",
            "reference_known": reference_known(task),
            "baseline": {"cost": BASE_COST[task], "sha256": baseline_sha[task]},
            "candidates": [],
        }
        for path_text in CANDIDATES[task]:
            path = ROOT / path_text
            data = path.read_bytes()
            structural = structural_audit(task, data)
            cost = actual_screen(data, task)
            row: dict[str, Any] = {
                "path": path_text,
                "sha256": digest(data),
                "serialized_bytes": len(data),
                "profiler_cost": cost,
                "baseline_cost": BASE_COST[task],
                "apparent_gain": math.log(BASE_COST[task] / cost) if cost and cost < BASE_COST[task] else 0.0,
                "strictly_cheaper_profiler": cost is not None and cost < BASE_COST[task],
                "structural": structural,
            }
            early_ok = (
                row["strictly_cheaper_profiler"]
                and structural["checker_full"]
                and structural["strict_shape_data_prop"]
                and structural["all_inferred_dims_static_positive"]
                and structural["truthful_runtime_shapes"]
                and structural["conv_bias_ub0"]
                and not structural.get("lookup")
                and not structural.get("giant_contraction")
            )
            if early_ok:
                row["known_dual"] = known_dual(task, data)
                known_ok = all(
                    mode["wrong"] == mode["errors"] == 0
                    for mode in row["known_dual"].values()
                )
                if known_ok:
                    row["fresh_reject_probe"] = fresh_probe(task, data)
            reasons = []
            if not row["strictly_cheaper_profiler"]:
                reasons.append("not_strictly_cheaper_actual_profiler_cost")
            if not structural["checker_full"] or not structural["strict_shape_data_prop"]:
                reasons.append("checker_or_strict_data_prop")
            if not structural["all_inferred_dims_static_positive"]:
                reasons.append("nonpositive_or_symbolic_shape")
            if not structural["truthful_runtime_shapes"]:
                reasons.append("shape_cloak_or_runtime_trace_failure")
            if not structural["conv_bias_ub0"]:
                reasons.append("conv_family_bias_ub")
            if structural.get("lookup"):
                reasons.append("lookup")
            if structural.get("giant_contraction"):
                reasons.append("giant_contraction")
            if "known_dual" in row and any(m["wrong"] or m["errors"] for m in row["known_dual"].values()):
                reasons.append("complete_known_not_100_or_runtime_error")
            if "fresh_reject_probe" in row:
                if any(m["wrong"] or m["errors"] for m in row["fresh_reject_probe"]["modes"].values()):
                    reasons.append("fresh_seed_not_100")
                if row["fresh_reject_probe"]["reference"]["wrong"] or row["fresh_reject_probe"]["reference"]["errors"]:
                    reasons.append("decoded_reference_not_100")
            row["decision"] = "REJECT"
            row["reasons"] = sorted(set(reasons))
            task_row["candidates"].append(row)
            print(task, path.name, row["decision"], row["reasons"], flush=True)
        report["tasks"][str(task)] = task_row
    (HERE / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
