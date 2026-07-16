#!/usr/bin/env python3
"""Fail-closed SOUND/memshave lane for task077 and task361."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
CAPTURED_ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (77, 361)
HASHES = {77: "36fdfd69", 361: "e40b9e2f"}
MEMBER_SHA256 = {
    77: "db46560f4e633e057f960fe2db040f62b118051fc1bbbfc6e29871fcd0e84d56",
    361: "d606fcf6e11548c562db31cd942be67071b5932b9510a3858f87dd9ca4f315e4",
}
BASE_COSTS = {77: 3364, 361: 844}
KINDS = (
    "cleanup",
    "dedupe",
    "noops",
    "cse",
    "optional",
    "fold",
    "absorb",
    "combined",
    "normalize",
    "normalized_combined",
)
REFERENCE_SEEDS = (129_000_000, 129_100_000)
REFERENCE_FRESH_PER_SEED = 1500
ONNX_FRESH_PER_SEED = 1500

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "lane129_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
AUDIT = load_module(
    "lane129_audit",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
RANK = load_module("lane129_rank", ROOT / "scripts/golf/rank_dir.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def normalize(grid: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid]


def solve_077(grid: list[list[int]]) -> list[list[int]]:
    """Restore static-colored cells hidden inside every generator rectangle."""
    height, width = len(grid), len(grid[0])
    red = [[value == 2 for value in row] for row in grid]
    covered = [[False] * width for _ in range(height)]
    for tall in (2, 3):
        for wide in range(2, 8):
            for top in range(height - tall + 1):
                for left in range(width - wide + 1):
                    if not any(red[top][c] for c in range(left, left + wide)):
                        continue
                    if not any(red[top + tall - 1][c] for c in range(left, left + wide)):
                        continue
                    if not any(red[r][left] for r in range(top, top + tall)):
                        continue
                    if not any(red[r][left + wide - 1] for r in range(top, top + tall)):
                        continue
                    column_empty = [
                        not any(red[r][c] for r in range(top, top + tall))
                        for c in range(left, left + wide)
                    ]
                    if any(a and b for a, b in zip(column_empty, column_empty[1:])):
                        continue
                    for row in range(top, top + tall):
                        for col in range(left, left + wide):
                            covered[row][col] = True
    output = copy.deepcopy(grid)
    for row in range(height):
        for col in range(width):
            if covered[row][col] and grid[row][col] not in (0, 2):
                output[row][col] = 4
    return output


def find_center_twice(grid: list[list[int]]) -> tuple[int, int]:
    """Return twice the C4 center from the generator-mandated full core."""
    mask = [[value != 0 for value in row] for row in grid]
    for row in range(1, 9):
        for col in range(1, 9):
            if all(mask[r][c] for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)):
                return 2 * row, 2 * col
    for row in range(9):
        for col in range(9):
            if all(mask[r][c] for r in range(row, row + 2) for c in range(col, col + 2)):
                return 2 * row + 1, 2 * col + 1
    raise ValueError("task361 generator core not found")


def solve_361(grid: list[list[int]]) -> list[list[int]]:
    """Complete every colored pixel's fourfold rotation orbit."""
    height, width = len(grid), len(grid[0])
    two_row, two_col = find_center_twice(grid)
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == 0:
                continue
            rr, cc = row, col
            for _ in range(4):
                if 0 <= rr < height and 0 <= cc < width:
                    output[rr][cc] = color
                rr, cc = (two_row + two_col) // 2 - cc, (two_col - two_row) // 2 + rr
    return output


SOLVERS: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    77: solve_077,
    361: solve_361,
}


def verify_references() -> dict[str, Any]:
    result: dict[str, Any] = {
        "seeds": list(REFERENCE_SEEDS),
        "fresh_per_seed": REFERENCE_FRESH_PER_SEED,
        "tasks": {},
    }
    for task in TASKS:
        solver = SOLVERS[task]
        stored = scoring.load_examples(task)
        known_examples = stored["train"] + stored["test"] + stored["arc-gen"]
        known_right = sum(
            solver(normalize(example["input"])) == normalize(example["output"])
            for example in known_examples
        )
        generator = importlib.import_module(f"task_{HASHES[task]}")
        streams = []
        for seed_base in REFERENCE_SEEDS:
            seed = seed_base + task
            random.seed(seed)
            right = errors = 0
            first_failure = None
            for index in range(REFERENCE_FRESH_PER_SEED):
                try:
                    example = generator.generate()
                    actual = solver(normalize(example["input"]))
                    ok = actual == normalize(example["output"])
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    ok = False
                    if first_failure is None:
                        first_failure = {"index": index, "error": f"{type(exc).__name__}: {exc}"}
                right += int(ok)
                if not ok and first_failure is None:
                    first_failure = {"index": index}
            streams.append(
                {
                    "seed": seed,
                    "right": right,
                    "total": REFERENCE_FRESH_PER_SEED,
                    "errors": errors,
                    "first_failure": first_failure,
                }
            )
        passed = known_right == len(known_examples) and all(
            stream["right"] == REFERENCE_FRESH_PER_SEED and stream["errors"] == 0
            for stream in streams
        )
        result["tasks"][str(task)] = {
            "hash": HASHES[task],
            "known": {"right": known_right, "total": len(known_examples)},
            "fresh": streams,
            "pass": passed,
        }
        print(f"REF task{task:03d} known={known_right}/{len(known_examples)}", flush=True)
    return result


def value_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    record: dict[str, Any] = {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": scoring.calculate_params(model),
        "ops": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "input_shape": value_shape(model.graph.input[0]),
        "output_shape": value_shape(model.graph.output[0]),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        record["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(checker_full=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(
            strict_shape_data_prop=False,
            strict_shape_error=f"{type(exc).__name__}: {exc}",
        )
    return record


def audit_cost(path: Path) -> dict[str, int]:
    memory, params, cost = RANK.cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def make_session(data: bytes, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def fresh_onnx(task: int, data: bytes, count: int = ONNX_FRESH_PER_SEED) -> dict[str, Any]:
    sessions: dict[str, ort.InferenceSession] = {}
    session_errors: dict[str, str] = {}
    for mode, disable_all in (("disable_all", True), ("default", False)):
        try:
            sessions[mode] = make_session(data, disable_all)
        except Exception as exc:  # noqa: BLE001
            session_errors[mode] = f"{type(exc).__name__}: {exc}"
    if session_errors:
        return {
            "count_per_seed": count,
            "session_errors": session_errors,
            "runs": [],
            "pass": False,
            "not_run_reason": "both ORT sessions are required before fresh",
        }

    generator = importlib.import_module(f"task_{HASHES[task]}")
    runs = []
    for seed in (129_200_000 + task, 129_300_000 + task):
        random.seed(seed)
        stats = {
            key: {
                "right": 0,
                "wrong": 0,
                "errors": 0,
                "nonfinite": 0,
                "near_positive": 0,
                "min_positive": None,
            }
            for key in sessions
        }
        valid = 0
        while valid < count:
            benchmark = scoring.convert_to_numpy(generator.generate())
            if benchmark is None:
                continue
            valid += 1
            expected = benchmark["output"] > 0
            for mode, session in sessions.items():
                try:
                    raw = np.asarray(
                        session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    if not np.isfinite(raw).all():
                        stats[mode]["nonfinite"] += 1
                    positives = raw[raw > 0]
                    if positives.size:
                        minimum = float(positives.min())
                        previous = stats[mode]["min_positive"]
                        stats[mode]["min_positive"] = minimum if previous is None else min(previous, minimum)
                        stats[mode]["near_positive"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                    if np.array_equal(raw > 0, expected):
                        stats[mode]["right"] += 1
                    else:
                        stats[mode]["wrong"] += 1
                except Exception:  # noqa: BLE001
                    stats[mode]["errors"] += 1
        runs.append({"seed": seed, "valid": valid, "modes": stats})
    passed = all(
        mode["right"] == count
        and mode["wrong"] == 0
        and mode["errors"] == 0
        and mode["nonfinite"] == 0
        and mode["near_positive"] == 0
        for run in runs
        for mode in run["modes"].values()
    )
    return {"count_per_seed": count, "session_errors": {}, "runs": runs, "pass": passed}


def total_perfect(record: dict[str, Any], expected: int) -> bool:
    if "session_error" in record:
        return False
    total = record.get("total", record)
    return (
        total.get("right") == expected
        and total.get("wrong", 0) == 0
        and total.get("errors", 0) == 0
    )


def truthful_runtime(record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    trace = record.get("runtime_shape_trace") or {}
    mismatches = trace.get("declared_actual_mismatches", [])
    ok = bool(trace) and "error" not in trace and len(mismatches) == 0
    return ok, {
        "pass": ok,
        "error": trace.get("error"),
        "declared_actual_mismatch_count": len(mismatches),
        "single_example_intermediate_bytes": trace.get("single_example_intermediate_bytes"),
    }


def historical_evidence() -> dict[str, Any]:
    t77_base = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_rebuild_high4/baseline_task077_audit.json").read_text()
    )
    t77_sound = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_rebuild_high4/candidate_task077_exact_convpack_audit.json").read_text()
    )
    t361_current = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_71405_tail_99/audit/rows/task361_cost844.json").read_text()
    )
    frontier = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/root_high55/history_lead_audit.json").read_text()
    )
    return {
        "read_only_only_no_candidate_reused": True,
        "task077": {
            "current_lineage_prior_audit": {
                "sha256": MEMBER_SHA256[77],
                "known": [t77_base["known_right"], t77_base["known_total"]],
                "fresh": t77_base["fresh"],
            },
            "sound_rebuild_prior_audit": {
                "sha256": sha256(
                    (
                        ROOT
                        / "scripts/golf/loop_8004_42_plus20/agent_rebuild_high4/task077_exact_convpack.onnx"
                    ).read_bytes()
                ),
                "known": [t77_sound["known_right"], t77_sound["known_total"]],
                "fresh": t77_sound["fresh"],
                "cost": t77_sound["score_measurement"]["cost"],
            },
        },
        "task361": {
            "current_exact_prior_audit": {
                "sha256": t361_current["sha256"],
                "cost": t361_current["candidate_profile"]["cost"],
                "known_four_configs": t361_current["known_four_configs"],
            },
            "retained_frontier": [
                row for row in frontier["lead_rows"] if row.get("task") == 361
            ],
        },
    }


def main() -> int:
    for directory in ("baseline", "candidates", "candidate", "audit"):
        (HERE / directory).mkdir(parents=True, exist_ok=True)

    archive_sha_before = sha256(AUTHORITY.read_bytes())
    if archive_sha_before != CAPTURED_ARCHIVE_SHA256:
        raise RuntimeError(f"authority archive drift before run: {archive_sha_before}")
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task, data in payloads.items():
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} authority member drift")
        (HERE / "baseline" / f"task{task:03d}.onnx").write_bytes(data)

    references = verify_references()
    (HERE / "audit/reference_audit.json").write_text(json.dumps(references, indent=2) + "\n")
    history = historical_evidence()
    (HERE / "audit/history_evidence.json").write_text(json.dumps(history, indent=2) + "\n")

    baselines: dict[str, Any] = {}
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        cost = audit_cost(path)
        if cost["cost"] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {cost}")
        full = AUDIT.audit(f"baseline_task{task:03d}", task, path)
        runtime_ok, runtime_summary = truthful_runtime(full)
        onnx_fresh = fresh_onnx(task, path.read_bytes())
        baselines[str(task)] = {
            "task": task,
            "path": relative(path),
            "sha256": sha256(path.read_bytes()),
            "file_bytes": path.stat().st_size,
            "cost": cost,
            "structure": structure(onnx.load(path)),
            "full_audit": full,
            "truthful_runtime": runtime_summary,
            "fresh_onnx": onnx_fresh,
            "sound_incumbent": runtime_ok and onnx_fresh["pass"],
        }
        print(
            f"BASE task{task:03d} cost={cost['cost']} truthful={runtime_ok} fresh={onnx_fresh['pass']}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for task in TASKS:
        base = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        expected_known = sum(
            len(scoring.load_examples(task).get(split, []))
            for split in ("train", "test", "arc-gen")
        )
        for kind in KINDS:
            candidate, actions = EXACT.transform(base, kind)
            data = candidate.SerializeToString()
            digest = sha256(data)
            if digest == MEMBER_SHA256[task] or (task, digest) in seen:
                continue
            seen.add((task, digest))
            path = HERE / "candidates" / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
            path.write_bytes(data)
            row: dict[str, Any] = {
                "task": task,
                "kind": kind,
                "path": relative(path),
                "sha256": digest,
                "authority_sha256": MEMBER_SHA256[task],
                "authority_cost": BASE_COSTS[task],
                "actions": actions,
                "equivalence_claim": "regenerated from current authority; transform proofs only, no past candidate reused",
                "structure": structure(candidate),
            }
            if not row["structure"].get("checker_full") or not row["structure"].get("strict_shape_data_prop"):
                row["stage"] = "REJECT_CHECKER_OR_STRICT_SHAPE"
                rows.append(row)
                continue
            try:
                cost = audit_cost(path)
            except Exception as exc:  # noqa: BLE001
                row["cost_error"] = f"{type(exc).__name__}: {exc}"
                row["stage"] = "REJECT_UNSCORABLE"
                rows.append(row)
                continue
            row["cost"] = cost
            if cost["cost"] < 0:
                row["stage"] = "REJECT_UNSCORABLE"
                rows.append(row)
                continue
            if cost["cost"] >= BASE_COSTS[task]:
                row["stage"] = "REJECT_NOT_STRICTLY_LOWER"
                rows.append(row)
                continue

            full = AUDIT.audit(f"candidate_task{task:03d}_{kind}", task, path)
            row["full_audit"] = full
            runtime_ok, runtime_summary = truthful_runtime(full)
            row["truthful_runtime"] = runtime_summary
            official = full.get("official_like_score") or {}
            disable = full.get("known_disable_all") or {}
            default = full.get("known_default") or {}
            structural_ok = (
                full.get("full_check") is True
                and full.get("strict_shape_data_prop") is True
                and not full.get("banned_ops")
                and not full.get("nonstandard_domains")
                and not full.get("functions_count", 0)
                and not full.get("sparse_initializer_count", 0)
                and not any(full.get("lookup_red_flags", {}).values())
            )
            known_ok = total_perfect(disable, expected_known) and total_perfect(default, expected_known)
            official_ok = official.get("correct") is True and official.get("cost") == cost["cost"]
            if not (structural_ok and runtime_ok and known_ok and official_ok):
                row["gate_summary"] = {
                    "structural_ok": structural_ok,
                    "runtime_truthful": runtime_ok,
                    "known_both_ort": known_ok,
                    "official_correct_cost": official_ok,
                }
                row["stage"] = "REJECT_SOUND_GATES"
                rows.append(row)
                continue

            fresh = fresh_onnx(task, data)
            row["fresh_onnx"] = fresh
            if not fresh["pass"]:
                row["stage"] = "REJECT_FRESH"
                rows.append(row)
                continue
            gain = math.log(BASE_COSTS[task] / cost["cost"])
            row["stage"] = "ADMIT"
            row["projected_gain"] = gain
            destination = HERE / "candidate" / f"task{task:03d}.onnx"
            destination.write_bytes(data)
            row["admitted_path"] = relative(destination)
            winners.append(
                {
                    "task": task,
                    "old_cost": BASE_COSTS[task],
                    "new_cost": cost["cost"],
                    "gain": gain,
                    "sha256": digest,
                    "path": relative(destination),
                    "known_count": expected_known,
                    "fresh_count": 4 * ONNX_FRESH_PER_SEED,
                    "equivalence_or_rule_basis": row["equivalence_claim"],
                }
            )
            rows.append(row)
        print(f"SCAN task{task:03d} variants={sum(row['task'] == task for row in rows)}", flush=True)

    stage_counts = dict(Counter(row["stage"] for row in rows))
    archive_sha_after = sha256(AUTHORITY.read_bytes())
    if archive_sha_after != archive_sha_before:
        raise RuntimeError(f"authority archive drift during run: {archive_sha_after}")

    results = {
        "lane": 129,
        "authority": {
            "score": 8009.46,
            "archive": relative(AUTHORITY),
            "sha256_before": archive_sha_before,
            "sha256_after": archive_sha_after,
            "member_sha256": MEMBER_SHA256,
            "costs": BASE_COSTS,
        },
        "references": references,
        "history": history,
        "baselines": baselines,
        "rows": rows,
        "variant_count": len(rows),
        "stage_counts": stage_counts,
        "winner_count": len(winners),
        "winners": winners,
    }
    (HERE / "audit/results.json").write_text(json.dumps(results, indent=2) + "\n")
    manifest = {
        "authority_archive_sha256": archive_sha_after,
        "authority_member_sha256": MEMBER_SHA256,
        "authority_costs": BASE_COSTS,
        "winner_count": len(winners),
        "winners": winners,
        "root_files_modified": [],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "variant_count": len(rows),
                "stage_counts": stage_counts,
                "winner_count": len(winners),
                "reference_pass": all(item["pass"] for item in references["tasks"].values()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
