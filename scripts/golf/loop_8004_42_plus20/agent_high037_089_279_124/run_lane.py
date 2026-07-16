#!/usr/bin/env python3
"""Fail-closed SOUND/memshave lane for tasks 037, 089, and 279."""

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
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
CAPTURED_ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (37, 89, 279)
HASHES = {37: "1f876c06", 89: "3e980e27", 279: "b2862040"}
MEMBER_SHA256 = {
    37: "df9298f3b9e851bd815be5f53b8de142cbf944297477d73fc27ddc343d49c90b",
    89: "89183f12515ceee79eb73d69ce074a6a93145930e5fb9eb426d3a1f7f58e5607",
    279: "d3bb22792a3e44e09d21971f88642622a32d161f3a7888f9d0e9efe5862d0a9b",
}
BASE_COSTS = {37: 320, 89: 1340, 279: 397}
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
REFERENCE_SEEDS = (124_000_000, 124_100_000)
REFERENCE_FRESH_PER_SEED = 1500

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
    "lane124_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
AUDIT = load_module(
    "lane124_audit",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
RANK = load_module("lane124_rank", ROOT / "scripts/golf/rank_dir.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def normalize(grid: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid]


def solve_037(grid: list[list[int]]) -> list[list[int]]:
    """Connect the two equal-colored endpoints along their diagonal."""
    output = [[0 for _ in row] for row in grid]
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            if value:
                by_color[value].append((row, col))
    for color, points in by_color.items():
        if len(points) != 2:
            raise ValueError(f"task037 color {color} has {len(points)} endpoints")
        (r0, c0), (r1, c1) = sorted(points)
        dr, dc = r1 - r0, c1 - c0
        if dr <= 0 or abs(dc) != dr:
            raise ValueError("task037 endpoints are not a supported diagonal")
        step = 1 if dc > 0 else -1
        for offset in range(dr + 1):
            output[r0 + offset][c0 + step * offset] = color
    return output


def components8(grid: list[list[int]]) -> list[set[tuple[int, int]]]:
    height, width = len(grid), len(grid[0])
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c]}
    result = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            row, col = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not dr and not dc:
                        continue
                    point = (row + dr, col + dc)
                    if point in unseen:
                        unseen.remove(point)
                        component.add(point)
                        queue.append(point)
        result.append(component)
    return result


def solve_089(grid: list[list[int]]) -> list[list[int]]:
    """Learn each visible sprite and stamp it at its marker-only copies."""
    output = copy.deepcopy(grid)
    components = components8(grid)
    for marker in (2, 3):
        marker_points = [
            (r, c)
            for r, row in enumerate(grid)
            for c, value in enumerate(row)
            if value == marker
        ]
        if not marker_points:
            continue
        full = next(
            component
            for component in components
            if len(component) > 1
            and any(grid[r][c] == marker for r, c in component)
        )
        anchor = next((point for point in full if grid[point[0]][point[1]] == marker))
        relative = [
            (row - anchor[0], col - anchor[1], grid[row][col])
            for row, col in full
        ]
        for target in marker_points:
            if target == anchor:
                continue
            for dr, dc, color in relative:
                if marker == 2:
                    dc = -dc
                output[target[0] + dr][target[1] + dc] = color
    return output


def solve_279(grid: list[list[int]]) -> list[list[int]]:
    """Recolor four-connected blue components exactly when they have a cycle."""
    height, width = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == 1}
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = deque([seed])
        degree_sum = 0
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                    continue
                if grid[neighbor[0]][neighbor[1]] != 1:
                    continue
                degree_sum += 1
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        if degree_sum // 2 >= len(component):
            for row, col in component:
                output[row][col] = 8
    return output


SOLVERS: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    37: solve_037,
    89: solve_089,
    279: solve_279,
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
        for seed in REFERENCE_SEEDS:
            random.seed(seed + task)
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
                    "seed": seed + task,
                    "right": right,
                    "total": REFERENCE_FRESH_PER_SEED,
                    "errors": errors,
                    "first_failure": first_failure,
                }
            )
        result["tasks"][str(task)] = {
            "hash": HASHES[task],
            "known": {"right": known_right, "total": len(known_examples)},
            "fresh": streams,
            "pass": known_right == len(known_examples)
            and all(
                stream["right"] == REFERENCE_FRESH_PER_SEED and stream["errors"] == 0
                for stream in streams
            ),
        }
        print(f"REF task{task:03d} known={known_right}/{len(known_examples)}", flush=True)
    return result


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
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
        "input_shape": shape(model.graph.input[0]),
        "output_shape": shape(model.graph.output[0]),
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
        record["strict_shape"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(strict_shape=False, strict_shape_error=f"{type(exc).__name__}: {exc}")
    return record


def audit_cost(path: Path) -> dict[str, int]:
    memory, params, cost = RANK.cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


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


def fresh_candidate(task: int, data: bytes, count: int = 2000) -> dict[str, Any]:
    generator = importlib.import_module(f"task_{HASHES[task]}")
    active = {"disable_all": make_session(data, True), "default": make_session(data, False)}
    runs = []
    for seed in (124_200_000 + task, 124_300_000 + task):
        random.seed(seed)
        stats = {
            key: {"right": 0, "wrong": 0, "errors": 0, "near_positive": 0}
            for key in active
        }
        valid = 0
        while valid < count:
            benchmark = scoring.convert_to_numpy(generator.generate())
            if benchmark is None:
                continue
            valid += 1
            expected = benchmark["output"] > 0
            for key, session in active.items():
                try:
                    raw = np.asarray(
                        session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    stats[key]["near_positive"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                    if np.array_equal(raw > 0, expected):
                        stats[key]["right"] += 1
                    else:
                        stats[key]["wrong"] += 1
                except Exception:  # noqa: BLE001
                    stats[key]["errors"] += 1
        runs.append({"seed": seed, "valid": valid, "modes": stats})
    passed = all(
        mode["right"] == count
        and mode["wrong"] == 0
        and mode["errors"] == 0
        and mode["near_positive"] == 0
        for run in runs
        for mode in run["modes"].values()
    )
    return {"count_per_seed": count, "runs": runs, "pass": passed}


def main() -> int:
    for directory in ("baseline", "candidates", "candidate", "audit"):
        (HERE / directory).mkdir(parents=True, exist_ok=True)
    archive_sha = sha256(AUTHORITY.read_bytes())
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task, data in payloads.items():
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} authority member drift")
        (HERE / "baseline" / f"task{task:03d}.onnx").write_bytes(data)

    references = verify_references()
    (HERE / "audit/reference_audit.json").write_text(json.dumps(references, indent=2) + "\n")

    baselines: dict[str, Any] = {}
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        cost = audit_cost(path)
        if cost["cost"] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {cost}")
        full = AUDIT.audit(f"baseline_task{task:03d}", task, path)
        baselines[str(task)] = {
            "task": task,
            "path": relative(path),
            "sha256": sha256(path.read_bytes()),
            "file_bytes": path.stat().st_size,
            "cost": cost,
            "structure": structure(model),
            "full_audit": full,
        }
        trace = full.get("runtime_shape_trace") or {}
        print(
            f"BASE task{task:03d} cost={cost['cost']} mismatches={len(trace.get('declared_actual_mismatches', []))}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for task in TASKS:
        base = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
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
                "equivalence_basis": sorted(
                    {
                        item.get("proof", key)
                        for key, value in actions.items()
                        if isinstance(value, list)
                        for item in value
                        if isinstance(item, dict)
                    }
                ),
                "structure": structure(candidate),
            }
            if not row["structure"].get("checker_full") or not row["structure"].get("strict_shape"):
                row["stage"] = "REJECT_CHECKER_OR_STRICT_SHAPE"
                rows.append(row)
                continue
            row["cost"] = audit_cost(path)
            if row["cost"]["cost"] < 0 or row["cost"]["cost"] >= BASE_COSTS[task]:
                row["stage"] = "REJECT_NOT_STRICTLY_LOWER"
                rows.append(row)
                continue
            full = AUDIT.audit(f"task{task:03d}_{kind}", task, path)
            row["full_audit"] = full
            official = full.get("official_like_score") or {}
            disabled = (full.get("known_disable_all") or {}).get("total", {})
            default = (full.get("known_default") or {}).get("total", {})
            trace = full.get("runtime_shape_trace") or {}
            known_ok = all(
                item.get("right", 0) > 0
                and item.get("wrong", 0) == 0
                and item.get("errors", 0) == 0
                for item in (disabled, default)
            )
            mismatches = trace.get("declared_actual_mismatches")
            truthful = isinstance(mismatches, list) and not mismatches and not trace.get("error")
            if (
                not official
                or not official.get("correct")
                or int(official.get("cost", BASE_COSTS[task])) >= BASE_COSTS[task]
            ):
                row["stage"] = "REJECT_OFFICIAL_NOT_CORRECT_LOWER"
            elif not known_ok:
                row["stage"] = "REJECT_DUAL_ORT_KNOWN"
            elif not truthful:
                row["stage"] = "REJECT_UNTRUTHFUL_RUNTIME_SHAPES"
            else:
                row["fresh"] = fresh_candidate(task, data)
                row["stage"] = "SAFE_EXACT_WINNER" if row["fresh"]["pass"] else "REJECT_FRESH"
            rows.append(row)
        print(f"SCAN task{task:03d} variants={sum(row['task'] == task for row in rows)}", flush=True)

    winners = [row for row in rows if row["stage"] == "SAFE_EXACT_WINNER"]
    result = {
        "lane": "agent_high037_089_279_124",
        "authority": {
            "path": "submission.zip",
            "captured_archive_sha256": CAPTURED_ARCHIVE_SHA256,
            "observed_archive_sha256": archive_sha,
            "archive_matches_capture": archive_sha == CAPTURED_ARCHIVE_SHA256,
            "member_hashes_match": True,
        },
        "references": references,
        "baselines": baselines,
        "variant_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "winner_count": len(winners),
        "winners": winners,
        "rows": rows,
    }
    (HERE / "audit/results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "authority_member_sha256": {str(key): value for key, value in MEMBER_SHA256.items()},
        "authority_costs": {str(key): value for key, value in BASE_COSTS.items()},
        "winner_count": len(winners),
        "winners": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["sha256"],
                "cost": row["full_audit"]["official_like_score"]["cost"],
                "fresh": row["fresh"],
            }
            for row in winners
        ],
        "root_files_modified": [],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "variant_count": len(rows),
                "stage_counts": result["stage_counts"],
                "winner_count": len(winners),
                "reference_pass": all(item["pass"] for item in references["tasks"].values()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
