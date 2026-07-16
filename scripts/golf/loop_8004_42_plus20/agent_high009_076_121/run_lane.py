#!/usr/bin/env python3
"""Fail-closed exact/memshave scan for authority task009 and task076.

This lane never edits the root submission, ledgers, artifacts, or others/71407.
It snapshots only the two assigned members, applies all-input-preserving graph
rewrites, and advances strictly cheaper candidates through the full audit.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
CAPTURED_ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (9, 76)
MEMBER_SHA256 = {
    9: "372fef762ffbc873f8c6ef0f3e2f59478773e17702f4129d5e7e9ce8c783bfaa",
    76: "9d31114f8af80bf54b6c908ad61eadd6dbe4fb63f52b5b97ecb70f1f0fcce791",
}
BASE_COSTS = {9: 2619, 76: 2550}
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
    "lane121_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
AUDIT = load_module(
    "lane121_audit",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
RANK = load_module("lane121_rank", ROOT / "scripts/golf/rank_dir.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def static_memory_breakdown(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    io = {value.name for value in list(inferred.graph.input) + list(inferred.graph.output)}
    inits = {item.name for item in inferred.graph.initializer}
    values = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    rows = []
    groups: dict[str, int] = defaultdict(int)
    for name, value in values.items():
        if name in io or name in inits or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        dims = shape(value)
        if not all(isinstance(item, int) and item > 0 for item in dims):
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        byte_count = int(math.prod(dims) * np.dtype(dtype).itemsize)
        prefix = name.split("_")[0]
        if name.startswith("xf"):
            prefix = "xf_decode_float"
        elif name.startswith("x") and len(name) > 1 and name[1].isdigit():
            prefix = "x_decode_u8"
        elif name.startswith("z") and len(name) > 1 and name[1].isdigit():
            prefix = "z_blank_bool"
        elif name.startswith("row_cell"):
            prefix = "row_cell"
        elif name.startswith("row_line"):
            prefix = "row_line"
        elif name.startswith("base_label"):
            prefix = "base_label"
        groups[prefix] += byte_count
        rows.append(
            {
                "tensor": name,
                "bytes": byte_count,
                "dtype": str(np.dtype(dtype)),
                "shape": dims,
            }
        )
    rows.sort(key=lambda item: (-item["bytes"], item["tensor"]))
    return {
        "static_total": sum(item["bytes"] for item in rows),
        "tensor_count": len(rows),
        "groups": dict(sorted(groups.items(), key=lambda item: (-item[1], item[0]))),
        "top_tensors": rows[:50],
    }


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


def audit_cost(path: Path) -> dict[str, int | None]:
    memory, params, cost = RANK.cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def session(data: bytes, disable_all: bool) -> ort.InferenceSession:
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


def fresh(task: int, data: bytes, count: int = 2000) -> dict[str, Any]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    seeds = (121_000_000 + task, 121_100_000 + task)
    sessions = {
        "disable_all": session(data, True),
        "default": session(data, False),
    }
    runs = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {
            key: {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
            for key in sessions
        }
        valid = attempts = generation_errors = 0
        while valid < count:
            attempts += 1
            try:
                benchmark = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if benchmark is None:
                continue
            valid += 1
            expected = benchmark["output"] > 0
            for key, active in sessions.items():
                try:
                    raw = active.run(
                        [active.get_outputs()[0].name],
                        {active.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    if np.array_equal(raw > 0, expected):
                        stats[key]["right"] += 1
                    else:
                        stats[key]["wrong"] += 1
                        if stats[key]["first_failure"] is None:
                            stats[key]["first_failure"] = {
                                "case": valid,
                                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                            }
                except Exception as exc:  # noqa: BLE001
                    stats[key]["errors"] += 1
                    if stats[key]["first_failure"] is None:
                        stats[key]["first_failure"] = {
                            "case": valid,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        runs.append(
            {
                "seed": seed,
                "valid": valid,
                "attempts": attempts,
                "generation_errors": generation_errors,
                "modes": stats,
            }
        )
    passed = all(
        mode["right"] == count and mode["wrong"] == 0 and mode["errors"] == 0
        for run in runs
        for mode in run["modes"].values()
    )
    return {"task": task, "count_per_seed": count, "runs": runs, "pass": passed}


def noninjective_task076() -> dict[str, Any]:
    generator = importlib.import_module("task_36d67576")
    params = {
        "width": 14,
        "height": 15,
        "rows": [2, 2, 1, 0, 1, 2, 0, 0, 3, 1],
        "cols": [0, 1, 2, 2, 1, 2, 1, 3, 1, 0],
        "colors": [4, 4, 4, 4, 4, 2, 1, 3, 3, 3],
        "megarows": [1, 1, 8],
        "megacols": [1, 7, 2],
    }
    left = generator.generate(**params, megarotates=[0, 3, 3])
    right = generator.generate(**params, megarotates=[0, 1, 1])
    encode = lambda grid: json.dumps(grid, separators=(",", ":")).encode()
    return {
        "valid_generator_parameterizations": True,
        "same_input": left["input"] == right["input"],
        "same_output": left["output"] == right["output"],
        "input_sha256": sha256(encode(left["input"])),
        "output_a_sha256": sha256(encode(left["output"])),
        "output_b_sha256": sha256(encode(right["output"])),
        "megarotates_a": [0, 3, 3],
        "megarotates_b": [0, 1, 1],
        "different_output_cells": sum(
            a != b
            for row_a, row_b in zip(left["output"], right["output"])
            for a, b in zip(row_a, row_b)
        ),
        "conclusion": "no deterministic input-only ONNX can be exact on the full generator relation",
    }


def main() -> int:
    (HERE / "baseline").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)
    archive_sha = sha256(AUTHORITY.read_bytes())
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task, data in payloads.items():
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} authority member drift")
        (HERE / "baseline" / f"task{task:03d}.onnx").write_bytes(data)

    baselines: dict[str, Any] = {}
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        cost = audit_cost(path)
        if cost["cost"] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {cost}")
        baseline = {
            "task": task,
            "path": relative(path),
            "sha256": sha256(path.read_bytes()),
            "file_bytes": path.stat().st_size,
            "cost": cost,
            "structure": structure(model),
            "static_memory_breakdown": static_memory_breakdown(model),
        }
        try:
            baseline["runtime_shape_trace"] = AUDIT.runtime_shape_trace(task, model)
        except Exception as exc:  # noqa: BLE001
            baseline["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
        baseline["full_audit"] = AUDIT.audit(f"baseline_task{task:03d}", task, path)
        if task == 9:
            baseline["fresh"] = fresh(task, path.read_bytes())
        baselines[str(task)] = baseline
        print(f"BASE task{task:03d} cost={cost['cost']}", flush=True)

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
                "all_input_equivalence_basis": sorted(
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
            row["full_audit"] = AUDIT.audit(f"task{task:03d}_{kind}", task, path)
            official = row["full_audit"].get("official_like_score") or {}
            disabled = (row["full_audit"].get("known_disable_all") or {}).get("total", {})
            default = (row["full_audit"].get("known_default") or {}).get("total", {})
            shape_trace = row["full_audit"].get("runtime_shape_trace") or {}
            known_ok = all(
                item.get("right", 0) > 0
                and item.get("wrong", 0) == 0
                and item.get("errors", 0) == 0
                for item in (disabled, default)
            )
            truthful = not shape_trace.get("error") and not shape_trace.get(
                "declared_actual_mismatches", ["missing"]
            )
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
                row["fresh"] = fresh(task, data)
                row["stage"] = (
                    "SAFE_EXACT_WINNER" if row["fresh"]["pass"] else "REJECT_FRESH"
                )
            rows.append(row)
        print(
            f"SCAN task{task:03d} variants={sum(row['task'] == task for row in rows)}",
            flush=True,
        )

    proof = noninjective_task076()
    winners = [row for row in rows if row["stage"] == "SAFE_EXACT_WINNER"]
    result = {
        "lane": "agent_high009_076_121",
        "authority": {
            "path": "submission.zip",
            "captured_archive_sha256": CAPTURED_ARCHIVE_SHA256,
            "observed_archive_sha256": archive_sha,
            "archive_matches_capture": archive_sha == CAPTURED_ARCHIVE_SHA256,
            "member_hashes_match": True,
        },
        "baselines": baselines,
        "task076_noninjectivity_proof": proof,
        "variant_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "winner_count": len(winners),
        "winners": winners,
        "rows": rows,
    }
    (HERE / "audit/results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "authority_member_sha256": MEMBER_SHA256,
        "authority_costs": BASE_COSTS,
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
                "task076_noninjective": proof["same_input"] and not proof["same_output"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
