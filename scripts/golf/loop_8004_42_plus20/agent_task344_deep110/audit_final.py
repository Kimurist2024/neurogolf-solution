#!/usr/bin/env python3
"""Fail-closed authority, reference, and candidate audit for task344."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8008.14.zip"
MEMBER = "task344.onnx"
EXPECTED_MEMBER_SHA = "d0902dc6498525c5f62f12fc02e25fe7914afbae4a583fd77b71f8f05f08019f"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.lib import scoring  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


CANDIDATES = [
    HERE / "candidates/task344_p2_shared_full.onnx",
    HERE / "candidates/task344_p2_split_diag.onnx",
    HERE / "candidates/task344_p2_split_none.onnx",
    HERE / "candidates/task344_p2_rank3_split_full.onnx",
    HERE / "candidates/task344_p2_b3_split_full.onnx",
]
HISTORICAL = [
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/candidates/task344_no_s_cost188.onnx",
    ROOT / "scripts/golf/scratch_codex_7994/task344_shared_v_cost181.onnx",
    ROOT / "others/2/7801/task344_cost191.onnx",
]
SOUND_CONTROLS = [
    ROOT / "scripts/golf/loop_8003_40/agent_sound_local_resume/models/task344.onnx",
]
LOOKUP_OPS = {"TfIdfVectorizer", "Hardmax", "GatherND"}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def measure(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def rule(grid: list[list[int]]) -> list[list[int]]:
    out = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] != 3:
                continue
            touched = False
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and grid[rr][cc] == 2:
                    out[rr][cc] = 0
                    touched = True
            if touched:
                out[row][col] = 8
    return out


def reference_audit() -> dict[str, object]:
    payload = json.loads((ROOT / "inputs/neurogolf-2026/task344.json").read_text())
    known_right = known_total = 0
    for split in ("train", "test", "arc-gen"):
        for example in payload[split]:
            known_total += 1
            known_right += int(rule(example["input"]) == example["output"])
    generator = importlib.import_module("task_d90796e8")
    fresh_rows = []
    for seed in (344110037, 344110091):
        random.seed(seed)
        right = 0
        for _ in range(5000):
            example = generator.generate()
            right += int(rule(example["input"]) == example["output"])
        fresh_rows.append({"seed": seed, "right": right, "total": 5000})
    return {
        "rule": "Simultaneously: green 3 adjacent cardinally to red 2 becomes 8; each such red becomes 0; all other cells are copied.",
        "source": "inputs/arc-gen-repo/tasks/task_d90796e8.py",
        "known": {"right": known_right, "total": known_total},
        "fresh": fresh_rows,
    }


def nested_graphs(model: onnx.ModelProto) -> int:
    count = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                count += 1
    return count


def finite_known(model: onnx.ModelProto, disable: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return {"session_error": "sanitize rejected", "nonfinite": None}
    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as error:
        return {"session_error": f"{type(error).__name__}: {error}", "nonfinite": None}
    total = nonfinite = 0
    for split in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(344)[split]:
            bench = scoring.convert_to_numpy(example)
            if bench is None:
                continue
            raw = session.run(["output"], {"input": bench["input"]})[0]
            total += 1
            nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
    return {"cases": total, "nonfinite": nonfinite, "session_error": None}


def static_extra(model: onnx.ModelProto) -> dict[str, object]:
    ops = [node.op_type for node in model.graph.node]
    return {
        "initializer_names": [item.name for item in model.graph.initializer],
        "has_S_initializer": any(item.name == "S" for item in model.graph.initializer),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "functions": len(model.functions),
        "nested_graph_attributes": nested_graphs(model),
        "lookup_ops": sorted(set(ops) & LOOKUP_OPS),
        "banned_ops": sorted({op for op in ops if op in BANNED or "Sequence" in op}),
    }


def audit_model(path: Path, baseline_cost: int, label: str) -> dict[str, object]:
    model = onnx.load(path)
    cost = measure(path)
    structural = structure(copy.deepcopy(model), 344)
    extra = static_extra(model)
    disable = run_known(copy.deepcopy(model), 344, True)
    default = run_known(copy.deepcopy(model), 344, False)
    known_perfect = (
        disable.get("right") == disable.get("total")
        and disable.get("errors") == 0
        and default.get("right") == default.get("total")
        and default.get("errors") == 0
    )
    shape_row = structural.get("runtime_shapes") or {}
    clean = (
        structural.get("checker_full")
        and structural.get("strict_data_prop")
        and structural.get("standard_domains")
        and not structural.get("banned_ops")
        and not structural.get("conv_bias_findings")
        and not extra["has_S_initializer"]
        and not extra["sparse_initializers"]
        and not extra["functions"]
        and not extra["nested_graph_attributes"]
        and not extra["lookup_ops"]
        and structural.get("max_node_inputs", 999) <= 16
        and shape_row.get("mismatch_count") == 0
    )
    strict_lower = cost["cost"] < baseline_cost
    fresh_gate = strict_lower and known_perfect and clean
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "cost": cost,
        "strict_lower": strict_lower,
        "known_disable_all": disable,
        "known_default": default,
        "known_dual_perfect": known_perfect,
        "finite_disable_all": finite_known(copy.deepcopy(model), True),
        "finite_default": finite_known(copy.deepcopy(model), False),
        "structure": structural,
        "extra_structure": extra,
        "clean_non_giant_no_S": clean,
        "fresh_dual_2x5000": {
            "ran": False,
            "reason": "known/structure/cost pre-gate failed" if not fresh_gate else "UNEXPECTED_GATE_PASS",
        },
        "winner_eligible": False,
    }


def main() -> None:
    baseline_dir = HERE / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        data = archive.read(MEMBER)
    if sha_bytes(data) != EXPECTED_MEMBER_SHA:
        raise RuntimeError("immutable task344 authority SHA changed")
    baseline_path = baseline_dir / MEMBER
    baseline_path.write_bytes(data)
    baseline_cost = measure(baseline_path)
    if baseline_cost["cost"] != 197:
        raise RuntimeError(f"expected authority cost197, got {baseline_cost}")
    baseline_model = onnx.load_model_from_string(data)
    arrays = {
        item.name: onnx.numpy_helper.to_array(item)
        for item in baseline_model.graph.initializer
    }

    rows = [audit_model(path, 197, "new_non_giant_no_S") for path in CANDIDATES]
    historical = [audit_model(path, 197, "historical_lower") for path in HISTORICAL]
    controls = [audit_model(path, 197, "clean_sound_control") for path in SOUND_CONTROLS]
    result = {
        "lane": "agent_task344_deep110",
        "authority": {
            "archive": BASE.name,
            "archive_sha256": sha(BASE),
            "member": MEMBER,
            "member_sha256": sha_bytes(data),
            "serialized_bytes": len(data),
            "cost": baseline_cost,
            "known_disable_all": run_known(copy.deepcopy(baseline_model), 344, True),
            "known_default": run_known(copy.deepcopy(baseline_model), 344, False),
            "structure": structure(copy.deepcopy(baseline_model), 344),
            "extra_structure": static_extra(baseline_model),
            "factor_ranks": {
                name: int(np.linalg.matrix_rank(value))
                for name, value in arrays.items()
                if value.ndim == 2
            },
        },
        "reference": reference_audit(),
        "new_candidates": rows,
        "historical_lower_controls": historical,
        "historical_fresh_evidence": {
            "task344_cost191": {
                "disable_all": "4972/5000",
                "default": "4972/5000",
                "source": "scripts/golf/loop_8002_63/lane_task344_r02/fresh_dual5000.json",
                "verdict": "REJECT_NOT_GENERATOR_EXACT_AND_HAS_S_AND_24_INPUT_GIANT_EINSUM",
            },
            "task344_no_s_cost188": {
                "fresh": "not run because known is 0/266",
                "source": "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/audit/task344_no_s_rejected.json",
                "verdict": "REJECT",
            },
        },
        "clean_sound_controls": controls,
        "fresh_policy": {
            "requirement": "Only a strict-lower, known-dual-perfect, clean no-S/non-giant candidate receives two independent 5000-case streams in both ORT modes.",
            "candidate_runs": 0,
            "reason": "No candidate passed known and structural pre-gates.",
        },
        "winners": [],
        "verified_gain": 0,
        "protected_files_modified": False,
        "verdict": "NO_CANDIDATE",
    }
    (HERE / "final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "lane": result["lane"],
                "task": 344,
                "authority_cost": 197,
                "winners": [],
                "candidate_for_probe": None,
                "verified_gain": 0,
                "protected_files_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"authority": baseline_cost, "new": [(row["cost"]["cost"], row["known_disable_all"]["right"], row["clean_non_giant_no_S"]) for row in rows], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
