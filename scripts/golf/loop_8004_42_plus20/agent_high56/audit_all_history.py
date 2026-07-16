#!/usr/bin/env python3
"""Fail-closed audit of every retained ONNX file for agent_high56 tasks.

The archive inventory only retains a small static-cost shortlist.  This pass walks
the complete repository history, deduplicates models by SHA-256, recomputes the
runtime-aware cost, and runs both known-data execution modes only for candidates
that appear strictly cheaper than the immutable 8005.16 baseline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
TASKS = (348, 369, 306, 106, 91, 121, 108, 265)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(data: bytes, name: str) -> dict[str, object]:
    try:
        with tempfile.TemporaryDirectory(prefix="high56_cost_", dir="/tmp") as temp:
            path = Path(temp) / name
            path.write_bytes(data)
            memory, params, cost = cost_of(str(path))
        return {"memory": memory, "params": params, "cost": cost, "error": None}
    except Exception as exc:
        return {
            "memory": None,
            "params": None,
            "cost": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def perfect(row: dict[str, object]) -> bool:
    return bool(row.get("total")) and row.get("right") == row.get("total") and row.get("errors") == 0


def belongs(path: Path, task: int) -> bool:
    text = path.as_posix().lower()
    return re.search(rf"task0*{task}(?:[^0-9]|$)", text) is not None


def main() -> None:
    all_models = tuple((ROOT / "scripts").rglob("*.onnx"))
    baselines: dict[int, dict[str, object]] = {}
    baseline_bytes: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            baseline_bytes[task] = data
            baselines[task] = {
                "path": f"{BASE.name}:{member}",
                "sha256": digest(data),
                "actual_cost": profile(data, member),
            }

    rows: list[dict[str, object]] = []
    for task in TASKS:
        base_cost = baselines[task]["actual_cost"]["cost"]  # type: ignore[index]
        seen = {digest(baseline_bytes[task])}
        paths = sorted(path for path in all_models if belongs(path, task))
        for path in paths:
            data = path.read_bytes()
            model_sha = digest(data)
            if model_sha in seen:
                continue
            seen.add(model_sha)
            cost = profile(data, path.name)
            value = cost.get("cost")
            # rank_dir uses (-1, -1, -1) when the official sanitizer/profiler
            # cannot produce a score.  Never treat that sentinel as a win.
            cheaper = (
                value is not None
                and base_cost is not None
                and int(value) > 0
                and int(value) < int(base_cost)
            )
            disable: dict[str, object] = {"not_run": "not_strictly_cheaper"}
            default: dict[str, object] = {"not_run": "not_strictly_cheaper"}
            structural: dict[str, object] | None = None
            try:
                model = onnx.load_model_from_string(data)
                if cheaper:
                    disable = run_known(copy.deepcopy(model), task, True)
                    if perfect(disable):
                        default = run_known(copy.deepcopy(model), task, False)
                    else:
                        default = {"not_run": "disable_all_not_perfect"}
                    if perfect(disable) and perfect(default):
                        structural = structure(copy.deepcopy(model), task)
            except Exception as exc:
                disable = {"error": f"{type(exc).__name__}: {exc}"}
                default = {"not_run": "model_load_or_disable_error"}

            shape_cloak = bool(
                structural
                and isinstance(structural.get("runtime_shapes"), dict)
                and structural["runtime_shapes"].get("shape_cloak")
            )
            structure_safe = bool(
                structural
                and structural.get("checker_full")
                and structural.get("strict_data_prop")
                and structural.get("static_positive")
                and structural.get("standard_domains")
                and not structural.get("banned_ops")
                and not structural.get("conv_bias_findings")
                and not structural.get("giant_einsum")
                and not structural.get("huge_fanin")
                and not structural.get("lookup_or_scatter")
                and not shape_cloak
                and not structural.get("errors")
                and isinstance(structural.get("declared_cost"), dict)
                and int(structural["declared_cost"].get("cost", -1)) > 0
            )
            row = {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": model_sha,
                "baseline_actual_cost": base_cost,
                "actual_cost": cost,
                "strictly_cheaper": cheaper,
                "known_disable_all": disable,
                "known_default": default,
                "structure": structural,
                "safe_pre_fresh": bool(cheaper and perfect(disable) and perfect(default) and structure_safe),
            }
            rows.append(row)
            if cheaper:
                print(
                    f"task{task:03d} cost={value} "
                    f"known={disable.get('right')}/{disable.get('total')} "
                    f"safe_pre_fresh={row['safe_pre_fresh']} {row['path']}",
                    flush=True,
                )

    result = {
        "baseline": {"path": BASE.name, "sha256": digest(BASE.read_bytes())},
        "tasks": list(TASKS),
        "baselines": baselines,
        "unique_candidate_count": len(rows),
        "strictly_cheaper_count": sum(bool(row["strictly_cheaper"]) for row in rows),
        "known_dual_perfect_count": sum(
            perfect(row["known_disable_all"]) and perfect(row["known_default"]) for row in rows
        ),
        "safe_pre_fresh_count": sum(bool(row["safe_pre_fresh"]) for row in rows),
        "safe_pre_fresh": [
            {key: row[key] for key in ("task", "path", "sha256", "baseline_actual_cost", "actual_cost")}
            for row in rows
            if row["safe_pre_fresh"]
        ],
        "rows": rows,
    }
    (HERE / "all_history_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
