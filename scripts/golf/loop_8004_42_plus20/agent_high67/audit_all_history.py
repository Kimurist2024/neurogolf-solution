#!/usr/bin/env python3
"""Exhaustive SHA-deduplicated loose-history audit for high67 targets.

Every local ONNX path carrying one of the explicit task tokens is inventoried.
Cost and correctness are evaluated once per (task, SHA256).  Strictly-lower
models fail closed on non-finite payloads, shortened Conv bias, lookup/scatter,
shape cloaking, giant fan-in/Einsum, or any official structural gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
TASKS = (71, 137, 183, 210, 234, 241, 267, 308)
BASE_SHA256 = "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00"
TASK_TOKEN = re.compile(r"task0*(\d+)(?!\d)", re.IGNORECASE)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.audit_retained_group import (  # noqa: E402
    perfect,
    profile,
)
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identify(path: Path) -> int | None:
    matches = [int(item.group(1)) for item in TASK_TOKEN.finditer(path.as_posix())]
    # The right-most token denotes the model itself.  Do not inherit a target
    # number merely from an ancestor scratch directory containing other tasks.
    return matches[-1] if matches and matches[-1] in TASKS else None


def finite_initializers(model: onnx.ModelProto) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for tensor in model.graph.initializer:
        try:
            array = numpy_helper.to_array(tensor)
            if np.issubdtype(array.dtype, np.number) and not bool(np.all(np.isfinite(array))):
                findings.append({"name": tensor.name, "dtype": str(array.dtype)})
        except Exception as exc:
            findings.append({
                "name": tensor.name,
                "decode_error": f"{type(exc).__name__}: {exc}",
            })
    return {"all_finite": not findings, "findings": findings}


def structure_safe(row: dict[str, object]) -> bool:
    runtime = row.get("runtime_shapes")
    declared = row.get("declared_cost")
    return bool(
        row.get("checker_full")
        and row.get("strict_data_prop")
        and row.get("static_positive")
        and row.get("standard_domains")
        and not row.get("banned_ops")
        and not row.get("conv_bias_findings")
        and not row.get("giant_einsum")
        and not row.get("huge_fanin")
        and not row.get("lookup_or_scatter")
        and isinstance(runtime, dict)
        and not runtime.get("shape_cloak")
        and not runtime.get("mismatch_count")
        and not row.get("errors")
        and isinstance(declared, dict)
        and int(declared.get("cost", -1)) > 0
    )


def main() -> None:
    if digest(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("immutable baseline SHA-256 mismatch")

    baselines: dict[int, dict[str, object]] = {}
    baseline_sha: dict[int, str] = {}
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            actual = profile(data, member)
            baselines[task] = {
                "path": f"{BASE.name}:{member}",
                "sha256": digest(data),
                "actual_cost": actual,
            }
            baseline_sha[task] = digest(data)

    observations = {task: 0 for task in TASKS}
    unique: dict[int, dict[str, dict[str, object]]] = {task: {} for task in TASKS}
    for path in sorted(ROOT.rglob("*.onnx")):
        task = identify(path)
        if task is None or HERE in path.parents:
            continue
        observations[task] += 1
        try:
            data = path.read_bytes()
        except OSError:
            continue
        model_sha = digest(data)
        relative = str(path.relative_to(ROOT))
        prior = unique[task].get(model_sha)
        if prior is not None:
            prior["source_count"] = int(prior["source_count"]) + 1
            sources = prior["sources"]
            assert isinstance(sources, list)
            if len(sources) < 30:
                sources.append(relative)
            continue
        unique[task][model_sha] = {
            "task": task,
            "sha256": model_sha,
            "sources": [relative],
            "source_count": 1,
            "data": data,
        }

    rows: list[dict[str, object]] = []
    for task in TASKS:
        raw_base = baselines[task]["actual_cost"]
        assert isinstance(raw_base, dict)
        base_cost = int(raw_base["cost"])
        for model_sha, item in sorted(unique[task].items()):
            data = item.pop("data")
            assert isinstance(data, bytes)
            row = dict(item)
            actual = profile(data, f"task{task:03d}_{model_sha[:10]}.onnx")
            value = actual.get("cost")
            lower = (
                base_cost > 0
                and value is not None
                and int(value) > 0
                and int(value) < base_cost
            )
            row.update(
                same_as_baseline=model_sha == baseline_sha[task],
                baseline_actual_cost=base_cost,
                actual_cost=actual,
                strictly_cheaper=lower,
                rejection_reasons=[],
                safe_pre_fresh=False,
            )
            if lower:
                try:
                    model = onnx.load_model_from_string(data)
                    finite = finite_initializers(model)
                    structural = structure(copy.deepcopy(model), task)
                    row["finite_initializers"] = finite
                    row["structure"] = structural
                    immediate: list[str] = []
                    if not finite["all_finite"]:
                        immediate.append("nonfinite_initializer")
                    if structural.get("conv_bias_findings"):
                        immediate.append("shortened_conv_bias")
                    if structural.get("giant_einsum"):
                        immediate.append("giant_einsum")
                    if structural.get("huge_fanin"):
                        immediate.append("huge_fanin")
                    if structural.get("lookup_or_scatter"):
                        immediate.append("lookup_or_scatter")
                    if immediate:
                        row["rejection_reasons"] = immediate
                        row["known_disable_all"] = {"not_run": "immediate_policy_reject"}
                        row["known_default"] = {"not_run": "immediate_policy_reject"}
                    else:
                        disable = run_known(copy.deepcopy(model), task, True)
                        default = (
                            run_known(copy.deepcopy(model), task, False)
                            if perfect(disable)
                            else {"not_run": "disable_all_not_perfect"}
                        )
                        row["known_disable_all"] = disable
                        row["known_default"] = default
                        reasons: list[str] = []
                        if not perfect(disable):
                            reasons.append("known_disable_all_not_100")
                        if not perfect(default):
                            reasons.append("known_default_not_100")
                        if perfect(disable) and perfect(default) and not structure_safe(structural):
                            reasons.append("structural_policy_gate")
                        row["rejection_reasons"] = reasons
                    row["safe_pre_fresh"] = not row["rejection_reasons"]
                    print(
                        f"LOW task{task:03d} cost={value}/{base_cost} "
                        f"safe={row['safe_pre_fresh']} reasons={row['rejection_reasons']} "
                        f"sha={model_sha[:12]}",
                        flush=True,
                    )
                except Exception as exc:
                    row["audit_error"] = f"{type(exc).__name__}: {exc}"
                    row["rejection_reasons"] = ["audit_error"]
                    row["safe_pre_fresh"] = False
            rows.append(row)

    output = {
        "baseline": {"path": BASE.name, "sha256": digest(BASE.read_bytes())},
        "tasks": list(TASKS),
        "baselines": baselines,
        "summary": [
            {
                "task": task,
                "observed_paths": observations[task],
                "unique_sha": len(unique[task]),
                "nonbaseline_sha": sum(
                    not bool(row.get("same_as_baseline"))
                    for row in rows if row["task"] == task
                ),
                "strictly_cheaper": sum(
                    bool(row.get("strictly_cheaper"))
                    for row in rows if row["task"] == task
                ),
                "safe_pre_fresh": sum(
                    bool(row.get("safe_pre_fresh"))
                    for row in rows if row["task"] == task
                ),
            }
            for task in TASKS
        ],
        "unique_task_model_pairs": len(rows),
        "path_aliases": sum(observations.values()),
        "strictly_cheaper_count": sum(bool(row.get("strictly_cheaper")) for row in rows),
        "safe_pre_fresh_count": sum(bool(row.get("safe_pre_fresh")) for row in rows),
        "safe_pre_fresh": [
            {
                key: row[key]
                for key in ("task", "sha256", "sources", "baseline_actual_cost", "actual_cost")
            }
            for row in rows if row.get("safe_pre_fresh")
        ],
        "rows": rows,
    }
    (HERE / "all_history_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
