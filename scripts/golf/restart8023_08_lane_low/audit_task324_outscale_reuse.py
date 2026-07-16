#!/usr/bin/env python3
"""Fail-closed gold/fresh audit for the exact task324 outscale reuse."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8023.08.zip"
CANDIDATE = HERE / "candidates" / "task324_outscale_reuse.onnx"
AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
CANDIDATE_SHA256 = "701cc1a9a6ecbde54f8b7cd9b65d000e89c665fc615db9c20430afd1c3588de0"
SEEDS = (824_324_001, 824_324_002)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive",
        "maximum_nonpositive", "sign_mismatch_cases_vs_disable_threads1",
        "sign_mismatch_cells_vs_disable_threads1", "sign_sha256", "raw_sha256",
        "optimization", "threads", "first_wrong", "first_error",
        "first_shape_mismatch", "first_sign_mismatch", "elapsed_seconds",
    )
    return {key: row.get(key) for key in keys if key in row}


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("total", 0) > 0
        and row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    candidate = CANDIDATE.read_bytes()
    if sha256(candidate) != CANDIDATE_SHA256:
        raise RuntimeError("candidate SHA drift")
    support = import_path(
        "restart8023_task324_support",
        ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py",
    )
    support.POLICY_THRESHOLD = 1.0
    support.FRESH_PER_SEED = 2_000
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    model = onnx.load_model_from_string(candidate)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [
            *inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output
        ]
    }
    nonstatic = []
    for node in inferred.graph.node:
        for name in node.output:
            if not name:
                continue
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                nonstatic.append(name)
                continue
            dims = value.type.tensor_type.shape.dim
            if not dims or any(
                dim.HasField("dim_param")
                or not dim.HasField("dim_value")
                or int(dim.dim_value) <= 0
                for dim in dims
            ):
                nonstatic.append(name)
    known_cases, known_counts = support.known_cases(324)
    known_raw = support.evaluate_four(candidate, known_cases)
    known = {name: compact(row) for name, row in known_raw.items()}
    fresh = []
    for seed in SEEDS:
        cases, generation = support.fresh_cases(324, seed, task_map)
        rows = support.evaluate_four(candidate, cases)
        compacted = {name: compact(row) for name, row in rows.items()}
        fresh.append(
            {
                "seed": seed,
                "generation": generation,
                "runtime": compacted,
                "pass": len(cases) == 2_000 and all(exact(row) for row in rows.values()),
            }
        )
        print(json.dumps({"seed": seed, "pass": fresh[-1]["pass"],
                          "runtime": compacted}), flush=True)
    official = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
            "--task", "324", "--onnx", str(CANDIDATE), "--timeout", "90",
            "--label", "restart8023_task324_final",
        ], cwd=ROOT, capture_output=True, text=True,
    )
    official_line = official.stdout.strip().splitlines()[-1] if official.stdout.strip() else ""
    try:
        official_result = json.loads(official_line)
    except json.JSONDecodeError:
        official_result = {"ok": False, "unparseable": official_line}
    with zipfile.ZipFile(AUTHORITY) as archive:
        source = archive.read("task324.onnx")
    result = {
        "task": 324,
        "authority_sha256": AUTHORITY_SHA256,
        "source_member_sha256": sha256(source),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": CANDIDATE_SHA256,
        "full_check": True,
        "strict_data_prop": True,
        "all_node_outputs_static": not nonstatic,
        "nonstatic_outputs": nonstatic,
        "known_counts": known_counts,
        "known": known,
        "known_exact": all(exact(row) for row in known_raw.values()),
        "fresh": fresh,
        "fresh_exact": all(row["pass"] for row in fresh),
        "official_gold": {
            "returncode": official.returncode,
            "result": official_result,
            "tail": (official.stdout + official.stderr)[-4000:],
        },
        "authority_cost": 425,
        "candidate_cost": int(official_result.get("cost", -1)),
        "gain": math.log(425 / int(official_result["cost"]))
        if int(official_result.get("cost", -1)) > 0 else None,
        "inherited_authority_output_declaration": [1, 1, 30, 30],
        "canonical_runtime_output": [1, 10, 30, 30],
        "strict_promotion_pass": bool(
            not nonstatic
            and all(exact(row) for row in known_raw.values())
            and all(row["pass"] for row in fresh)
            and official.returncode == 0
            and official_result.get("ok") is True
            and official_result.get("correct") is True
            and int(official_result.get("cost", -1)) == 423
        ),
    }
    (HERE / "task324_outscale_reuse_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "task", "known_exact", "fresh_exact", "official_gold",
        "candidate_cost", "gain", "strict_promotion_pass"
    )}, indent=2), flush=True)
    return 0 if result["strict_promotion_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
