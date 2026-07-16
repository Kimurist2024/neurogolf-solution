#!/usr/bin/env python3
"""Search scale-free task188 tie predicates while preserving exact dimension logic."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
OUTPUT = HERE / "task188_scale_free_search.json"
CANDIDATES = HERE / "candidates"


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("cost297_task188_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error", "session_error",
        "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total") and row.get("wrong") == 0
        and row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0 and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and not row.get("session_error")
    )


def candidate(authority: onnx.ModelProto, fifth: str | None, compare: str) -> onnx.ModelProto:
    model = copy.deepcopy(authority)
    hs = model.graph.node[2]
    base_inputs = ["input"] * 4
    equation = "ncha,nchb,ndka,ndkb"
    if fifth is not None:
        base_inputs.append("input")
        equation += ",n" + fifth
    equation += "->n"
    del hs.input[:]
    hs.input.extend(base_inputs)
    for attribute in hs.attribute:
        if attribute.name == "equation":
            attribute.s = equation.encode("ascii")
    less = model.graph.node[3]
    less.input[1] = compare
    # scale becomes dead after removing it from the Einsum.
    model = onnxoptimizer.optimize(model, ["eliminate_deadend", "eliminate_unused_initializer"])
    return model


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = onnx.load_from_string(archive.read("task188.onnx"))
    cases, known_counts = SUPPORT.known_cases(188)
    # Channel labels may only share channel axes; spatial axes all have extent 30.
    channels = ("c", "d", "e")
    spatial = ("h", "k", "a", "b", "r", "s")
    fifths: list[str | None] = [None]
    fifths.extend(c + r + s for c in channels for r in spatial for s in spatial)
    rows: list[dict[str, Any]] = []
    exact_known: list[tuple[onnx.ModelProto, dict[str, Any]]] = []
    for fifth in fifths:
        for compare in ("col_pairs", "area"):
            model = candidate(authority, fifth, compare)
            data = model.SerializeToString()
            label = "none" if fifth is None else fifth
            row: dict[str, Any] = {
                "fifth": fifth,
                "compare": compare,
                "equation": next(a.s.decode() for a in model.graph.node[2].attribute if a.name == "equation"),
                "sha256": sha256(data),
            }
            try:
                runtime = SUPPORT.make_session(data, True, 1)
                baseline, _ = SUPPORT.evaluate_config(runtime, cases, None)
            except Exception as exc:  # noqa: BLE001
                baseline = {"total": len(cases), "right": 0, "wrong": 0, "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}"}
            row["known_disable_threads1"] = compact(baseline)
            if exact(baseline):
                profile = SUPPORT.official_profile(188, model, f"task188_sf_{label}_{compare}")
                structure = SUPPORT.structural_audit(188, model, data)
                row["official_profile"] = profile
                row["structure"] = structure
                if structure["pass"] and profile and int(profile["cost"]) < 46:
                    exact_known.append((model, row))
            rows.append(row)
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    accepted: list[dict[str, Any]] = []
    for model, row in exact_known:
        data = model.SerializeToString()
        known_four = SUPPORT.evaluate_four(data, cases)
        row["known_four"] = {name: compact(value) for name, value in known_four.items()}
        if not all(exact(value) for value in known_four.values()):
            continue
        fresh_rows = []
        for seed in (297_188_001, 297_188_002):
            SUPPORT.FRESH_PER_SEED = 2_000
            fresh_cases, generation = SUPPORT.fresh_cases(188, seed, task_map)
            four = SUPPORT.evaluate_four(data, fresh_cases)
            fresh_rows.append({
                "seed": seed,
                "generation": generation,
                "four": {name: compact(value) for name, value in four.items()},
                "exact": all(exact(value) for value in four.values()),
            })
        row["fresh"] = fresh_rows
        if all(value["exact"] for value in fresh_rows):
            path = CANDIDATES / f"task188_scale_free_{row['fifth'] or 'none'}_{row['compare']}_cost{row['official_profile']['cost']}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            row["candidate_path"] = str(path.relative_to(ROOT))
            accepted.append(row)
    result = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": AUTHORITY_SHA256, "lb": 8011.05, "task_cost": 46},
        "known_counts": known_counts,
        "attempts": len(rows),
        "known_exact_lower": len(exact_known),
        "accepted": accepted,
        "best_known": sorted(rows, key=lambda x: x["known_disable_threads1"].get("right", 0), reverse=True)[:20],
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "attempts": len(rows),
        "known_exact_lower": len(exact_known),
        "accepted": [row["candidate_path"] for row in accepted],
        "best": [(row["fifth"], row["compare"], row["known_disable_threads1"].get("right")) for row in result["best_known"][:10]],
        "elapsed_seconds": result["elapsed_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
