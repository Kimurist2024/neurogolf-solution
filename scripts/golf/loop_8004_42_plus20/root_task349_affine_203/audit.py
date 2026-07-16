#!/usr/bin/env python3
"""Four-configuration differential audit for the exact task349 affine table shave."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others" / "71407" / "task349.onnx"
CANDIDATE = HERE / "task349_affine_max29.onnx"
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((203_349_11, 2500), (203_349_29, 2500))


def load_shared():
    spec = importlib.util.spec_from_file_location("task349_shared_203", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load shared audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    result: dict[str, object] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes": len(model.graph.node),
        "params": int(sum(np.asarray(onnx.numpy_helper.to_array(x)).size for x in model.graph.initializer)),
        "conv_family_nodes": sum(n.op_type in {"Conv", "ConvTranspose", "QLinearConv"} for n in model.graph.node),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:
        result["full_check"] = False
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"
    for data_prop in (False, True):
        key = "strict_data_prop" if data_prop else "strict"
        try:
            onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=data_prop)
            result[key] = True
        except Exception as exc:
            result[key] = False
            result[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    shared = load_shared()
    source = SOURCE.read_bytes()
    candidate = CANDIDATE.read_bytes()
    src_model = onnx.load(SOURCE)
    cand_model = onnx.load(CANDIDATE)
    src_init = {x.name: onnx.numpy_helper.to_array(x) for x in src_model.graph.initializer}
    cand_init = {x.name: onnx.numpy_helper.to_array(x) for x in cand_model.graph.initializer}
    table_identity = np.array_equal(
        src_init["hstart_offset_by_mod_i8"].astype(np.int16),
        1 - 3 * src_init["hend_offset_by_mod_i8"].astype(np.int16),
    )
    report: dict[str, object] = {
        "source_structure": structure(SOURCE),
        "candidate_structure": structure(CANDIDATE),
        "table_identity_all_11": bool(table_identity),
        "candidate_removed_hstart_table": "hstart_offset_by_mod_i8" not in cand_init,
        "candidate_removed_max30": "max30_i8" not in cand_init,
        "candidate_max29_greater_outputs": sorted(
            node.output[0]
            for node in cand_model.graph.node
            if node.op_type == "Greater"
            and list(node.input)[-1:] == ["max29_i8"]
            and node.output
        ),
        "candidate_runtime_shape_truth": shared.runtime_shape_truth(349, candidate),
        "known_four_configs": {},
        "fresh": [],
    }
    cases = shared.known(349)
    for disable, threads, label in CONFIGS:
        report["known_four_configs"][label] = shared.evaluate_cases(
            source, candidate, cases, disable, threads
        )
    for seed, count in FRESH:
        fresh, attempts = shared.generate(349, seed, count)
        stream = {"seed": seed, "requested": count, "attempts": attempts, "modes": {}}
        for disable, threads, label in CONFIGS:
            stream["modes"][label] = shared.evaluate_cases(
                source, candidate, fresh, disable, threads
            )
        report["fresh"].append(stream)
        print("fresh", seed, len(fresh), flush=True)
    comparisons = list(report["known_four_configs"].values()) + [
        row for stream in report["fresh"] for row in stream["modes"].values()
    ]
    report["all_raw_equivalent"] = all(row.get("exact_equivalent") for row in comparisons)
    report["runtime_errors_total"] = sum(row.get("runtime_errors_total", 0) for row in comparisons)
    report["candidate_nonfinite_total"] = sum(
        row.get("nonfinite_values", {}).get("candidate", 0) for row in comparisons
    )
    report["pass"] = bool(
        table_identity
        and report["candidate_removed_hstart_table"]
        and report["candidate_removed_max30"]
        and report["candidate_max29_greater_outputs"] == ["beam_end_is30", "halo_end_is30"]
        and report["candidate_structure"].get("full_check")
        and report["candidate_structure"].get("strict")
        and report["candidate_runtime_shape_truth"].get("truthful")
        and report["candidate_structure"].get("conv_family_nodes") == 0
        and report["all_raw_equivalent"]
        and report["runtime_errors_total"] == 0
        and report["candidate_nonfinite_total"] == 0
    )
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", report["pass"])


if __name__ == "__main__":
    main()
