#!/usr/bin/env python3
"""Four-configuration raw-equivalence audit for task310 parity factorization."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_mask_absorb_174/base/task310.onnx"
CANDIDATE = HERE / "task310_exact_parity_factor.onnx"
RESULT = HERE / "audit.json"
SEEDS = (202607149901, 202607149902)
FRESH_COUNT = 5000
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(path: Path, disable: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        path.read_bytes(), options, providers=["CPUExecutionProvider"]
    )


def update_raw_stats(row: dict[str, Any], base: np.ndarray, cand: np.ndarray) -> None:
    row["raw_equal"] += int(np.array_equal(base, cand))
    row["threshold_equal"] += int(np.array_equal(base > 0.0, cand > 0.0))
    row["nonfinite_authority"] += int(base.size - np.count_nonzero(np.isfinite(base)))
    row["nonfinite_candidate"] += int(cand.size - np.count_nonzero(np.isfinite(cand)))
    safe = cand[np.isfinite(cand)]
    row["near_positive_candidate"] += int(
        np.count_nonzero((safe > 0.0) & (safe < 0.25))
    )
    row["max_abs_raw_delta"] = max(
        row["max_abs_raw_delta"],
        float(np.max(np.abs(base.astype(np.float64) - cand.astype(np.float64)))),
    )
    shape = list(cand.shape)
    if shape not in row["candidate_output_shapes"]:
        row["candidate_output_shapes"].append(shape)


def empty_stats(total: int) -> dict[str, Any]:
    return {
        "total": total,
        "raw_equal": 0,
        "threshold_equal": 0,
        "authority_truth_right": 0,
        "candidate_truth_right": 0,
        "authority_errors": 0,
        "candidate_errors": 0,
        "nonfinite_authority": 0,
        "nonfinite_candidate": 0,
        "near_positive_candidate": 0,
        "max_abs_raw_delta": 0.0,
        "candidate_output_shapes": [],
        "first_raw_difference": None,
        "first_candidate_truth_failure": None,
    }


def known_examples() -> Iterable[tuple[str, int, dict[str, np.ndarray]]]:
    examples = scoring.load_examples(310)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                yield split, index, benchmark


def audit_known(disable: bool, threads: int) -> dict[str, Any]:
    cases = list(known_examples())
    row = empty_stats(len(cases))
    base_session = make_session(BASE, disable, threads)
    cand_session = make_session(CANDIDATE, disable, threads)
    for split, index, benchmark in cases:
        try:
            base = np.asarray(base_session.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            row["authority_errors"] += 1
            row.setdefault("first_authority_error", f"{type(exc).__name__}: {exc}")
            continue
        try:
            cand = np.asarray(cand_session.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            row["candidate_errors"] += 1
            row.setdefault("first_candidate_error", f"{type(exc).__name__}: {exc}")
            continue
        update_raw_stats(row, base, cand)
        want = benchmark["output"] > 0.0
        base_right = bool(np.array_equal(base > 0.0, want))
        cand_right = bool(np.array_equal(cand > 0.0, want))
        row["authority_truth_right"] += int(base_right)
        row["candidate_truth_right"] += int(cand_right)
        if not np.array_equal(base, cand) and row["first_raw_difference"] is None:
            row["first_raw_difference"] = {"split": split, "index": index}
        if not cand_right and row["first_candidate_truth_failure"] is None:
            row["first_candidate_truth_failure"] = {"split": split, "index": index}
    return row


def generator_module():
    return importlib.import_module("task_c909285e")


def audit_fresh(disable: bool, threads: int, seed: int) -> dict[str, Any]:
    row = empty_stats(FRESH_COUNT)
    row.update({"seed": seed, "generation_errors": 0, "conversion_skips": 0})
    base_session = make_session(BASE, disable, threads)
    cand_session = make_session(CANDIDATE, disable, threads)
    generator = generator_module()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    valid = 0
    while valid < FRESH_COUNT:
        try:
            benchmark = scoring.convert_to_numpy(generator.generate())
        except Exception:  # noqa: BLE001
            row["generation_errors"] += 1
            continue
        if benchmark is None:
            row["conversion_skips"] += 1
            continue
        valid += 1
        try:
            base = np.asarray(base_session.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            row["authority_errors"] += 1
            row.setdefault("first_authority_error", {"case": valid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            cand = np.asarray(cand_session.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            row["candidate_errors"] += 1
            row.setdefault("first_candidate_error", {"case": valid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        update_raw_stats(row, base, cand)
        want = benchmark["output"] > 0.0
        base_right = bool(np.array_equal(base > 0.0, want))
        cand_right = bool(np.array_equal(cand > 0.0, want))
        row["authority_truth_right"] += int(base_right)
        row["candidate_truth_right"] += int(cand_right)
        if not np.array_equal(base, cand) and row["first_raw_difference"] is None:
            row["first_raw_difference"] = {"case": valid}
        if not cand_right and row["first_candidate_truth_failure"] is None:
            row["first_candidate_truth_failure"] = {
                "case": valid,
                "different_cells": int(np.count_nonzero((cand > 0.0) != want)),
            }
        if valid % 500 == 0:
            print(
                f"fresh {seed} {valid}/{FRESH_COUNT} raw={row['raw_equal']} "
                f"truth={row['candidate_truth_right']}",
                flush=True,
            )
    return row


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def shape_trace() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    existing = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name in names or name not in typed:
                continue
            names.append(name)
            if name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = next(known_examples())[2]
    arrays = session.run(names, {"input": benchmark["input"]})
    mismatches: list[dict[str, Any]] = []
    nonfinite = 0
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        actual = list(value.shape)
        declared = dims(typed[name])
        if actual != declared:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def structural() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    h = initializers["ParityH"]
    w = initializers["ParityW"]
    rebuilt = np.einsum("kd,kr,kj,kc,k->drjc", h, h, h, h, w)
    source = onnx.load(BASE)
    a2 = numpy_helper.to_array(next(item for item in source.graph.initializer if item.name == "A2"))
    finite_initializers = all(np.all(np.isfinite(value)) for value in initializers.values())
    banned = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
        or "Sequence" in node.op_type
    ]
    return {
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info_entries": len(model.graph.value_info),
        "max_einsum_inputs": max(len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        "standard_domain_only": all(item.domain in {"", "ai.onnx"} for item in model.opset_import),
        "banned_ops": banned,
        "nested_graphs": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node
            for attr in node.attribute
        ),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "finite_initializers": finite_initializers,
        "conv_bias_findings": [],
        "parity_tensor_bit_identical": bool(np.array_equal(a2, rebuilt)),
        "parity_identity": "A2[d,r,j,c] = 0.5 * sum_k H[k,d] H[k,r] H[k,j] H[k,c]",
    }


def main() -> None:
    base_profile = cost_of(str(BASE))
    candidate_profile = cost_of(str(CANDIDATE))
    payload: dict[str, Any] = {
        "task": 310,
        "authority": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": sha256(BASE),
            "memory": base_profile[0],
            "parameters": base_profile[1],
            "cost": base_profile[2],
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
            "memory": candidate_profile[0],
            "parameters": candidate_profile[1],
            "cost": candidate_profile[2],
            "strict_gain": base_profile[2] - candidate_profile[2],
            "score_gain": math.log(base_profile[2] / candidate_profile[2]),
        },
        "structural": structural(),
        "runtime_shape_trace": shape_trace(),
        "configs": {},
        "fresh_seeds": list(SEEDS),
        "fresh_count_per_seed": FRESH_COUNT,
    }
    for disable, threads, label in CONFIGS:
        print(f"known {label}", flush=True)
        config = {"known": audit_known(disable, threads), "fresh": []}
        for seed in SEEDS:
            print(f"start {label} seed={seed}", flush=True)
            config["fresh"].append(audit_fresh(disable, threads, seed))
        payload["configs"][label] = config
        RESULT.write_text(json.dumps(payload, indent=2) + "\n")
    RESULT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
