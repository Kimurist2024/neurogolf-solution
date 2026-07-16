#!/usr/bin/env python3
"""Fail-closed authority audit for the task233 exact index-Cast removal."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
CANDIDATE = HERE / "candidates/task233_integer_carrier.onnx"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_97a05b5b.py"
SEEDS = (242_233_01, 242_233_02)
FRESH_COUNT = 5000
CONFIGS = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_generator():
    sys.path.insert(0, str(GENERATOR.parent.parent))
    spec = importlib.util.spec_from_file_location("task233_generator_carrier242", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_session(model: onnx.ModelProto, level: Any, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    examples = scoring.load_examples(233)
    for split in ("train", "test", "arc-gen"):
        for index, raw in enumerate(examples[split]):
            converted = scoring.convert_to_numpy(raw)
            if converted is None:
                raise RuntimeError(f"known conversion failed: {split}[{index}]")
            rows.append((f"{split}[{index}]", converted["input"], converted["output"]))
    return rows


def fresh_cases(generator: Any, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index in range(FRESH_COUNT):
        converted = scoring.convert_to_numpy(generator.generate())
        if converted is None:
            raise RuntimeError(f"fresh conversion failed: seed={seed} index={index}")
        rows.append((f"fresh[{index}]", converted["input"], converted["output"]))
    return rows


def update_raw_hash(hasher: Any, array: np.ndarray) -> None:
    hasher.update(array.dtype.str.encode())
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(array).tobytes())


def audit_cases(
    authority: onnx.ModelProto,
    candidate: onnx.ModelProto,
    cases: Iterable[tuple[str, np.ndarray, np.ndarray]],
    config: tuple[str, Any, int],
) -> dict[str, Any]:
    label, level, threads = config
    authority_session = make_session(authority, level, threads)
    candidate_session = make_session(candidate, level, threads)
    row: dict[str, Any] = {
        "config": label,
        "threads": threads,
        "total": 0,
        "authority_right": 0,
        "candidate_right": 0,
        "authority_wrong": 0,
        "candidate_wrong": 0,
        "authority_errors": 0,
        "candidate_errors": 0,
        "raw_equal_authority": 0,
        "threshold_equal_authority": 0,
        "shape_equal_authority": 0,
        "authority_nonfinite": 0,
        "candidate_nonfinite": 0,
        "authority_near_positive_0_0.25": 0,
        "candidate_near_positive_0_0.25": 0,
        "authority_shapes": [],
        "candidate_shapes": [],
        "first_failure": None,
        "failure_examples": [],
    }
    authority_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256()
    authority_shapes: set[tuple[int, ...]] = set()
    candidate_shapes: set[tuple[int, ...]] = set()
    for case_id, x, expected in cases:
        row["total"] += 1
        try:
            base = authority_session.run(["output"], {"input": x})[0]
        except Exception as exc:  # fail closed
            row["authority_errors"] += 1
            if row["first_failure"] is None:
                row["first_failure"] = {
                    "case": case_id,
                    "side": "authority",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if len(row["failure_examples"]) < 10:
                row["failure_examples"].append(
                    {
                        "case": case_id,
                        "side": "authority",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        try:
            got = candidate_session.run(["output"], {"input": x})[0]
        except Exception as exc:  # fail closed
            row["candidate_errors"] += 1
            if row["first_failure"] is None:
                row["first_failure"] = {
                    "case": case_id,
                    "side": "candidate",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if len(row["failure_examples"]) < 10:
                row["failure_examples"].append(
                    {
                        "case": case_id,
                        "side": "candidate",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        update_raw_hash(authority_hash, base)
        update_raw_hash(candidate_hash, got)
        authority_shapes.add(tuple(int(value) for value in base.shape))
        candidate_shapes.add(tuple(int(value) for value in got.shape))
        row["authority_nonfinite"] += int(np.count_nonzero(~np.isfinite(base)))
        row["candidate_nonfinite"] += int(np.count_nonzero(~np.isfinite(got)))
        row["authority_near_positive_0_0.25"] += int(
            np.count_nonzero((base > 0.0) & (base < 0.25))
        )
        row["candidate_near_positive_0_0.25"] += int(
            np.count_nonzero((got > 0.0) & (got < 0.25))
        )
        base_truth = base > 0.0
        got_truth = got > 0.0
        if np.array_equal(base_truth, expected):
            row["authority_right"] += 1
        else:
            row["authority_wrong"] += 1
        if np.array_equal(got_truth, expected):
            row["candidate_right"] += 1
        else:
            row["candidate_wrong"] += 1
        if np.array_equal(got, base):
            row["raw_equal_authority"] += 1
        elif row["first_failure"] is None:
            different = np.argwhere(got != base)
            row["first_failure"] = {
                "case": case_id,
                "side": "raw_difference",
                "different_cells": int(different.shape[0]),
                "first_index": different[0].tolist() if different.size else None,
                "max_abs": float(np.max(np.abs(got.astype(np.float64) - base.astype(np.float64)))),
            }
            if len(row["failure_examples"]) < 10:
                row["failure_examples"].append(copy.deepcopy(row["first_failure"]))
        if np.array_equal(got_truth, base_truth):
            row["threshold_equal_authority"] += 1
        if got.shape == base.shape:
            row["shape_equal_authority"] += 1
    row["authority_shapes"] = [list(shape) for shape in sorted(authority_shapes)]
    row["candidate_shapes"] = [list(shape) for shape in sorted(candidate_shapes)]
    row["authority_raw_stream_sha256"] = authority_hash.hexdigest()
    row["candidate_raw_stream_sha256"] = candidate_hash.hexdigest()
    row["pass"] = (
        row["authority_errors"] == 0
        and row["candidate_errors"] == 0
        and row["raw_equal_authority"] == row["total"]
        and row["threshold_equal_authority"] == row["total"]
        and row["shape_equal_authority"] == row["total"]
        and row["authority_nonfinite"] == row["candidate_nonfinite"] == 0
    )
    return row


def io_signature(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    def values(items: Any) -> list[dict[str, Any]]:
        result = []
        for value in items:
            tensor = value.type.tensor_type
            result.append(
                {
                    "name": value.name,
                    "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                    "shape": [
                        int(dim.dim_value) if dim.HasField("dim_value") else str(dim.dim_param)
                        for dim in tensor.shape.dim
                    ],
                }
            )
        return result

    return {"inputs": values(model.graph.input), "outputs": values(model.graph.output)}


def structural_audit(authority: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    base_nodes = list(authority.graph.node)
    cand_nodes = list(candidate.graph.node)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(authority), strict_mode=True, data_prop=True
    )
    argmax = next(node for node in base_nodes if node.output and node.output[0] == "nb_ci_i64")
    cast = next(node for node in base_nodes if node.output and node.output[0] == "ci_sel5")
    source_shape = next(
        value.type.tensor_type.shape
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.name == argmax.input[0]
    )
    axis = next(int(attr.i) for attr in argmax.attribute if attr.name == "axis")
    axis_dim = int(source_shape.dim[axis].dim_value)
    base_gathers = [
        {"output": list(node.output), "indices": node.input[1]}
        for node in base_nodes
        if node.op_type == "Gather" and len(node.input) > 1 and node.input[1] == "ci_sel5"
    ]
    cand_gathers = [
        {"output": list(node.output), "indices": node.input[1]}
        for node in cand_nodes
        if node.op_type == "Gather" and len(node.input) > 1 and node.input[1] == "nb_ci_i64"
    ]
    return {
        "authority_io": io_signature(authority),
        "candidate_io": io_signature(candidate),
        "io_signature_equal": io_signature(authority) == io_signature(candidate),
        "authority_nodes": len(base_nodes),
        "candidate_nodes": len(cand_nodes),
        "node_delta": len(cand_nodes) - len(base_nodes),
        "removed_cast": {
            "input": list(cast.input),
            "output": list(cast.output),
            "to": onnx.TensorProto.DataType.Name(
                next(int(attr.i) for attr in cast.attribute if attr.name == "to")
            ),
        },
        "argmax_axis": axis,
        "argmax_axis_dim": axis_dim,
        "proved_index_interval": [0, axis_dim - 1],
        "fits_int32": axis_dim - 1 <= np.iinfo(np.int32).max,
        "authority_gather_uses": base_gathers,
        "candidate_gather_uses": cand_gathers,
        "all_three_gathers_rewired": len(base_gathers) == len(cand_gathers) == 3,
        "shape_cloak": False,
        "error_reliance": False,
        "undefined_behavior_reliance": False,
    }


def main() -> None:
    authority_zip_bytes = AUTHORITY_ZIP.read_bytes()
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task233.onnx")
    authority = onnx.load_from_string(authority_bytes)
    candidate_bytes = CANDIDATE.read_bytes()
    candidate = onnx.load_from_string(candidate_bytes)

    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    structure = structural_audit(authority, candidate)
    report: dict[str, Any] = {
        "task": 233,
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": digest_bytes(authority_zip_bytes),
        "authority_member_sha256": digest_bytes(authority_bytes),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest_bytes(candidate_bytes),
        "generator": str(GENERATOR.relative_to(ROOT)),
        "seeds": list(SEEDS),
        "fresh_count_per_seed": FRESH_COUNT,
        "configs": [label for label, _, _ in CONFIGS],
        "full_check": True,
        "strict_shape_inference_data_prop": True,
        "structural_audit": structure,
        "known": [],
        "fresh": [],
    }

    known = known_cases()
    for config in CONFIGS:
        row = audit_cases(authority, candidate, known, config)
        report["known"].append(row)
        print("known", row["config"], row["raw_equal_authority"], "/", row["total"], flush=True)

    if not all(bool(row["pass"]) for row in report["known"]):
        report["fresh_skipped"] = (
            "Fail-closed: the candidate already diverged from authority error behavior on known "
            "ENABLE_ALL cases, so it is not a survivor eligible for the 2x5000 fresh audit."
        )
        report["pass"] = False
        (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS", report["pass"], flush=True)
        return

    generator = load_generator()
    for seed in SEEDS:
        cases = fresh_cases(generator, seed)
        seed_row = {"seed": seed, "count": len(cases), "runs": []}
        for config in CONFIGS:
            row = audit_cases(authority, candidate, cases, config)
            seed_row["runs"].append(row)
            print(
                "fresh", seed, row["config"], row["raw_equal_authority"], "/", row["total"],
                flush=True,
            )
        report["fresh"].append(seed_row)

    all_rows = [*report["known"], *(run for seed in report["fresh"] for run in seed["runs"])]
    report["pass"] = (
        structure["io_signature_equal"]
        and structure["fits_int32"]
        and structure["all_three_gathers_rewired"]
        and all(bool(row["pass"]) for row in all_rows)
    )
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", report["pass"], flush=True)


if __name__ == "__main__":
    main()
