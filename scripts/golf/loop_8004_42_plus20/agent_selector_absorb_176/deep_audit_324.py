#!/usr/bin/env python3
"""Fail-closed known4/fresh10000/truthful audit for task324 synth candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base" / "task324.onnx"
CANDIDATE = HERE / "candidates" / "task324_onehot_synth.onnx"
TASK = 324
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (176_324_01, 176_324_02)
FRESH_PER_SEED = 5000

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402


class GenerationTimeout(RuntimeError):
    pass


def _generation_timeout(_signum: int, _frame: Any) -> None:
    raise GenerationTimeout("generator.generate exceeded 0.25 seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def options(disable: bool, threads: int) -> ort.SessionOptions:
    value = ort.SessionOptions()
    value.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    value.intra_op_num_threads = value.inter_op_num_threads = threads
    value.log_severity_level = 4
    return value


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitizer rejected")
    return ort.InferenceSession(model.SerializeToString(), options(disable, threads), providers=["CPUExecutionProvider"])


def update_margin(row: dict[str, Any], value: np.ndarray) -> None:
    finite = np.isfinite(value)
    row["nonfinite"] += int(value.size - np.count_nonzero(finite))
    positives = value[finite & (value > 0)]
    nonpositives = value[finite & (value <= 0)]
    if positives.size:
        current = float(np.min(positives))
        row["min_positive"] = current if row["min_positive"] is None else min(row["min_positive"], current)
    if nonpositives.size:
        current = float(np.max(nonpositives))
        row["max_nonpositive"] = current if row["max_nonpositive"] is None else max(row["max_nonpositive"], current)


def known_config(source_data: bytes, candidate_data: bytes, disable: bool, threads: int) -> dict[str, Any]:
    examples = scoring.load_examples(TASK)
    ordered = [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[split])
    ]
    row: dict[str, Any] = {
        "total": len(ordered),
        "source_right": 0,
        "candidate_right": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "runtime_errors": 0,
        "nonfinite": 0,
        "min_positive": None,
        "max_nonpositive": None,
        "first_failure": None,
    }
    try:
        source_session = make_session(source_data, disable, threads)
        candidate_session = make_session(candidate_data, disable, threads)
    except Exception as exc:
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["perfect"] = False
        return row
    for split, index, example in ordered:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            row["runtime_errors"] += 1
            row["first_failure"] = row["first_failure"] or {"split": split, "index": index, "error": "conversion"}
            continue
        try:
            source = np.asarray(source_session.run(["output"], {"input": benchmark["input"]})[0])
            candidate = np.asarray(candidate_session.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:
            row["runtime_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"
            }
            continue
        expected = benchmark["output"] > 0
        source_right = np.array_equal(source > 0, expected)
        candidate_right = np.array_equal(candidate > 0, expected)
        raw_equal = np.array_equal(source, candidate)
        threshold_equal = np.array_equal(source > 0, candidate > 0)
        row["source_right"] += int(source_right)
        row["candidate_right"] += int(candidate_right)
        row["raw_equal"] += int(raw_equal)
        row["threshold_equal"] += int(threshold_equal)
        update_margin(row, candidate)
        if row["first_failure"] is None and not (source_right and candidate_right and raw_equal and threshold_equal):
            row["first_failure"] = {
                "split": split,
                "index": index,
                "source_right": bool(source_right),
                "candidate_right": bool(candidate_right),
                "raw_equal": bool(raw_equal),
                "threshold_equal": bool(threshold_equal),
            }
    total = row["total"]
    row["perfect"] = bool(
        row["source_right"] == total
        and row["candidate_right"] == total
        and row["raw_equal"] == total
        and row["threshold_equal"] == total
        and row["runtime_errors"] == 0
        and row["nonfinite"] == 0
    )
    return row


def runtime_trace(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {value.name for value in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    session = ort.InferenceSession(traced.SerializeToString(), options(True, 1), providers=["CPUExecutionProvider"])
    benchmark = scoring.convert_to_numpy(scoring.load_examples(TASK)["train"][0])
    if benchmark is None:
        raise RuntimeError("first known example cannot convert")
    arrays = session.run(names, {"input": benchmark["input"]})
    mismatches: list[dict[str, Any]] = []
    nonfinite = 0
    actual_shapes: dict[str, list[int]] = {}
    for name, value in zip(names, arrays):
        value = np.asarray(value)
        actual = list(value.shape)
        actual_shapes[name] = actual
        declared = dims(typed[name])
        if actual != declared:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "actual_shapes": actual_shapes,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def formal_proof(source_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    source = onnx.load_model_from_string(source_data)
    candidate = onnx.load_model_from_string(candidate_data)
    source_inits = {item.name: item.SerializeToString(deterministic=True) for item in source.graph.initializer}
    candidate_inits = {item.name: item.SerializeToString(deterministic=True) for item in candidate.graph.initializer}
    expected_names = set(source_inits) - {"onehot_values"}
    unchanged_inits = all(candidate_inits.get(name) == source_inits[name] for name in expected_names)
    changed_nodes = [
        index for index, (left, right) in enumerate(zip(source.graph.node, candidate.graph.node))
        if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True)
    ]
    unchanged_nodes = all(
        source.graph.node[index].SerializeToString(deterministic=True)
        == candidate.graph.node[index].SerializeToString(deterministic=True)
        for index in range(len(source.graph.node)) if index not in {10, 11, 21}
    )
    def init_array(name: str) -> np.ndarray:
        return np.asarray(onnx.numpy_helper.to_array(next(item for item in source.graph.initializer if item.name == name)))
    e = init_array("onehot_values")
    seedsel, bgsel, emap, refdiff = map(init_array, ("seedsel", "bgsel", "Emap", "refdiff"))
    derived = np.einsum("au,vw,eu,ew->a", seedsel, bgsel, emap, emap)
    pair = np.einsum("Au,vw,eu,ew,dA,dh->Ah", seedsel, bgsel, emap, emap, refdiff, refdiff)
    target_pair = np.einsum("A,h->Ah", e, e)
    remaining_refs = sum(list(node.input).count("onehot_values") for node in candidate.graph.node)
    return {
        "pass": bool(
            set(candidate_inits) == expected_names
            and unchanged_inits
            and changed_nodes == [10, 11, 21]
            and unchanged_nodes
            and np.array_equal(derived, e)
            and np.array_equal(pair, target_pair)
            and remaining_refs == 0
        ),
        "only_changed_nodes": changed_nodes,
        "removed_initializer": "onehot_values",
        "new_initializer_names": sorted(set(candidate_inits) - set(source_inits)),
        "remaining_onehot_refs": remaining_refs,
        "single_selector_identity": {"derived": derived.tolist(), "expected": e.tolist()},
        "paired_selector_identity": {"derived": pair.tolist(), "expected": target_pair.tolist()},
        "semantic_statement": (
            "Every removed e[x]=[0,1] factor is replaced by an exact contraction of unchanged "
            "seedsel/bgsel/Emap tensors. The paired final-node factors use e[A]*delta[A,h], "
            "which equals e[A]*e[h]. Therefore the tensor polynomial is identical for every input."
        ),
    }


def fresh10000(source_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    generator = importlib.import_module("task_d07ae81c")
    source_session = make_session(source_data, True, 1)
    candidate_session = make_session(candidate_data, True, 1)
    runs: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        row: dict[str, Any] = {
            "seed": seed,
            "generated": 0,
            "generation_errors": 0,
            "source_right": 0,
            "candidate_right": 0,
            "raw_equal": 0,
            "threshold_equal": 0,
            "runtime_errors": 0,
            "nonfinite": 0,
            "min_positive": None,
            "max_nonpositive": None,
            "first_failure": None,
        }
        while row["generated"] < FRESH_PER_SEED:
            try:
                previous = signal.signal(signal.SIGALRM, _generation_timeout)
                signal.setitimer(signal.ITIMER_REAL, 0.25)
                try:
                    example = generator.generate()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous)
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    raise RuntimeError("conversion")
            except Exception:
                row["generation_errors"] += 1
                continue
            index = row["generated"]
            row["generated"] += 1
            try:
                source = np.asarray(source_session.run(["output"], {"input": benchmark["input"]})[0])
                candidate = np.asarray(candidate_session.run(["output"], {"input": benchmark["input"]})[0])
            except Exception as exc:
                row["runtime_errors"] += 1
                row["first_failure"] = row["first_failure"] or {"case": index, "error": f"{type(exc).__name__}: {exc}"}
                continue
            expected = benchmark["output"] > 0
            source_right = np.array_equal(source > 0, expected)
            candidate_right = np.array_equal(candidate > 0, expected)
            raw_equal = np.array_equal(source, candidate)
            threshold_equal = np.array_equal(source > 0, candidate > 0)
            row["source_right"] += int(source_right)
            row["candidate_right"] += int(candidate_right)
            row["raw_equal"] += int(raw_equal)
            row["threshold_equal"] += int(threshold_equal)
            update_margin(row, candidate)
            if row["first_failure"] is None and not (source_right and candidate_right and raw_equal and threshold_equal):
                row["first_failure"] = {
                    "case": index,
                    "source_right": bool(source_right),
                    "candidate_right": bool(candidate_right),
                    "raw_equal": bool(raw_equal),
                    "threshold_equal": bool(threshold_equal),
                }
            if row["generated"] % 1000 == 0:
                print(f"fresh task324 seed={seed} {row['generated']}/{FRESH_PER_SEED}", flush=True)
        row["perfect"] = bool(
            row["source_right"] == FRESH_PER_SEED
            and row["candidate_right"] == FRESH_PER_SEED
            and row["raw_equal"] == FRESH_PER_SEED
            and row["threshold_equal"] == FRESH_PER_SEED
            and row["runtime_errors"] == 0
            and row["nonfinite"] == 0
        )
        runs.append(row)
    return {
        "total": FRESH_PER_SEED * len(FRESH_SEEDS),
        "mode": "ORT_DISABLE_ALL_threads1",
        "generator": "inputs/arc-gen-repo/tasks/task_d07ae81c.py",
        "runs": runs,
        "perfect": all(row["perfect"] for row in runs),
        "default_mode": "NOT_RUN_SESSION_CONSTRUCTION_FAILS_FOR_AUTHORITY_AND_CANDIDATE",
    }


def main() -> None:
    source_data = SOURCE.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    screen = json.loads((HERE / "screen_results.json").read_text())
    synth = next(row for row in screen["rows"] if row["label"] == "onehot_synth")
    known = {
        label: known_config(source_data, candidate_data, disable, threads)
        for disable, threads, label in CONFIGS
    }
    print("known4 complete", {label: row["perfect"] for label, row in known.items()}, flush=True)
    traces = {"source": runtime_trace(source_data), "candidate": runtime_trace(candidate_data)}
    proof = formal_proof(source_data, candidate_data)
    # Fresh is still run under the competition runtime to document the exact
    # rewrite fully; the final adoption remains fail-closed on known4/truthful.
    fresh = fresh10000(source_data, candidate_data)
    known4 = all(row["perfect"] for row in known.values())
    truthful = traces["candidate"]["truthful"]
    accepted = bool(
        synth["competition"]["result"]["correct"]
        and synth["strict_lower"]
        and proof["pass"]
        and known4
        and truthful
        and fresh["perfect"]
        and synth["structural_ub_no_lookup"]["pass"]
    )
    payload = {
        "task": TASK,
        "source": {"path": str(SOURCE.relative_to(REPO)), "sha256": sha256(SOURCE), "profile": screen["baselines"][str(TASK)]},
        "candidate": {
            "path": str(CANDIDATE.relative_to(REPO)),
            "sha256": sha256(CANDIDATE),
            "profile": synth["profile"]["result"],
            "competition": synth["competition"]["result"],
            "projected_gain": math.log(screen["baselines"][str(TASK)]["cost"] / synth["profile"]["result"]["cost"]),
        },
        "formal_all_input_proof": proof,
        "known_four_configs": known,
        "known4_pass": known4,
        "runtime_shape_trace": traces,
        "truthful_pass": truthful,
        "fresh10000": fresh,
        "nonfinite_pass": all(row.get("nonfinite", 1) == 0 for row in known.values() if "session_error" not in row) and all(row["nonfinite"] == 0 for row in fresh["runs"]),
        "structural_ub_no_lookup": synth["structural_ub_no_lookup"],
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECT_INHERITED_UB_SHAPE_CLOAK_DEFAULT_FAILURE_NONFINITE",
        "reasons": [] if accepted else [
            "authority_and_candidate_default_ORT_TopK_shape_failure",
            "declared_runtime_shape_mismatch_inherited_from_authority",
            "authority_and_candidate_emit_nonfinite_values_despite_threshold_correctness",
        ],
    }
    out = HERE / "deep_audit_324.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cost": [payload["source"]["profile"]["cost"], payload["candidate"]["profile"]["cost"]],
        "formal": proof["pass"],
        "known4": known4,
        "truthful": truthful,
        "fresh10000": fresh["perfect"],
        "accepted": accepted,
        "result": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
