#!/usr/bin/env python3
"""Full structural-support audit for one task013 exact-reuse candidate.

The task013 generator has 37,800 non-colour structural states.  Ordered colour
pairs add a factor of 72, but the candidate rewrite is input-independent: each
use of the constant vector ``T_zero=[1,0]`` is replaced by a contraction of
``Qor`` that is proved exactly equal to that vector.  This proves real-valued
equivalence to the immutable LB-white member for every input tensor and every
colour pair.  We then execute every structural state with canonical colours in
the four required ORT configurations to cover platform/contraction stability.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS))
from lib import scoring  # noqa: E402


CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(data: bytes, disable: bool, threads: int):
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_from_string(data)))
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def extract_qor_vector(qor: np.ndarray, subscript: str, target: str) -> np.ndarray:
    """Contract a rank-5 Qor operand down to the shared target label."""
    if len(subscript) != 5 or target not in subscript:
        raise ValueError((subscript, target))
    others = sorted(set(subscript) - {target})
    result = np.zeros(2, dtype=np.float64)
    for target_value in range(2):
        for mask in range(1 << len(others)):
            values = {target: target_value}
            values.update({name: (mask >> i) & 1 for i, name in enumerate(others)})
            index = tuple(values[name] for name in subscript)
            result[target_value] += float(qor[index])
    return result.astype(np.float32)


def exact_rewrite_proof(base: onnx.ModelProto, candidate: onnx.ModelProto) -> dict:
    base_arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    candidate_arrays = {
        item.name: numpy_helper.to_array(item) for item in candidate.graph.initializer
    }
    target = np.asarray(base_arrays["T_zero"], dtype=np.float32)
    qor = np.asarray(candidate_arrays["Qor"], dtype=np.float32)
    uses = []
    unchanged_nodes = 0
    for index, (before, after) in enumerate(zip(base.graph.node, candidate.graph.node)):
        if before.SerializeToString() == after.SerializeToString():
            unchanged_nodes += 1
            continue
        if before.op_type != "Einsum" or after.op_type != "Einsum":
            raise AssertionError(f"non-Einsum node changed at {index}")
        before_eq = next(
            helper.get_attribute_value(attr).decode()
            for attr in before.attribute
            if attr.name == "equation"
        )
        after_eq = next(
            helper.get_attribute_value(attr).decode()
            for attr in after.attribute
            if attr.name == "equation"
        )
        before_inputs = before_eq.split("->")[0].split(",")
        after_inputs = after_eq.split("->")[0].split(",")
        if len(before.input) != len(after.input) or len(before_inputs) != len(after_inputs):
            raise AssertionError(f"arity changed at {index}")
        replacements = []
        for operand, (old_name, new_name) in enumerate(zip(before.input, after.input)):
            if old_name == new_name:
                if before_inputs[operand] != after_inputs[operand]:
                    raise AssertionError(f"unrelated subscript changed at {index}:{operand}")
                continue
            if old_name != "T_zero" or new_name != "Qor":
                raise AssertionError((index, operand, old_name, new_name))
            old_sub = before_inputs[operand]
            new_sub = after_inputs[operand]
            if len(old_sub) != 1:
                raise AssertionError((index, old_sub))
            effective = extract_qor_vector(qor, new_sub, old_sub)
            replacements.append(
                {
                    "operand": operand,
                    "old_subscript": old_sub,
                    "new_subscript": new_sub,
                    "effective_vector": effective.tolist(),
                    "target_vector": target.tolist(),
                    "exact_equal": bool(np.array_equal(effective, target)),
                }
            )
        uses.append(
            {
                "node_index": index,
                "before_equation": before_eq,
                "after_equation": after_eq,
                "replacements": replacements,
            }
        )
    if not uses or not all(
        replacement["exact_equal"]
        for use in uses
        for replacement in use["replacements"]
    ):
        raise AssertionError("rewrite proof failed")
    base_other = {
        name: value.tobytes()
        for name, value in base_arrays.items()
        if name != "T_zero"
    }
    candidate_other = {name: value.tobytes() for name, value in candidate_arrays.items()}
    initializer_identity = all(
        name in candidate_other and candidate_other[name] == value
        for name, value in base_other.items()
    ) and "T_zero" not in candidate_other
    return {
        "target_vector": target.tolist(),
        "qor_finite": bool(np.isfinite(qor).all()),
        "qor_unique": np.unique(qor).tolist(),
        "changed_nodes": len(uses),
        "unchanged_nodes": unchanged_nodes,
        "other_initializers_byte_identical_and_only_tzero_removed": initializer_identity,
        "uses": uses,
        "real_semantics_equal_for_every_input_tensor": initializer_identity,
    }


def empty_stats() -> dict:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "min_positive": None,
        "max_abs_raw": 0.0,
        "first_failure": None,
    }


def update_stats(stats: dict, raw: np.ndarray, expected: np.ndarray, state: dict) -> None:
    finite = np.isfinite(raw)
    stats["nonfinite_values"] += int(raw.size - np.count_nonzero(finite))
    safe = raw[finite]
    if safe.size:
        positive = safe[safe > 0]
        stats["near_positive_values"] += int(np.count_nonzero((safe > 0) & (safe < 0.25)))
        if positive.size:
            value = float(positive.min())
            stats["min_positive"] = value if stats["min_positive"] is None else min(stats["min_positive"], value)
        stats["max_abs_raw"] = max(stats["max_abs_raw"], float(np.abs(safe).max(initial=0.0)))
    if np.array_equal(raw > 0, expected):
        stats["right"] += 1
    else:
        stats["wrong"] += 1
        if stats["first_failure"] is None:
            stats["first_failure"] = {
                **state,
                "different_bits": int(np.count_nonzero((raw > 0) != expected)),
            }


def run_known(task: int, sessions: dict) -> dict:
    examples = scoring.load_examples(task)
    result = {}
    for label, sess in sessions.items():
        stats = empty_stats()
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[split]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    stats.setdefault("skipped", 0)
                    stats["skipped"] += 1
                    continue
                try:
                    raw = sess.run(
                        [sess.get_outputs()[0].name],
                        {sess.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    update_stats(stats, raw, benchmark["output"].astype(bool), {"split": split, "index": index})
                except Exception as exc:  # noqa: BLE001
                    stats["runtime_errors"] += 1
                    if stats["first_failure"] is None:
                        stats["first_failure"] = {"split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
        stats["total"] = stats["right"] + stats["wrong"] + stats["runtime_errors"]
        stats["perfect"] = stats["wrong"] == stats["runtime_errors"] == stats["nonfinite_values"] == stats["near_positive_values"] == 0
        result[label] = stats
    return result


def structural_states():
    for width in range(20, 31):
        for height in range(6, 13):
            for start in range(1, width // 2 + 1):
                for sep in range(1, 6):
                    for bottom0 in range(2):
                        for bottom1 in range(2):
                            for xpose in range(2):
                                yield {
                                    "width": width,
                                    "height": height,
                                    "start": start,
                                    "sep": sep,
                                    "bottoms": [bottom0, bottom1],
                                    "xpose": xpose,
                                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config",
        choices=[label for _, _, label in CONFIGS],
        help="Run one ORT configuration (used to parallelize the four-config audit).",
    )
    args = parser.parse_args()
    candidate_data = args.model.read_bytes()
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        base_data = archive.read("task013.onnx")
    base = onnx.load_from_string(base_data)
    candidate = onnx.load_from_string(candidate_data)
    proof = exact_rewrite_proof(base, candidate)

    selected_configs = (
        tuple(item for item in CONFIGS if item[2] == args.config)
        if args.config
        else CONFIGS
    )
    sessions = {
        label: make_session(candidate_data, disable, threads)
        for disable, threads, label in selected_configs
    }
    known = run_known(13, sessions)
    if not all(row["perfect"] for row in known.values()):
        raise RuntimeError("known gate failed")

    generator = importlib.import_module("task_0a938d79")
    stats = {label: empty_stats() for label in sessions}
    expected_states = sum(width // 2 for width in range(20, 31)) * 7 * 5 * 4 * 2
    total = 0
    started = time.time()
    for state in structural_states():
        if args.limit is not None and total >= args.limit:
            break
        example = generator.generate(colors=[1, 2], **state)
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError({"conversion_failed": state})
        total += 1
        for label, sess in sessions.items():
            try:
                raw = sess.run(
                    [sess.get_outputs()[0].name],
                    {sess.get_inputs()[0].name: benchmark["input"]},
                )[0]
                update_stats(stats[label], raw, benchmark["output"].astype(bool), state)
            except Exception as exc:  # noqa: BLE001
                stats[label]["runtime_errors"] += 1
                if stats[label]["first_failure"] is None:
                    stats[label]["first_failure"] = {**state, "error": f"{type(exc).__name__}: {exc}"}
        if total % 5000 == 0:
            print(f"{args.model.name}: {total}/{expected_states}", flush=True)

    for row in stats.values():
        row["total"] = total
        row["perfect"] = (
            row["right"] == total
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
            and row["nonfinite_values"] == 0
            and row["near_positive_values"] == 0
        )
    arrays = [numpy_helper.to_array(item) for item in candidate.graph.initializer]
    params = int(sum(array.size for array in arrays))
    result = {
        "task": 13,
        "generator_hash": "0a938d79",
        "model": str(args.model.resolve().relative_to(ROOT)),
        "sha256": sha256(candidate_data),
        "immutable_baseline": "submission_base_8005.17.zip::task013.onnx",
        "immutable_baseline_sha256": sha256(base_data),
        "baseline_cost": 638,
        "candidate_cost": 636,
        "projected_gain": math.log(638 / 636),
        "params": params,
        "rewrite_proof": proof,
        "known_four_configs": known,
        "support": {
            "structural_state_formula": "sum(width//2,width=20..30)*7*5*4*2",
            "expected_structural_states": expected_states,
            "executed_structural_states": total,
            "ordered_distinct_colour_pairs": 72,
            "full_parameter_support_states": expected_states * 72,
            "canonical_colours": [1, 2],
            "colour_coverage_proof": "rewrite is input-independent and real-semantics equal to the immutable LB-white member for every tensor input; therefore all ordered colour pairs are covered algebraically",
            "configs": stats,
            "complete": total == expected_states and all(row["perfect"] for row in stats.values()),
        },
        "selected_configs": [label for _, _, label in selected_configs],
        "elapsed_seconds": time.time() - started,
        "accepted": total == expected_states and all(row["perfect"] for row in stats.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": args.model.name, "states": total, "accepted": result["accepted"], "elapsed": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
