#!/usr/bin/env python3
"""Execute task333's complete trilinear coefficient support in one ORT mode.

For generator-valid inputs the second input occurrence is algebraically filtered
to the four green box cells.  With the box fixed, the one-node network is
linear in the source-pixel occurrence and pointwise linear in the current-cell
occurrence.  Therefore box(36) x source-basis(10*10*10) x current-colour(10)
= 360,000 states are the complete residual coefficient support.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx"
CONFIGS = {
    "disable_all_threads1": (True, 1),
    "disable_all_threads4": (True, 4),
    "default_threads1": (False, 1),
    "default_threads4": (False, 4),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_model(data: bytes, batch: int) -> bytes:
    model = copy.deepcopy(onnx.load_from_string(data))
    original = copy.deepcopy(model.graph.input[0])
    del model.graph.input[:]
    names = ("input_A", "input_B", "input_C")
    for name in names:
        value = copy.deepcopy(original)
        value.name = name
        value.type.tensor_type.shape.dim[0].dim_value = batch
        model.graph.input.append(value)
    occurrence = 0
    for index, name in enumerate(model.graph.node[0].input):
        if name != original.name:
            continue
        model.graph.node[0].input[index] = names[occurrence]
        occurrence += 1
    assert occurrence == 3
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = batch
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def make_session(data: bytes, batch: int, disable: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(split_model(data, batch), options, providers=["CPUExecutionProvider"])


def states():
    for boxrow in range(2, 8):
        for boxcol in range(2, 8):
            for source_colour in range(10):
                for source_row in range(10):
                    for source_col in range(10):
                        for current_colour in range(10):
                            yield boxrow, boxcol, source_colour, source_row, source_col, current_colour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    disable, threads = CONFIGS[args.config]
    candidate_data = CANDIDATE.read_bytes()
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        baseline_data = archive.read("task333.onnx")
    candidate_session = make_session(candidate_data, args.batch, disable, threads)
    baseline_session = make_session(baseline_data, args.batch, disable, threads)

    total_expected = 36 * 10 * 10 * 10 * 10
    total = batches = runtime_errors = nonfinite_values = 0
    different_values = sign_differences = 0
    max_abs_difference = 0.0
    first_difference = None
    started = time.time()
    iterator = iter(states())
    while total < total_expected and (args.limit is None or total < args.limit):
        wanted = min(args.batch, total_expected - total)
        if args.limit is not None:
            wanted = min(wanted, args.limit - total)
        batch_states = []
        for _ in range(wanted):
            try:
                batch_states.append(next(iterator))
            except StopIteration:
                break
        if not batch_states:
            break
        input_a = np.zeros((args.batch, 10, 30, 30), dtype=np.float32)
        input_b = np.zeros_like(input_a)
        input_c = np.zeros_like(input_a)
        for row, state in enumerate(batch_states):
            boxrow, boxcol, source_colour, source_row, source_col, current_colour = state
            input_a[row, source_colour, source_row, source_col] = 1.0
            input_b[row, 3, boxrow : boxrow + 2, boxcol : boxcol + 2] = 1.0
            input_c[row, current_colour, :10, :10] = 1.0
        if len(batch_states) < args.batch:
            input_a[len(batch_states) :] = input_a[0]
            input_b[len(batch_states) :] = input_b[0]
            input_c[len(batch_states) :] = input_c[0]
        feed = {"input_A": input_a, "input_B": input_b, "input_C": input_c}
        try:
            candidate_raw = candidate_session.run(["output"], feed)[0][: len(batch_states)]
            baseline_raw = baseline_session.run(["output"], feed)[0][: len(batch_states)]
        except Exception as exc:  # noqa: BLE001
            runtime_errors += len(batch_states)
            if first_difference is None:
                first_difference = {"state": batch_states[0], "error": f"{type(exc).__name__}: {exc}"}
            total += len(batch_states)
            continue
        finite = np.isfinite(candidate_raw) & np.isfinite(baseline_raw)
        nonfinite_values += int(finite.size - np.count_nonzero(finite))
        unequal = candidate_raw != baseline_raw
        different_values += int(np.count_nonzero(unequal))
        sign_differences += int(np.count_nonzero((candidate_raw > 0) != (baseline_raw > 0)))
        if finite.any():
            max_abs_difference = max(max_abs_difference, float(np.abs(candidate_raw[finite] - baseline_raw[finite]).max(initial=0.0)))
        if first_difference is None and unequal.any():
            batch_index, channel, out_row, out_col = np.argwhere(unequal)[0]
            first_difference = {
                "state": batch_states[int(batch_index)],
                "output": [int(channel), int(out_row), int(out_col)],
                "baseline": float(baseline_raw[batch_index, channel, out_row, out_col]),
                "candidate": float(candidate_raw[batch_index, channel, out_row, out_col]),
            }
        total += len(batch_states)
        batches += 1
        if total % (args.batch * 50) == 0:
            print(args.config, total, total_expected, "diff", different_values, "sign", sign_differences, flush=True)

    complete = total == total_expected
    result = {
        "task": 333,
        "config": args.config,
        "batch_size": args.batch,
        "baseline_sha256": digest(baseline_data),
        "candidate_sha256": digest(candidate_data),
        "support_formula": "boxrow6*boxcol6*source_colour10*source_row10*source_col10*current_colour10",
        "expected_states": total_expected,
        "executed_states": total,
        "batches": batches,
        "runtime_errors": runtime_errors,
        "nonfinite_values": nonfinite_values,
        "different_values": different_values,
        "sign_differences": sign_differences,
        "max_abs_difference": max_abs_difference,
        "first_difference": first_difference,
        "complete": complete,
        "perfect_platform_sign_equivalence": complete and runtime_errors == nonfinite_values == sign_differences == 0,
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
