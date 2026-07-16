#!/usr/bin/env python3
"""Dual-ORT, known, and two-seed fresh proof for the task233 exact alias."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import runpy
import sys
from pathlib import Path
import zipfile

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402

TASK = 233
COUNT = 5000
SEEDS = (714_233_101, 714_233_202)
CANDIDATE = HERE / "task233_exact_alias.onnx"
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
REPORT = HERE / "verification.json"


def make_session(model: onnx.ModelProto, disable: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def run(session: ort.InferenceSession, array: np.ndarray) -> np.ndarray:
    return session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: array}
    )[0]


def stats() -> dict[str, object]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "decoded_equal_base": 0,
        "raw_bitwise_equal_base": 0,
        "first_failure": None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        base_bytes = archive.read("task233.onnx")
    base_model = onnx.load_from_string(base_bytes)
    candidate_model = onnx.load(CANDIDATE)
    modes = ((True, "disable_all"), (False, "default"))
    base_sessions = {name: make_session(base_model, disable) for disable, name in modes}
    cand_sessions = {name: make_session(candidate_model, disable) for disable, name in modes}

    output: dict[str, object] = {
        "task": TASK,
        "base_zip_sha256": hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest(),
        "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "proof": "remove a TensorProto-identical scalar initializer and repoint its two consumers",
        "known": {},
        "fresh": {},
    }

    known = json.loads((ROOT / f"inputs/neurogolf-2026/task{TASK:03d}.json").read_text())
    known_rows = [row for split in ("train", "test", "arc-gen") for row in known.get(split, [])]
    for _, mode in modes:
        row = stats()
        for index, example in enumerate(known_rows, 1):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            expected = benchmark["output"] > 0
            try:
                base_raw = run(base_sessions[mode], benchmark["input"])
                cand_raw = run(cand_sessions[mode], benchmark["input"])
                correct = np.array_equal(cand_raw > 0, expected)
                row["right" if correct else "wrong"] += 1
                row["decoded_equal_base"] += int(np.array_equal(cand_raw > 0, base_raw > 0))
                row["raw_bitwise_equal_base"] += int(np.array_equal(cand_raw, base_raw, equal_nan=True))
                if not correct and row["first_failure"] is None:
                    row["first_failure"] = {"known_case": index, "kind": "gold"}
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {"known_case": index, "kind": "runtime", "error": repr(exc)}
        row["total"] = len(known_rows)
        output["known"][mode] = row

    task_hashes = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_hashes[f'{TASK:03d}']}")
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        seed_rows = {mode: stats() for _, mode in modes}
        valid = attempts = skips = generation_errors = 0
        while valid < COUNT:
            attempts += 1
            try:
                example = generator.generate()
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                skips += 1
                continue
            valid += 1
            expected = benchmark["output"] > 0
            for _, mode in modes:
                row = seed_rows[mode]
                try:
                    base_raw = run(base_sessions[mode], benchmark["input"])
                    cand_raw = run(cand_sessions[mode], benchmark["input"])
                    correct = np.array_equal(cand_raw > 0, expected)
                    row["right" if correct else "wrong"] += 1
                    row["decoded_equal_base"] += int(np.array_equal(cand_raw > 0, base_raw > 0))
                    row["raw_bitwise_equal_base"] += int(np.array_equal(cand_raw, base_raw, equal_nan=True))
                    if not correct and row["first_failure"] is None:
                        row["first_failure"] = {"fresh_case": valid, "kind": "gold"}
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {"fresh_case": valid, "kind": "runtime", "error": repr(exc)}
            if valid % 250 == 0:
                output["fresh"][str(seed)] = {
                    "valid": valid,
                    "attempts": attempts,
                    "conversion_skips": skips,
                    "generation_errors": generation_errors,
                    "modes": seed_rows,
                }
                REPORT.write_text(json.dumps(output, indent=2) + "\n")
                print(f"seed={seed} valid={valid}/{COUNT}", flush=True)
        output["fresh"][str(seed)] = {
            "valid": valid,
            "attempts": attempts,
            "conversion_skips": skips,
            "generation_errors": generation_errors,
            "modes": seed_rows,
        }
        REPORT.write_text(json.dumps(output, indent=2) + "\n")

    passed = True
    for row in output["known"].values():
        passed &= row["right"] == row["total"] and row["runtime_errors"] == 0
        passed &= row["raw_bitwise_equal_base"] == row["total"]
    for seed_row in output["fresh"].values():
        for row in seed_row["modes"].values():
            passed &= row["right"] == COUNT and row["runtime_errors"] == 0
            passed &= row["raw_bitwise_equal_base"] == COUNT
    output["pass"] = bool(passed)
    REPORT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"pass": output["pass"], "report": str(REPORT)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
