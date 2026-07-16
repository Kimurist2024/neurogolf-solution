#!/usr/bin/env python3
"""Fail-closed preliminary POLICY90 screen for the five cost-249 task355 probes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_DIR = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task355.json"
FRESH_SEEDS = (283_355_001, 283_455_001)
FRESH_COUNT = 5_000
EXPECTED = (1, 10, 30, 30)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GENERATOR = importlib.import_module("task_de1cd16c")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cases_known() -> list[dict]:
    payload = json.loads(KNOWN_PATH.read_text())
    return [payload[split][i] for split in ("train", "test", "arc-gen") for i in range(len(payload[split]))]


def cases_fresh(seed: int) -> list[dict]:
    random.seed(seed)
    return [GENERATOR.generate() for _ in range(FRESH_COUNT)]


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def evaluate(run: ort.InferenceSession, cases: list[dict]) -> dict:
    right = wrong = errors = nonfinite = shape = zero = 0
    signs = hashlib.sha256()
    first_failure = None
    for index, case in enumerate(cases):
        converted = scoring.convert_to_numpy(case)
        if converted is None:
            errors += 1
            continue
        try:
            raw = run.run(["output"], {"input": converted["input"]})[0]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
            continue
        shape += int(tuple(raw.shape) != EXPECTED)
        nonfinite += int(not np.isfinite(raw).all())
        zero += int(np.count_nonzero(raw == 0.0))
        actual = np.asarray(raw > 0.0, dtype=np.uint8)
        signs.update(np.packbits(actual).tobytes())
        expected = np.asarray(converted["output"] > 0.0, dtype=np.uint8)
        if np.array_equal(actual, expected):
            right += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "wrong",
                    "differing_cells": int(np.count_nonzero(actual != expected)),
                }
    total = len(cases)
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": total,
        "accuracy": right / total,
        "nonfinite_cases": nonfinite,
        "shape_mismatches": shape,
        "zero_margin_elements": zero,
        "sign_sha256": signs.hexdigest(),
        "first_failure": first_failure,
    }


def main() -> None:
    started = time.time()
    authority_bytes = AUTHORITY_ZIP.read_bytes()
    if digest(authority_bytes) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_member = archive.read("task355.onnx")
    temporary = HERE / "authority_task355.onnx.audit-copy"
    temporary.write_bytes(authority_member)
    authority_cost = cost_of(str(temporary))
    temporary.unlink()

    corpora = [("known", cases_known())]
    corpora.extend((f"fresh_{seed}", cases_fresh(seed)) for seed in FRESH_SEEDS)
    results = []
    for rank in range(1, 6):
        path = SOURCE_DIR / f"task355_r{rank:02d}_static249.onnx"
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        cost = cost_of(str(path))
        conv = check_conv_bias(model)
        modes = {}
        session_error = None
        try:
            for label, disabled in (("disabled", True), ("default", False)):
                run = session(model, disabled)
                modes[label] = {name: evaluate(run, cases) for name, cases in corpora}
        except Exception as exc:  # noqa: BLE001
            session_error = repr(exc)
        passed = bool(
            session_error is None
            and tuple(cost) == (227, 22, 249)
            and not conv
            and all(
                row["accuracy"] >= 0.90
                and row["errors"] == 0
                and row["nonfinite_cases"] == 0
                and row["shape_mismatches"] == 0
                for mode in modes.values() for row in mode.values()
            )
            and all(
                modes["disabled"][name]["sign_sha256"]
                == modes["default"][name]["sign_sha256"]
                for name, _ in corpora
            )
        )
        results.append({
            "rank": rank,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "cost": {"memory": cost[0], "params": cost[1], "cost": cost[2]},
            "node_count": len(model.graph.node),
            "strict_shape_inference": True,
            "inferred_output_shape": [
                int(dim.dim_value) if dim.HasField("dim_value") else None
                for dim in inferred.graph.output[0].type.tensor_type.shape.dim
            ],
            "conv_bias_findings": conv,
            "session_error": session_error,
            "modes": modes,
            "preliminary_pass": passed,
        })
    output = {
        "status": "PRELIMINARY_ONLY_DO_NOT_STAGE",
        "authority_zip_sha256": digest(authority_bytes),
        "authority_member_sha256": digest(authority_member),
        "authority_cost": {"memory": authority_cost[0], "params": authority_cost[1], "cost": authority_cost[2]},
        "fresh_count_per_seed": FRESH_COUNT,
        "fresh_seeds": FRESH_SEEDS,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "screen.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "authority_cost": output["authority_cost"],
        "elapsed_seconds": output["elapsed_seconds"],
        "results": [
            {
                "rank": row["rank"],
                "pass": row["preliminary_pass"],
                "known": row["modes"].get("disabled", {}).get("known", {}).get("accuracy"),
                "fresh": [
                    item["accuracy"]
                    for name, item in row["modes"].get("disabled", {}).items()
                    if name.startswith("fresh_")
                ],
            }
            for row in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
