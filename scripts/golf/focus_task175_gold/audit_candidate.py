#!/usr/bin/env python3
"""Independent 2-seed strict audit for the exact task175 gauge candidate."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
AUTHORITY = ROOT / "submission_base_8014.69.zip"
EXPECTED_AUTHORITY_SHA = "a5a811393b5b378c3bfe1e9aef29680b8af1671440aa21e900fe8c05ad54c328"
EXPECTED_MEMBER_SHA = "b6404486ccc1a74c36bab6031f11c54c7326f787a743f02dff77e63c782af343"
SEEDS = (777_175, 1_775_175)
K = 2_000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def raw_session(model: onnx.ModelProto) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def fresh_seed(session: ort.InferenceSession, seed: int) -> dict[str, float | int]:
    generator = importlib.import_module("task_73251a56")
    random.seed(seed)
    total = failures = attempts = 0
    min_positive = float("inf")
    max_nonpositive = -float("inf")
    while total < K:
        attempts += 1
        example = generator.generate()
        if not isinstance(example, dict) or "input" not in example or "output" not in example:
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        raw = session.run(["output"], {"input": benchmark["input"]})[0]
        predicted = raw > 0.0
        expected = benchmark["output"] > 0.0
        if not np.array_equal(predicted, expected):
            failures += 1
        if np.any(predicted):
            min_positive = min(min_positive, float(raw[predicted].min()))
        if np.any(~predicted):
            max_nonpositive = max(max_nonpositive, float(raw[~predicted].max()))
        total += 1
    return {
        "seed": seed,
        "attempts": attempts,
        "total": total,
        "failures": failures,
        "accuracy": (total - failures) / total,
        "min_positive_raw": min_positive,
        "max_nonpositive_raw": max_nonpositive,
    }


def main() -> None:
    archive_blob = AUTHORITY.read_bytes()
    if digest(archive_blob) != EXPECTED_AUTHORITY_SHA:
        raise RuntimeError("authority archive drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        member = archive.read("task175.onnx")
    if digest(member) != EXPECTED_MEMBER_SHA:
        raise RuntimeError("authority member drift")

    candidate_blob = CANDIDATE.read_bytes()
    model = onnx.load_model_from_string(candidate_blob)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_shape = [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim]
    memory, params, cost = cost_of(str(CANDIDATE))
    session = raw_session(model)
    audits = [fresh_seed(session, seed) for seed in SEEDS]
    result = {
        "task": 175,
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "archive_sha256": EXPECTED_AUTHORITY_SHA,
            "member_sha256": EXPECTED_MEMBER_SHA,
            "cost": 140,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(candidate_blob),
            "file_size": len(candidate_blob),
            "memory": memory,
            "params": params,
            "cost": cost,
            "output_shape": output_shape,
        },
        "fresh": audits,
        "strict_fresh_pass": all(item["failures"] == 0 and item["total"] == K for item in audits),
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["strict_fresh_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
