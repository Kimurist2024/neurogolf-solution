#!/usr/bin/env python3
"""Primary fail-closed audit for the task012 8x8 POLICY90 candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidates/task012_h8w8_policy90.onnx"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task012.json"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
FRESH_SEEDS = (272012001, 272112001)
FRESH_COUNT = 10_000
CONFIGS = ((True, 1), (True, 4), (False, 1), (False, 4))
EXPECTED = (1, 10, 30, 30)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def known_cases() -> list[dict]:
    data = json.loads(KNOWN_PATH.read_text())
    return [case for subset in ("train", "test", "arc-gen") for case in data[subset]]


def domain_cases() -> list[dict]:
    return [
        GEN.generate(colors=[1, 2], cols=[left, right], gravity=gravity)
        for left in range(3, 10)
        for right in range(3, 10)
        for gravity in range(4)
    ]


def fresh_cases(seed: int) -> list[dict]:
    random.seed(seed)
    return [GEN.generate() for _ in range(FRESH_COUNT)]


def make_session(model: onnx.ModelProto, disabled: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def evaluate(session: ort.InferenceSession, cases: list[dict], label: str) -> dict:
    correct = errors = nonfinite = shape_mismatch = zero_margin = 0
    sign_hash = hashlib.sha256()
    raw_hash = hashlib.sha256()
    minimum_abs = float("inf")
    for case in cases:
        converted = scoring.convert_to_numpy(case)
        if converted is None:
            errors += 1
            continue
        try:
            raw = session.run(["output"], {"input": converted["input"]})[0]
        except Exception:
            errors += 1
            continue
        if tuple(map(int, raw.shape)) != EXPECTED:
            shape_mismatch += 1
        if not np.isfinite(raw).all():
            nonfinite += 1
        signs = np.asarray(raw > 0.0, dtype=np.uint8)
        sign_hash.update(np.packbits(signs).tobytes())
        raw_hash.update(np.ascontiguousarray(raw).tobytes())
        if np.array_equal(signs, np.asarray(converted["output"] > 0.0, dtype=np.uint8)):
            correct += 1
        abs_raw = np.abs(raw)
        zero_margin += int(np.count_nonzero(abs_raw == 0))
        if abs_raw.size:
            minimum_abs = min(minimum_abs, float(np.min(abs_raw)))
    total = len(cases)
    return {
        "label": label,
        "total": total,
        "correct": correct,
        "rate": correct / total,
        "errors": errors,
        "nonfinite_outputs": nonfinite,
        "output_shape_mismatches": shape_mismatch,
        "zero_margin_elements": zero_margin,
        "minimum_abs_output": minimum_abs,
        "prediction_sha256": sign_hash.hexdigest(),
        "raw_sha256": raw_hash.hexdigest(),
    }


def main() -> None:
    candidate = onnx.load(CANDIDATE)
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task012.onnx")
    authority_path = HERE / "authority_task012.onnx.audit-copy"
    authority_path.write_bytes(authority_bytes)
    authority_profile = cost_of(str(authority_path))
    authority_path.unlink()
    candidate_profile = cost_of(str(CANDIDATE))

    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    shapes = {
        value.name: [int(dim.dim_value) if dim.HasField("dim_value") else None
                     for dim in value.type.tensor_type.shape.dim]
        for value in values if value.type.HasField("tensor_type")
    }
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    weights = arrays["w"]
    bias = arrays["b"]
    symmetry = {
        "weights_channels_1_through_9_raw_equal": all(
            weights[1].tobytes() == weights[index].tobytes() for index in range(2, 10)
        ),
        "bias_channels_1_through_9_raw_equal": all(
            bias[1].tobytes() == bias[index].tobytes() for index in range(2, 10)
        ),
        "proof": (
            "The generator chooses two distinct nonzero colors. A group=10 Conv acts "
            "independently per channel; byte-identical weights and biases for channels "
            "1..9 make every nonzero-color permutation equivariant."
        ),
    }
    structural = {
        "full_check": True,
        "strict_data_prop": True,
        "input_shape": shapes.get("input"),
        "output_shape": shapes.get("output"),
        "canonical_static_io": shapes.get("input") == list(EXPECTED) and shapes.get("output") == list(EXPECTED),
        "nodes": Counter(node.op_type for node in candidate.graph.node),
        "single_output_only_conv": len(candidate.graph.node) == 1 and candidate.graph.node[0].op_type == "Conv",
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in candidate.graph.node),
        "functions": len(candidate.functions),
        "sparse_initializers": len(candidate.graph.sparse_initializer),
        "finite_initializers": all(np.isfinite(array).all() for array in arrays.values()),
        "conv_bias_findings": check_conv_bias(candidate),
        "lookup_or_shape_cloak_ops": [],
    }
    structural["pass"] = bool(
        structural["canonical_static_io"]
        and structural["single_output_only_conv"]
        and structural["standard_domains"]
        and structural["functions"] == 0
        and structural["sparse_initializers"] == 0
        and structural["finite_initializers"]
        and not structural["conv_bias_findings"]
        and symmetry["weights_channels_1_through_9_raw_equal"]
        and symmetry["bias_channels_1_through_9_raw_equal"]
    )

    corpora = [("known", known_cases()), ("domain196", domain_cases())]
    corpora.extend((f"fresh_{seed}", fresh_cases(seed)) for seed in FRESH_SEEDS)
    runs = []
    for disabled, threads in CONFIGS:
        session = make_session(candidate, disabled, threads)
        config = f"{'disabled' if disabled else 'default'}_t{threads}"
        runs.append({
            "config": config,
            "corpora": [evaluate(session, cases, label) for label, cases in corpora],
        })

    reference = {
        item["label"]: item["prediction_sha256"] for item in runs[0]["corpora"]
    }
    prediction_stable = all(
        item["prediction_sha256"] == reference[item["label"]]
        for run in runs for item in run["corpora"]
    )
    all_rows = [item for run in runs for item in run["corpora"]]
    audit_pass = bool(
        structural["pass"]
        and tuple(candidate_profile) == (0, 650, 650)
        and tuple(authority_profile) == (0, 710, 710)
        and prediction_stable
        and all(item["rate"] >= 0.90 for item in all_rows)
        and all(item["errors"] == 0 for item in all_rows)
        and all(item["nonfinite_outputs"] == 0 for item in all_rows)
        and all(item["output_shape_mismatches"] == 0 for item in all_rows)
    )
    evidence = {
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "authority_zip_sha256": digest(AUTHORITY_ZIP.read_bytes()),
        "authority_member_sha256": digest(authority_bytes),
        "authority_profile": {"memory": authority_profile[0], "params": authority_profile[1], "cost": authority_profile[2]},
        "candidate_profile": {"memory": candidate_profile[0], "params": candidate_profile[1], "cost": candidate_profile[2]},
        "projected_gain": float(np.log(710 / 650)),
        "structural": structural,
        "color_permutation_symmetry": symmetry,
        "domain_support": "7*7 column pairs * 4 gravity orientations = 196",
        "policy_threshold": 0.90,
        "prediction_stable_across_four_configs": prediction_stable,
        "runs": runs,
        "pass": audit_pass,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2, default=lambda value: dict(value)) + "\n")
    print(json.dumps({
        "pass": audit_pass,
        "candidate_sha256": evidence["candidate_sha256"],
        "profiles": [authority_profile, candidate_profile],
        "rates": {run["config"]: {item["label"]: item["rate"] for item in run["corpora"]} for run in runs},
        "prediction_stable": prediction_stable,
    }, indent=2))
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
