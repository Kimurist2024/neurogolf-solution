#!/usr/bin/env python3
"""Fail-closed four-configuration POLICY90 audit for task007 cost 68."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / (
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task007_r01_static68.onnx"
)
BASE_ZIP = ROOT / "submission_base_8009.46.zip"
BASE_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task007.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
OUTPUT = HERE / "evidence.json"
EXPECTED_SHA256 = "fa22f345634e3f059b0b2d334e6b9d85d60973d5cc2a6c92003b8f7cfc60486a"
EXPECTED_IO = (1, 10, 30, 30)
POLICY_THRESHOLD = 0.90
FRESH_SEEDS = (277_007_001, 277_107_001)
FRESH_PER_SEED = 10_000
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("default_threads1", "default", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads4", "default", 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GENERATOR = importlib.import_module("task_05269061")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def profile(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"task007_277_{label}_", dir="/tmp") as work:
        path = Path(work) / "task007.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                count += 1
            elif attribute.type == AttributeProto.GRAPHS:
                count += len(attribute.graphs)
    return count


def static_audit(model: onnx.ModelProto) -> dict[str, Any]:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    node_outputs = [name for node in inferred.graph.node for name in node.output if name]
    nonstatic = [
        name for name, value in typed.items()
        if not dims(value) or any(dim is None or dim <= 0 for dim in dims(value))
    ]
    missing = [name for name in node_outputs if name not in typed]
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializer_rows = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": (
                int(np.count_nonzero(~np.isfinite(array)))
                if array.dtype.kind in "fc" else 0
            ),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }
        for name, array in arrays.items()
    }
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    banned = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in BANNED or "Sequence" in node.op_type
    })
    domains = sorted({
        domain
        for domain in [
            *(item.domain for item in model.opset_import),
            *(node.domain for node in model.graph.node),
        ]
        if domain not in ("", "ai.onnx")
    })
    external = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    graph_input_shape = dims(inferred.graph.input[0]) if len(inferred.graph.input) == 1 else []
    graph_output_shape = dims(inferred.graph.output[0]) if len(inferred.graph.output) == 1 else []
    explicit_lookup_ops = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax", "Gather", "GatherND", "ScatterND"}
    })
    giant_initializers = [
        name for name, array in arrays.items() if int(array.size) >= 10_000
    ]
    conv_findings = check_conv_bias(model)
    return {
        "full_check": True,
        "strict_data_prop": True,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "canonical_input": graph_input_shape == list(EXPECTED_IO),
        "canonical_output": graph_output_shape == list(EXPECTED_IO),
        "input_shape": graph_input_shape,
        "output_shape": graph_output_shape,
        "all_typed_node_outputs_static_positive": not nonstatic and not missing,
        "nonstatic_typed_values": nonstatic,
        "missing_typed_node_outputs": missing,
        "standard_ops_only": not domains,
        "nonstandard_domains": domains,
        "banned_ops": banned,
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "initializers": initializer_rows,
        "finite_initializers": all(row["nonfinite"] == 0 for row in initializer_rows.values()),
        "initializer_elements": sum(row["elements"] for row in initializer_rows.values()),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum >= 15,
        "giant_initializers": giant_initializers,
        "explicit_lookup_ops": explicit_lookup_ops,
        "no_lookup_or_fixture_table": not explicit_lookup_ops and not giant_initializers,
        "conv_bias_findings": conv_findings,
        "conv_bias_ub0": not conv_findings,
    }


def make_session(model: onnx.ModelProto, optimization: str, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
    options = ort.SessionOptions()
    if optimization == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif optimization != "default":
        raise ValueError(optimization)
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "shape_mismatches": 0,
        "nonfinite_cases": 0,
        "nonfinite_elements": 0,
        "small_positive_elements": 0,
        "minimum_positive": math.inf,
        "maximum_nonpositive": -math.inf,
        "sign_mismatch_cases_vs_disable_threads1": 0,
        "sign_mismatch_cells_vs_disable_threads1": 0,
        "first_failure": None,
        "sign_sha256": hashlib.sha256(),
        "raw_sha256": hashlib.sha256(),
    }


def finish_stats(row: dict[str, Any], elapsed: float) -> dict[str, Any]:
    total = row["right"] + row["wrong"] + row["errors"]
    return {
        **{
            key: value for key, value in row.items()
            if key not in {"sign_sha256", "raw_sha256", "minimum_positive", "maximum_nonpositive"}
        },
        "total": total,
        "accuracy": row["right"] / total if total else 0.0,
        "policy90": total > 0 and row["right"] / total >= POLICY_THRESHOLD,
        "minimum_positive": None if math.isinf(row["minimum_positive"]) else row["minimum_positive"],
        "maximum_nonpositive": None if math.isinf(row["maximum_nonpositive"]) else row["maximum_nonpositive"],
        "sign_sha256": row["sign_sha256"].hexdigest(),
        "raw_sha256": row["raw_sha256"].hexdigest(),
        "elapsed_seconds": elapsed,
    }


def evaluate_cases(
    sessions: dict[str, ort.InferenceSession],
    examples: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stats = {label: empty_stats() for label, _, _ in CONFIGS}
    started = time.monotonic()
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"case {index} did not convert")
        want = benchmark["output"] > 0
        baseline_sign = None
        for label, _, _ in CONFIGS:
            row = stats[label]
            try:
                raw = sessions[label].run(None, {"input": benchmark["input"]})[0]
                raw = np.asarray(raw)
                row["raw_sha256"].update(np.ascontiguousarray(raw).tobytes())
                if tuple(raw.shape) != EXPECTED_IO:
                    row["shape_mismatches"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {"case": index, "reason": "shape", "shape": list(raw.shape)}
                    continue
                finite = np.isfinite(raw)
                nonfinite = int(np.count_nonzero(~finite))
                if nonfinite:
                    row["nonfinite_cases"] += 1
                    row["nonfinite_elements"] += nonfinite
                sign = raw > 0
                packed = np.packbits(sign.reshape(-1), bitorder="little").tobytes()
                row["sign_sha256"].update(packed)
                if baseline_sign is None:
                    baseline_sign = sign
                else:
                    differing = int(np.count_nonzero(sign != baseline_sign))
                    if differing:
                        row["sign_mismatch_cases_vs_disable_threads1"] += 1
                        row["sign_mismatch_cells_vs_disable_threads1"] += differing
                positives = raw[(raw > 0) & finite]
                nonpositives = raw[(raw <= 0) & finite]
                if positives.size:
                    row["minimum_positive"] = min(row["minimum_positive"], float(np.min(positives)))
                    row["small_positive_elements"] += int(np.count_nonzero(positives < 0.25))
                if nonpositives.size:
                    row["maximum_nonpositive"] = max(row["maximum_nonpositive"], float(np.max(nonpositives)))
                if nonfinite == 0 and np.array_equal(sign, want):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "case": index,
                            "reason": "wrong" if nonfinite == 0 else "nonfinite",
                            "different_cells": int(np.count_nonzero(sign != want)),
                        }
            except Exception as exc:  # noqa: BLE001
                row["errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "reason": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    elapsed = time.monotonic() - started
    return {label: finish_stats(row, elapsed) for label, row in stats.items()}


def fresh_examples(seed: int, count: int) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return [GENERATOR.generate() for _ in range(count)]


def main() -> None:
    data = SOURCE.read_bytes()
    if sha256(data) != EXPECTED_SHA256:
        raise RuntimeError("candidate SHA changed")
    if sha256(BASE_ZIP.read_bytes()) != BASE_ZIP_SHA256:
        raise RuntimeError("pinned 8009.46 authority ZIP changed")
    with zipfile.ZipFile(BASE_ZIP) as archive:
        authority = archive.read("task007.onnx")

    model = onnx.load_model_from_string(data)
    static = static_audit(model)
    candidate_profile = profile(data, "candidate")
    authority_profile = profile(authority, "authority")
    if candidate_profile != {"memory": 0, "params": 68, "cost": 68}:
        raise RuntimeError(f"unexpected candidate profile: {candidate_profile}")
    if authority_profile["cost"] != 70:
        raise RuntimeError(f"unexpected authority profile: {authority_profile}")

    sessions = {
        label: make_session(model, optimization, threads)
        for label, optimization, threads in CONFIGS
    }
    known_payload = json.loads(KNOWN_FILE.read_text(encoding="utf-8"))
    split_counts = {
        name: len(known_payload[name]) for name in ("train", "test", "arc-gen")
    }
    known_examples = [
        example
        for name in ("train", "test", "arc-gen")
        for example in known_payload[name]
    ]
    known = evaluate_cases(sessions, known_examples)
    fresh: dict[str, Any] = {}
    for seed in FRESH_SEEDS:
        fresh[str(seed)] = evaluate_cases(sessions, fresh_examples(seed, FRESH_PER_SEED))
        print(f"task007 fresh seed {seed} complete", flush=True)

    runtime_rows = [*known.values(), *(row for seed in fresh.values() for row in seed.values())]
    structural_pass = bool(
        static["full_check"] and static["strict_data_prop"]
        and static["canonical_input"] and static["canonical_output"]
        and static["all_typed_node_outputs_static_positive"]
        and static["standard_ops_only"] and not static["banned_ops"]
        and static["nested_graphs"] == 0 and static["functions"] == 0
        and static["sparse_initializers"] == 0 and not static["external_initializers"]
        and static["finite_initializers"] and not static["giant_einsum"]
        and not static["giant_initializers"] and static["no_lookup_or_fixture_table"]
        and static["conv_bias_ub0"]
    )
    runtime_pass = all(
        row["policy90"] and row["errors"] == 0 and row["shape_mismatches"] == 0
        and row["nonfinite_elements"] == 0 and row["small_positive_elements"] == 0
        and row["sign_mismatch_cases_vs_disable_threads1"] == 0
        for row in runtime_rows
    )
    accepted = structural_pass and runtime_pass
    payload = {
        "task": 7,
        "lane": "root_task007_policy90_277",
        "decision": "ACCEPT_POLICY90" if accepted else "REJECT_FAIL_CLOSED",
        "accepted": accepted,
        "authority": {
            "zip": relative(BASE_ZIP),
            "zip_sha256": BASE_ZIP_SHA256,
            "member": "task007.onnx",
            "member_sha256": sha256(authority),
            "actual_profile": authority_profile,
        },
        "candidate": {
            "source": relative(SOURCE),
            "sha256": sha256(data),
            "file_bytes": len(data),
            "actual_profile": candidate_profile,
            "strict_lower_by": authority_profile["cost"] - candidate_profile["cost"],
            "score_gain": math.log(authority_profile["cost"] / candidate_profile["cost"]),
        },
        "static": static,
        "structural_pass": structural_pass,
        "known": {"split_counts": split_counts, "configs": known},
        "fresh": {
            "seeds": list(FRESH_SEEDS),
            "per_seed": FRESH_PER_SEED,
            "configs_by_seed": fresh,
        },
        "runtime_pass": runtime_pass,
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "four_ort_configurations": True,
            "errors_required": 0,
            "nonfinite_required": 0,
            "shape_mismatches_required": 0,
            "small_positive_values_required": 0,
            "private_zero_lineage": False,
            "normal_policy90": True,
            "root_or_71407_written": False,
            "kimi_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "candidate_profile": candidate_profile,
        "authority_profile": authority_profile,
        "known_accuracy": {label: row["accuracy"] for label, row in known.items()},
        "fresh_accuracy": {
            seed: {label: row["accuracy"] for label, row in rows.items()}
            for seed, rows in fresh.items()
        },
        "evidence": relative(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
