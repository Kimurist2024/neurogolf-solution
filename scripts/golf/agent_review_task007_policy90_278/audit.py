#!/usr/bin/env python3
"""Independent four-runtime review of the task007 cost68 POLICY90 model."""

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
from onnx import AttributeProto, TensorProto, helper, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / (
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task007_r01_static68.onnx"
)
EXPECTED_CANDIDATE_SHA256 = (
    "fa22f345634e3f059b0b2d334e6b9d85d60973d5cc2a6c92003b8f7cfc60486a"
)
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
EXPECTED_AUTHORITY_ZIP_SHA256 = (
    "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
)
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task007.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
OUTPUT = HERE / "evidence.json"
EXPECTED_IO = (1, 10, 30, 30)
FRESH_SEEDS = (278_007_001, 278_107_001)
FRESH_PER_SEED = 10_000
POLICY_THRESHOLD = 0.90
CONFIGS = (
    ("disable_threads1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_threads4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("default_threads1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("default_threads4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
LOOKUP_OPS = {
    "TfIdfVectorizer",
    "Hardmax",
    "Gather",
    "GatherElements",
    "GatherND",
    "ScatterElements",
    "ScatterND",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GENERATOR = importlib.import_module("task_05269061")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def nested_graphs(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                count += 1
                pending.extend(attr.g.node)
            elif attr.type == AttributeProto.GRAPHS:
                count += len(attr.graphs)
                for graph in attr.graphs:
                    pending.extend(graph.node)
    return count


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["full_check"] = True
    except Exception as exc:
        result.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        result["strict_data_prop"] = True
    except Exception as exc:
        result.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
        inferred = model

    values = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    node_outputs = [name for node in inferred.graph.node for name in node.output if name]
    nonstatic = [
        name
        for name, value in values.items()
        if not shape(value) or any(dim is None or dim <= 0 for dim in shape(value))
    ]
    missing = [name for name in node_outputs if name not in values]
    arrays = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer))
        for initializer in model.graph.initializer
    }
    initializer_audit = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": int(np.count_nonzero(~np.isfinite(array))),
            "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
        }
        for name, array in arrays.items()
    }
    domains = sorted(
        {
            domain
            for domain in [
                *(item.domain for item in model.opset_import),
                *(node.domain for node in model.graph.node),
            ]
            if domain not in ("", "ai.onnx")
        }
    )
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    lookup = sorted({node.op_type for node in model.graph.node if node.op_type in LOOKUP_OPS})
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    external = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL or initializer.external_data
    ]
    histogram = Counter(node.op_type for node in model.graph.node)
    einsum_equations = [
        helper.get_attribute_value(attr).decode("ascii")
        for node in model.graph.node
        if node.op_type == "Einsum"
        for attr in node.attribute
        if attr.name == "equation"
    ]
    input_shape = shape(inferred.graph.input[0]) if len(inferred.graph.input) == 1 else []
    output_shape = shape(inferred.graph.output[0]) if len(inferred.graph.output) == 1 else []
    result.update(
        {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "canonical_float_input": bool(
                input_shape == list(EXPECTED_IO)
                and inferred.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
            ),
            "canonical_float_output": bool(
                output_shape == list(EXPECTED_IO)
                and inferred.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
            ),
            "all_node_outputs_typed_static_positive": not nonstatic and not missing,
            "nonstatic_values": nonstatic,
            "missing_node_outputs": missing,
            "node_count": len(model.graph.node),
            "op_histogram": dict(sorted(histogram.items())),
            "standard_domains": not domains,
            "nonstandard_domains": domains,
            "banned_ops": banned,
            "nested_graphs": nested_graphs(model),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "external_initializers": external,
            "initializer_audit": initializer_audit,
            "initializer_elements": sum(row["elements"] for row in initializer_audit.values()),
            "finite_initializers": all(row["nonfinite"] == 0 for row in initializer_audit.values()),
            "lookup_ops": lookup,
            "max_einsum_inputs": max_einsum_inputs,
            "giant_einsum": max_einsum_inputs >= 15,
            "giant_initializers": [
                name for name, row in initializer_audit.items() if row["elements"] >= 10_000
            ],
            "einsum_equations": einsum_equations,
            "conv_bias_findings": check_conv_bias(model),
            "file_bytes": len(model.SerializeToString()),
            "under_file_limit": len(model.SerializeToString()) <= scoring.FILESIZE_LIMIT_IN_BYTES,
        }
    )
    result["no_lookup_cloak_or_giant"] = bool(
        len(model.graph.node) == 1
        and histogram == Counter({"Einsum": 1})
        and not lookup
        and max_einsum_inputs == 10
        and not result["giant_initializers"]
        and result["initializer_elements"] == 68
        and output_shape == list(EXPECTED_IO)
        and not model.graph.value_info
    )
    result["pass"] = bool(
        result.get("full_check")
        and result.get("strict_data_prop")
        and result["canonical_float_input"]
        and result["canonical_float_output"]
        and result["all_node_outputs_typed_static_positive"]
        and result["standard_domains"]
        and not banned
        and result["nested_graphs"] == 0
        and result["functions"] == 0
        and result["sparse_initializers"] == 0
        and not external
        and result["finite_initializers"]
        and result["no_lookup_cloak_or_giant"]
        and not result["conv_bias_findings"]
        and result["under_file_limit"]
    )
    return result


def profile(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"review278_{label}_", dir="/tmp") as directory:
        path = Path(directory) / "task007.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def official_profile(model: onnx.ModelProto, label: str, require_correct: bool) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"review278_official_{label}_", dir="/tmp") as directory:
        return scoring.score_and_verify(
            copy.deepcopy(model), 7, directory, label=label, require_correct=require_correct
        )


def session(
    model: onnx.ModelProto,
    optimization: ort.GraphOptimizationLevel,
    threads: int,
) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def direct_onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros(EXPECTED_IO, dtype=np.float32)
    rows, columns = np.indices(values.shape)
    result[0, values, rows, columns] = 1.0
    return result


def case_digest(examples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        for key in ("input", "output"):
            array = np.asarray(example[key], dtype=np.uint8)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def converter_crosscheck(examples: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = 0
    for example in examples:
        official = scoring.convert_to_numpy(example)
        if official is None or not (
            np.array_equal(official["input"], direct_onehot(example["input"]))
            and np.array_equal(official["output"], direct_onehot(example["output"]))
        ):
            mismatches += 1
    return {"cases": len(examples), "mismatches": mismatches, "exact": mismatches == 0}


def fresh_examples(seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return [GENERATOR.generate() for _ in range(FRESH_PER_SEED)]


def empty_runtime() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite_cases": 0,
        "nonfinite_elements": 0,
        "shape_mismatches": 0,
        "small_positive_elements_0_to_0_25": 0,
        "minimum_positive": math.inf,
        "maximum_nonpositive": -math.inf,
        "sign_mismatch_cases_vs_disable_threads1": 0,
        "sign_mismatch_cells_vs_disable_threads1": 0,
        "sign_sha256": hashlib.sha256(),
        "raw_sha256": hashlib.sha256(),
        "observed_shapes": set(),
        "first_failure": None,
    }


def evaluate(
    runtimes: dict[str, ort.InferenceSession], examples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = {name: empty_runtime() for name, _optimization, _threads in CONFIGS}
    started = time.monotonic()
    for index, example in enumerate(examples):
        official = scoring.convert_to_numpy(example)
        if official is None:
            raise RuntimeError(f"case {index} failed official conversion")
        expected = official["output"] > 0
        baseline_sign = None
        for name, _optimization, _threads in CONFIGS:
            row = rows[name]
            try:
                raw = np.asarray(runtimes[name].run(["output"], {"input": official["input"]})[0])
            except Exception as exc:
                row["errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "runtime_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                continue
            actual_shape = tuple(int(dim) for dim in raw.shape)
            row["observed_shapes"].add(actual_shape)
            if actual_shape != EXPECTED_IO:
                row["shape_mismatches"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "shape_mismatch",
                        "actual": list(actual_shape),
                    }
                continue
            finite = np.isfinite(raw)
            current_nonfinite = int(np.count_nonzero(~finite))
            row["nonfinite_cases"] += int(current_nonfinite > 0)
            row["nonfinite_elements"] += current_nonfinite
            positive = raw > 0
            packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
            row["sign_sha256"].update(packed)
            row["raw_sha256"].update(np.ascontiguousarray(raw).tobytes())
            if baseline_sign is None:
                baseline_sign = positive
            else:
                differing = int(np.count_nonzero(positive != baseline_sign))
                row["sign_mismatch_cases_vs_disable_threads1"] += int(differing > 0)
                row["sign_mismatch_cells_vs_disable_threads1"] += differing
            positive_values = raw[finite & positive]
            nonpositive_values = raw[finite & ~positive]
            if positive_values.size:
                row["minimum_positive"] = min(
                    row["minimum_positive"], float(np.min(positive_values))
                )
                row["small_positive_elements_0_to_0_25"] += int(
                    np.count_nonzero(positive_values < 0.25)
                )
            if nonpositive_values.size:
                row["maximum_nonpositive"] = max(
                    row["maximum_nonpositive"], float(np.max(nonpositive_values))
                )
            if current_nonfinite == 0 and np.array_equal(positive, expected):
                row["right"] += 1
            else:
                row["wrong"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "wrong_or_nonfinite",
                        "different_cells": int(np.count_nonzero(positive != expected)),
                    }
    elapsed = time.monotonic() - started
    result = {}
    for name, optimization, threads in CONFIGS:
        row = rows[name]
        result[name] = {
            "optimization": (
                "ORT_DISABLE_ALL"
                if optimization == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                else "ORT_ENABLE_ALL"
            ),
            "threads": threads,
            "total": len(examples),
            "right": row["right"],
            "wrong": row["wrong"],
            "accuracy": row["right"] / len(examples),
            "policy90": row["right"] / len(examples) >= POLICY_THRESHOLD,
            "errors": row["errors"],
            "nonfinite_cases": row["nonfinite_cases"],
            "nonfinite_elements": row["nonfinite_elements"],
            "shape_mismatches": row["shape_mismatches"],
            "observed_shapes": [list(value) for value in sorted(row["observed_shapes"])],
            "small_positive_elements_0_to_0_25": row[
                "small_positive_elements_0_to_0_25"
            ],
            "minimum_positive": (
                None if math.isinf(row["minimum_positive"]) else row["minimum_positive"]
            ),
            "maximum_nonpositive": (
                None
                if row["maximum_nonpositive"] == -math.inf
                else row["maximum_nonpositive"]
            ),
            "sign_mismatch_cases_vs_disable_threads1": row[
                "sign_mismatch_cases_vs_disable_threads1"
            ],
            "sign_mismatch_cells_vs_disable_threads1": row[
                "sign_mismatch_cells_vs_disable_threads1"
            ],
            "sign_sha256": row["sign_sha256"].hexdigest(),
            "raw_sha256": row["raw_sha256"].hexdigest(),
            "first_failure": row["first_failure"],
            "elapsed_seconds": elapsed,
        }
    return result


def runtime_row_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["policy90"]
        and row["errors"] == 0
        and row["nonfinite_elements"] == 0
        and row["shape_mismatches"] == 0
        and row["small_positive_elements_0_to_0_25"] == 0
        and row["sign_mismatch_cases_vs_disable_threads1"] == 0
        and row["sign_mismatch_cells_vs_disable_threads1"] == 0
        and row["observed_shapes"] == [list(EXPECTED_IO)]
    )


def main() -> None:
    started = time.monotonic()
    candidate_bytes = CANDIDATE.read_bytes()
    if sha256_bytes(candidate_bytes) != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("candidate SHA mismatch")
    if sha256(AUTHORITY_ZIP) != EXPECTED_AUTHORITY_ZIP_SHA256:
        raise RuntimeError("pinned authority ZIP mismatch")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task007.onnx")
    candidate = onnx.load_model_from_string(candidate_bytes)
    authority = onnx.load_model_from_string(authority_bytes)

    structure = structural(candidate)
    candidate_profile = profile(candidate_bytes, "candidate")
    authority_profile = profile(authority_bytes, "authority")
    candidate_official = official_profile(candidate, "candidate", require_correct=False)
    authority_official = official_profile(authority, "authority", require_correct=True)

    known_payload = json.loads(KNOWN_FILE.read_text())
    known_examples = [
        example
        for split in ("train", "test", "arc-gen")
        for example in known_payload[split]
    ]
    known_meta = {
        "split_counts": {split: len(known_payload[split]) for split in ("train", "test", "arc-gen")},
        "total": len(known_examples),
        "case_sha256": case_digest(known_examples),
        "converter_crosscheck": converter_crosscheck(known_examples),
    }

    runtimes = {
        name: session(candidate, optimization, threads)
        for name, optimization, threads in CONFIGS
    }
    known_runtime = evaluate(runtimes, known_examples)
    print(
        json.dumps(
            {
                "dataset": "known",
                "rates": {name: row["accuracy"] for name, row in known_runtime.items()},
            }
        ),
        flush=True,
    )

    fresh_meta = []
    fresh_runtime = {}
    for seed in FRESH_SEEDS:
        examples = fresh_examples(seed)
        fresh_meta.append(
            {
                "seed": seed,
                "count": len(examples),
                "case_sha256": case_digest(examples),
                "unique_input_output_pairs": len(
                    {
                        (
                            np.asarray(example["input"], dtype=np.uint8).tobytes(),
                            np.asarray(example["output"], dtype=np.uint8).tobytes(),
                        )
                        for example in examples
                    }
                ),
                "converter_crosscheck": converter_crosscheck(examples),
            }
        )
        fresh_runtime[str(seed)] = evaluate(runtimes, examples)
        print(
            json.dumps(
                {
                    "dataset": f"fresh_{seed}",
                    "rates": {
                        name: row["accuracy"]
                        for name, row in fresh_runtime[str(seed)].items()
                    },
                }
            ),
            flush=True,
        )

    runtime_rows = [
        *known_runtime.values(),
        *(row for seed_rows in fresh_runtime.values() for row in seed_rows.values()),
    ]
    gates = {
        "candidate_sha_pinned": sha256_bytes(candidate_bytes) == EXPECTED_CANDIDATE_SHA256,
        "full_strict_truthful_static_standard": structure["pass"],
        "no_lookup_cloak_or_giant": structure["no_lookup_cloak_or_giant"],
        "candidate_cost68": bool(
            candidate_profile == {"memory": 0, "params": 68, "cost": 68}
            and candidate_official
            and candidate_official.get("cost") == 68
        ),
        "authority_cost70_correct": bool(
            authority_profile["cost"] == 70
            and authority_official
            and authority_official.get("cost") == 70
            and authority_official.get("correct")
        ),
        "known_all_cases_evaluated": all(
            row["total"] == len(known_examples) for row in known_runtime.values()
        ),
        "fresh_each10000_all_configs": all(
            row["total"] == FRESH_PER_SEED
            for seed_rows in fresh_runtime.values()
            for row in seed_rows.values()
        ),
        "all_accuracy_at_least_90": all(row["policy90"] for row in runtime_rows),
        "errors_nonfinite_shape_smallpositive_zero": all(runtime_row_pass(row) for row in runtime_rows),
        "sign_stable_across_four_configs": all(
            len({rows[name]["sign_sha256"] for name, _o, _t in CONFIGS}) == 1
            for rows in [known_runtime, *fresh_runtime.values()]
        ),
        "converter_crosschecks_exact": bool(
            known_meta["converter_crosscheck"]["exact"]
            and all(row["converter_crosscheck"]["exact"] for row in fresh_meta)
        ),
    }
    accepted = all(gates.values())
    payload = {
        "task": 7,
        "lane": "agent_review_task007_policy90_278",
        "decision": "PASS_POLICY90_INDEPENDENT_REVIEW" if accepted else "FAIL_CLOSED",
        "accepted": accepted,
        "independence": {
            "root277_evidence_used_as_runtime_input": False,
            "candidate_binary_only_from_archive_lane": True,
            "fresh_seeds": list(FRESH_SEEDS),
            "kimi_used": False,
        },
        "candidate": {
            "path": relative(CANDIDATE),
            "sha256": sha256_bytes(candidate_bytes),
            "file_bytes": len(candidate_bytes),
            "profile": candidate_profile,
            "official_profile": candidate_official,
            "strict_lower_by": authority_profile["cost"] - candidate_profile["cost"],
            "score_gain": math.log(authority_profile["cost"] / candidate_profile["cost"]),
        },
        "authority": {
            "zip": relative(AUTHORITY_ZIP),
            "zip_sha256": sha256(AUTHORITY_ZIP),
            "member": "task007.onnx",
            "member_sha256": sha256_bytes(authority_bytes),
            "profile": authority_profile,
            "official_profile": authority_official,
        },
        "structure": structure,
        "known": {**known_meta, "runtime": known_runtime},
        "fresh": {
            "per_seed": FRESH_PER_SEED,
            "metadata": fresh_meta,
            "runtime_by_seed": fresh_runtime,
        },
        "runtime_configs": [
            {
                "name": name,
                "optimization": (
                    "ORT_DISABLE_ALL"
                    if optimization == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    else "ORT_ENABLE_ALL"
                ),
                "threads": threads,
            }
            for name, optimization, threads in CONFIGS
        ],
        "gates": gates,
        "aggregate": {
            "dataset_config_evaluations": len(runtime_rows),
            "runtime_case_executions": sum(row["total"] for row in runtime_rows),
            "errors": sum(row["errors"] for row in runtime_rows),
            "nonfinite_cases": sum(row["nonfinite_cases"] for row in runtime_rows),
            "nonfinite_elements": sum(row["nonfinite_elements"] for row in runtime_rows),
            "shape_mismatches": sum(row["shape_mismatches"] for row in runtime_rows),
            "small_positive_elements_0_to_0_25": sum(
                row["small_positive_elements_0_to_0_25"] for row in runtime_rows
            ),
            "sign_mismatch_cases": sum(
                row["sign_mismatch_cases_vs_disable_threads1"] for row in runtime_rows
            ),
            "sign_mismatch_cells": sum(
                row["sign_mismatch_cells_vs_disable_threads1"] for row in runtime_rows
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "normal_policy90": True,
            "private_zero_lineage": False,
            "lookup_or_shape_cloak": False,
            "root_or_others71407_written": False,
            "candidate_promoted": False,
            "kimi_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "accepted": accepted,
                "profiles": {"candidate": candidate_profile, "authority": authority_profile},
                "known": {name: row["accuracy"] for name, row in known_runtime.items()},
                "fresh": {
                    seed: {name: row["accuracy"] for name, row in rows.items()}
                    for seed, rows in fresh_runtime.items()
                },
                "aggregate": payload["aggregate"],
                "output": relative(OUTPUT),
            },
            indent=2,
        ),
        flush=True,
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
