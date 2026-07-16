#!/usr/bin/env python3
"""Independent POLICY90 audit of the quarantined task344 cost-132 model."""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import random
import re
import runpy
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "others/71407/PROBE_ONLY_DO_NOT_MERGE/task344_cost132.onnx.quarantine"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_d90796e8.py"
RAW_SOLVER = ROOT / "inputs/sakana-gcg-2025/raw/task344.py"
KNOWN_JSON = ROOT / "inputs/neurogolf-2026/task344.json"
PRIVATE_ZERO_CATALOG = ROOT / "docs/golf/private_zero_tasks.md"
PRIOR_WITNESS = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task344_margin_184/audit/margin_counterexample.json"
)

EXPECTED_CANDIDATE_SHA256 = "c5272a42bee419008a15d14bea734a6fb15956a863ad8e702deac0f02fcea5f6"
EXPECTED_AUTHORITY_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_AUTHORITY_SHA256 = "05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c"
FRESH_SEEDS = (344902091, 344902173)
FRESH_COUNT = int(os.environ.get("FRESH_COUNT", "20000"))

MODES = (
    ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from scripts.lib import scoring  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tensor_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    dims: list[int | str | None] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": dims,
    }


def structural_audit(data: bytes) -> dict[str, Any]:
    model = onnx.load_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    infos = [tensor_info(value) for value in values]
    static_positive = all(
        all(isinstance(dim, int) and dim > 0 for dim in info["shape"])
        for info in infos
    )
    output_names = {value.name for value in inferred.graph.output}
    intermediate_names = [
        output
        for node in inferred.graph.node
        for output in node.output
        if output and output not in output_names
    ]
    banned = {
        "LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"
    }
    lookup = {
        "ARGMAX", "GATHER", "GATHERELEMENTS", "GATHERND", "SCATTER",
        "SCATTERELEMENTS", "SCATTERND", "TFIDFVECTORIZER", "HARDMAX",
    }
    nested = [
        {"node": node.name, "attribute": attr.name}
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    initializers = []
    initializer_nonfinite = 0
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        initializer_nonfinite += nonfinite
        initializers.append(
            {
                "name": init.name,
                "dtype": onnx.TensorProto.DataType.Name(init.data_type),
                "shape": list(init.dims),
                "elements": int(array.size),
                "nonfinite": nonfinite,
            }
        )
    conv_ub: list[dict[str, Any]] = []
    init_map = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose"} or len(node.input) < 3:
            continue
        weight = init_map.get(node.input[1])
        bias = init_map.get(node.input[2])
        if weight is None or bias is None:
            conv_ub.append({"node": node.name, "reason": "dynamic_weight_or_bias"})
            continue
        expected = int(weight.dims[0]) if node.op_type == "Conv" else int(weight.dims[1])
        actual = math.prod(bias.dims)
        if actual != expected:
            conv_ub.append(
                {"node": node.name, "expected_bias": expected, "actual_bias": actual}
            )
    ops = [node.op_type for node in model.graph.node]
    domains = sorted({node.domain for node in model.graph.node})
    return {
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "node_count": len(model.graph.node),
        "ops": ops,
        "node_domains": domains,
        "standard_domain_only": all(domain in {"", "ai.onnx"} for domain in domains),
        "node_inputs": [len(node.input) for node in model.graph.node],
        "einsum_equations": [
            onnx.helper.get_attribute_value(attr).decode()
            for node in model.graph.node
            if node.op_type == "Einsum"
            for attr in node.attribute
            if attr.name == "equation"
        ],
        "graph_inputs": [tensor_info(item) for item in inferred.graph.input],
        "graph_outputs": [tensor_info(item) for item in inferred.graph.output],
        "intermediate_names": intermediate_names,
        "intermediate_count": len(intermediate_names),
        "all_inferred_shapes": infos,
        "all_shapes_static_positive": static_positive,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_ops": [op for op in ops if op.upper() in banned or "SEQUENCE" in op.upper()],
        "lookup_ops": [op for op in ops if op.upper() in lookup],
        "initializers": initializers,
        "initializer_nonfinite": initializer_nonfinite,
        "conv_bias_ub_findings": conv_ub,
    }


def make_session(data: bytes, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def exact_generator_rule(grid: list[list[int]]) -> list[list[int]]:
    """Exact row-major mutable-output loop used by task_d90796e8.generate."""
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] != 3:
                continue
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = row + dr, col + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if result[rr][cc] != 2:
                    continue
                result[rr][cc] = 0
                result[row][col] = 8
    return result


def simultaneous_support_rule(grid: list[list[int]]) -> list[list[int]]:
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] != 3:
                continue
            touched = False
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and grid[rr][cc] == 2:
                    result[rr][cc] = 0
                    touched = True
            if touched:
                result[row][col] = 8
    return result


def support_pair_degrees(grid: list[list[int]]) -> tuple[int, int]:
    height, width = len(grid), len(grid[0])
    red_max = green_max = 0
    for row in range(height):
        for col in range(width):
            if grid[row][col] not in {2, 3}:
                continue
            other = 3 if grid[row][col] == 2 else 2
            count = 0
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = row + dr, col + dc
                count += int(
                    0 <= rr < height and 0 <= cc < width and grid[rr][cc] == other
                )
            if grid[row][col] == 2:
                red_max = max(red_max, count)
            else:
                green_max = max(green_max, count)
    return red_max, green_max


def normalized_raw_solver(raw_solver: Any, grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in raw_solver([row[:] for row in grid])]


def empty_metrics() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "wrong_cells": 0,
        "errors": 0,
        "nonfinite": 0,
        "near_positive_0_lt_x_lt_0_25": 0,
        "near_positive_true_channel": 0,
        "near_positive_false_channel": 0,
        "min_true_channel_raw": float("inf"),
        "min_positive_true_channel_raw": float("inf"),
        "max_false_channel_raw": -float("inf"),
        "min_abs_nonzero": float("inf"),
        "runtime_shapes": set(),
        "first_failure": None,
    }


def update_metrics(
    metrics: dict[str, Any],
    raw: np.ndarray,
    expected: np.ndarray,
    index: int,
    example: dict[str, Any],
    capture_full_failure: bool,
) -> None:
    metrics["runtime_shapes"].add(tuple(int(item) for item in raw.shape))
    metrics["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
    predicted = raw > 0
    mismatch = predicted != expected
    cells = int(np.count_nonzero(mismatch))
    metrics["wrong_cells"] += cells
    if cells:
        metrics["wrong"] += 1
        if metrics["first_failure"] is None:
            indices = np.argwhere(mismatch)[:24]
            record: dict[str, Any] = {
                "index": index,
                "wrong_cells": cells,
                "mismatch_indices": indices.tolist(),
                "mismatch_raw": [float(raw[tuple(item)]) for item in indices],
            }
            if capture_full_failure:
                record["input"] = example["input"]
                record["truth_output"] = example["output"]
            metrics["first_failure"] = record
    else:
        metrics["right"] += 1
    true_values = raw[expected]
    false_values = raw[~expected]
    metrics["min_true_channel_raw"] = min(
        metrics["min_true_channel_raw"], float(true_values.min())
    )
    positive_true = true_values[true_values > 0]
    if positive_true.size:
        metrics["min_positive_true_channel_raw"] = min(
            metrics["min_positive_true_channel_raw"], float(positive_true.min())
        )
    metrics["max_false_channel_raw"] = max(
        metrics["max_false_channel_raw"], float(false_values.max())
    )
    near = (raw > 0) & (raw < 0.25)
    metrics["near_positive_0_lt_x_lt_0_25"] += int(np.count_nonzero(near))
    metrics["near_positive_true_channel"] += int(np.count_nonzero(near & expected))
    metrics["near_positive_false_channel"] += int(np.count_nonzero(near & ~expected))
    absolute = np.abs(raw)
    nonzero = absolute[absolute > 0]
    if nonzero.size:
        metrics["min_abs_nonzero"] = min(
            metrics["min_abs_nonzero"], float(nonzero.min())
        )


def finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["total"] = result["right"] + result["wrong"] + result["errors"]
    result["accuracy"] = result["right"] / result["total"] if result["total"] else 0.0
    result["runtime_shapes"] = [list(item) for item in sorted(result["runtime_shapes"])]
    for key in (
        "min_true_channel_raw",
        "min_positive_true_channel_raw",
        "max_false_channel_raw",
        "min_abs_nonzero",
    ):
        if not math.isfinite(result[key]):
            result[key] = None
    return result


def audit_examples(
    examples: list[dict[str, Any]],
    candidate_session: ort.InferenceSession,
    authority_session: ort.InferenceSession,
    capture_full_failure: bool,
) -> dict[str, Any]:
    candidate = empty_metrics()
    authority = empty_metrics()
    divergence_examples = divergence_cells = 0
    max_abs_raw_delta = 0.0
    first_divergence = None
    for index, example in enumerate(examples):
        bench = scoring.convert_to_numpy(example)
        if bench is None:
            candidate["errors"] += 1
            authority["errors"] += 1
            continue
        expected = bench["output"] > 0
        try:
            candidate_raw = candidate_session.run(
                ["output"], {"input": bench["input"]}
            )[0]
        except Exception as exc:  # fail closed and retain type only
            candidate["errors"] += 1
            if candidate["first_failure"] is None:
                candidate["first_failure"] = {"index": index, "error": type(exc).__name__}
            continue
        try:
            authority_raw = authority_session.run(
                ["output"], {"input": bench["input"]}
            )[0]
        except Exception as exc:
            authority["errors"] += 1
            if authority["first_failure"] is None:
                authority["first_failure"] = {"index": index, "error": type(exc).__name__}
            continue
        update_metrics(candidate, candidate_raw, expected, index, example, capture_full_failure)
        update_metrics(authority, authority_raw, expected, index, example, capture_full_failure)
        divergence = (candidate_raw > 0) != (authority_raw > 0)
        count = int(np.count_nonzero(divergence))
        divergence_cells += count
        if count:
            divergence_examples += 1
            if first_divergence is None:
                indices = np.argwhere(divergence)[:24]
                first_divergence = {
                    "index": index,
                    "cells": count,
                    "indices": indices.tolist(),
                    "candidate_raw": [float(candidate_raw[tuple(item)]) for item in indices],
                    "authority_raw": [float(authority_raw[tuple(item)]) for item in indices],
                }
        max_abs_raw_delta = max(
            max_abs_raw_delta, float(np.max(np.abs(candidate_raw - authority_raw)))
        )
    return {
        "candidate": finish_metrics(candidate),
        "authority": finish_metrics(authority),
        "candidate_authority_sign_divergence_examples": divergence_examples,
        "candidate_authority_sign_divergence_cells": divergence_cells,
        "candidate_authority_first_divergence": first_divergence,
        "max_abs_raw_delta": max_abs_raw_delta,
    }


def generate_stream(generator: Any, raw_solver: Any, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.seed(seed)
    examples: list[dict[str, Any]] = []
    input_hashes: set[str] = set()
    exact_rule_wrong = raw_solver_wrong = simultaneous_wrong = support_degree_violations = 0
    max_red_degree = max_green_degree = 0
    dims: set[tuple[int, int]] = set()
    colors: set[int] = set()
    for _ in range(FRESH_COUNT):
        example = generator.generate()
        examples.append(example)
        grid = example["input"]
        height, width = len(grid), len(grid[0])
        dims.add((height, width))
        colors.update(cell for row in grid for cell in row)
        input_hashes.add(sha256_bytes(json.dumps(grid, separators=(",", ":")).encode()))
        exact_rule_wrong += int(exact_generator_rule(grid) != example["output"])
        raw_solver_wrong += int(normalized_raw_solver(raw_solver, grid) != example["output"])
        simultaneous_wrong += int(simultaneous_support_rule(grid) != example["output"])
        red_degree, green_degree = support_pair_degrees(grid)
        max_red_degree = max(max_red_degree, red_degree)
        max_green_degree = max(max_green_degree, green_degree)
        support_degree_violations += int(red_degree > 1 or green_degree > 1)
    return examples, {
        "seed": seed,
        "count": FRESH_COUNT,
        "unique_input_hashes": len(input_hashes),
        "input_hashes": input_hashes,
        "dimensions_seen": [list(item) for item in sorted(dims)],
        "colors_seen": sorted(colors),
        "exact_generator_rule_wrong": exact_rule_wrong,
        "raw_solver_wrong": raw_solver_wrong,
        "simultaneous_support_rule_wrong": simultaneous_wrong,
        "support_degree_violations": support_degree_violations,
        "max_red_adjacent_green_degree": max_red_degree,
        "max_green_adjacent_red_degree": max_green_degree,
    }


def repeat_determinism(
    cases: list[dict[str, Any]], candidate_data: bytes, authority_data: bytes
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, data in (("candidate", candidate_data), ("authority", authority_data)):
        model_result: dict[str, Any] = {}
        cross_mode: dict[str, set[str]] = {f"case_{index}": set() for index in range(len(cases))}
        for mode_name, level in MODES:
            session = make_session(data, level)
            rows = []
            for index, example in enumerate(cases):
                bench = scoring.convert_to_numpy(example)
                hashes = []
                for _ in range(5):
                    raw = session.run(["output"], {"input": bench["input"]})[0]
                    hashes.append(sha256_bytes(raw.tobytes()))
                cross_mode[f"case_{index}"].add(hashes[0])
                rows.append(
                    {"case": index, "repeat_count": 5, "unique_raw_hashes": len(set(hashes))}
                )
            model_result[mode_name] = rows
        model_result["cross_mode_unique_raw_hashes"] = {
            key: len(value) for key, value in cross_mode.items()
        }
        result[model_name] = model_result
    return result


def main() -> None:
    candidate_data = CANDIDATE.read_bytes()
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_data = archive.read("task344.onnx")
    assert sha256_bytes(candidate_data) == EXPECTED_CANDIDATE_SHA256
    assert sha256_path(AUTHORITY_ZIP) == EXPECTED_AUTHORITY_ZIP_SHA256
    assert sha256_bytes(authority_data) == EXPECTED_AUTHORITY_SHA256

    known = json.loads(KNOWN_JSON.read_text())
    known_examples = [
        example for split in ("train", "test", "arc-gen") for example in known[split]
    ]
    generator = importlib.import_module("task_d90796e8")
    raw_solver = runpy.run_path(str(RAW_SOLVER))["p"]
    known_semantics = {
        "count": len(known_examples),
        "exact_generator_rule_wrong": sum(
            exact_generator_rule(example["input"]) != example["output"]
            for example in known_examples
        ),
        "raw_solver_wrong": sum(
            normalized_raw_solver(raw_solver, example["input"]) != example["output"]
            for example in known_examples
        ),
        "simultaneous_support_rule_wrong": sum(
            simultaneous_support_rule(example["input"]) != example["output"]
            for example in known_examples
        ),
        "support_degree_violations": sum(
            max(support_pair_degrees(example["input"])) > 1 for example in known_examples
        ),
    }

    print(f"generating fresh streams: seeds={FRESH_SEEDS}, n={FRESH_COUNT}", flush=True)
    streams: dict[int, list[dict[str, Any]]] = {}
    stream_semantics: dict[int, dict[str, Any]] = {}
    for seed in FRESH_SEEDS:
        examples, semantics = generate_stream(generator, raw_solver, seed)
        streams[seed] = examples
        stream_semantics[seed] = semantics
        print(f"generated seed={seed}", flush=True)
    overlap = len(
        stream_semantics[FRESH_SEEDS[0]]["input_hashes"]
        & stream_semantics[FRESH_SEEDS[1]]["input_hashes"]
    )
    for semantics in stream_semantics.values():
        semantics.pop("input_hashes")

    official_work = HERE / "official_profile_work"
    official_candidate = scoring.score_and_verify(
        onnx.load_from_string(candidate_data), 344, str(official_work),
        label="policy90_candidate", require_correct=True,
    )
    official_authority = scoring.score_and_verify(
        onnx.load_from_string(authority_data), 344, str(official_work),
        label="authority", require_correct=True,
    )

    def run_mode(
        mode_name: str, level: ort.GraphOptimizationLevel
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        print(f"runtime mode={mode_name} known", flush=True)
        candidate_session = make_session(candidate_data, level)
        authority_session = make_session(authority_data, level)
        mode_row: dict[str, Any] = {
            "known": audit_examples(
                known_examples, candidate_session, authority_session,
                capture_full_failure=False,
            ),
            "fresh": {},
        }
        full_failure = None
        for seed in FRESH_SEEDS:
            print(f"runtime mode={mode_name} seed={seed}", flush=True)
            capture = mode_name == "disable_all" and full_failure is None
            row = audit_examples(
                streams[seed], candidate_session, authority_session,
                capture_full_failure=capture,
            )
            mode_row["fresh"][str(seed)] = row
            if capture and row["candidate"]["first_failure"] is not None:
                full_failure = {
                    "seed": seed,
                    **row["candidate"]["first_failure"],
                }
        return mode_name, mode_row, full_failure

    runtime: dict[str, Any] = {}
    full_candidate_failure = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODES)) as executor:
        futures = [executor.submit(run_mode, name, level) for name, level in MODES]
        for future in concurrent.futures.as_completed(futures):
            mode_name, mode_row, failure = future.result()
            runtime[mode_name] = mode_row
            if mode_name == "disable_all":
                full_candidate_failure = failure

    prior = json.loads(PRIOR_WITNESS.read_text())
    witness_input = prior["witness"]["grid"]
    witness_example = {
        "input": witness_input,
        "output": exact_generator_rule(witness_input),
    }
    witness_runtime: dict[str, Any] = {}
    for mode_name, level in MODES:
        candidate_session = make_session(candidate_data, level)
        authority_session = make_session(authority_data, level)
        witness_runtime[mode_name] = audit_examples(
            [witness_example], candidate_session, authority_session,
            capture_full_failure=True,
        )

    repeat_cases = [
        known_examples[0], known_examples[-1], streams[FRESH_SEEDS[0]][0],
        streams[FRESH_SEEDS[1]][0], witness_example,
    ]
    if full_candidate_failure and "input" in full_candidate_failure:
        repeat_cases.append(
            {
                "input": full_candidate_failure["input"],
                "output": full_candidate_failure["truth_output"],
            }
        )
    determinism = repeat_determinism(repeat_cases, candidate_data, authority_data)

    private_catalog_text = PRIVATE_ZERO_CATALOG.read_text()
    result = {
        "task": 344,
        "verdict_basis": "POLICY90_NORMAL_TASK",
        "environment": {
            "python": sys.version.split()[0],
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "fresh_seeds": list(FRESH_SEEDS),
            "fresh_count_per_seed": FRESH_COUNT,
            "modes": [name for name, _ in MODES],
        },
        "authority": {
            "archive": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "archive_sha256": sha256_path(AUTHORITY_ZIP),
            "member_sha256": sha256_bytes(authority_data),
            "bytes": len(authority_data),
            "official_profile": official_authority,
            "structure": structural_audit(authority_data),
        },
        "candidate": {
            "source": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256_bytes(candidate_data),
            "bytes": len(candidate_data),
            "official_profile": official_candidate,
            "structure": structural_audit(candidate_data),
        },
        "gain": {
            "cost_delta": official_authority["cost"] - official_candidate["cost"],
            "score_delta": math.log(official_authority["cost"] / official_candidate["cost"]),
        },
        "private_zero_lookup_cloak": {
            "catalog": str(PRIVATE_ZERO_CATALOG.relative_to(ROOT)),
            "task_number_literal_present": bool(re.search(r"(?<!\d)344(?!\d)", private_catalog_text)),
            "confirmed_private_zero": False,
            "authority_is_lb_white": True,
            "candidate_lookup_ops": structural_audit(candidate_data)["lookup_ops"],
            "candidate_shape_cloak": False,
            "candidate_rule_type": "normal local Type-A approximation",
        },
        "generator": {
            "path": str(GENERATOR.relative_to(ROOT)),
            "sha256": sha256_path(GENERATOR),
            "raw_solver_path": str(RAW_SOLVER.relative_to(ROOT)),
            "raw_solver_sha256": sha256_path(RAW_SOLVER),
            "known_json_sha256": sha256_path(KNOWN_JSON),
            "known_semantics": known_semantics,
            "fresh_semantics": {str(key): value for key, value in stream_semantics.items()},
            "cross_seed_duplicate_input_hashes": overlap,
        },
        "runtime": runtime,
        "candidate_first_truth_counterexample": full_candidate_failure,
        "known_reachable_authority_candidate_witness": {
            "support": prior["generator"]["support_witness"],
            "input": witness_input,
            "truth_output": witness_example["output"],
            "runtime": witness_runtime,
        },
        "determinism": determinism,
    }
    print("FINAL_JSON")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
