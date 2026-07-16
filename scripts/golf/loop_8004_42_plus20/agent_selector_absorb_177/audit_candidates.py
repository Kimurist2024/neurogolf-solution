#!/usr/bin/env python3
"""Fail-closed audit for selector absorption and exact factor alternatives."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline"
CONTROLS = HERE / "audit_controls"
AUDIT = HERE / "audit"
TRACES = HERE / "traces"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ROOT_SCORES = ROOT / "all_scores.csv"
EXPECTED_AUTHORITY = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_SCORES = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
EXPECTED_MEMBERS = {
    246: "9d9428878051ec1c327999b5774946b6f4a84fbb6d4a875e192185a8c966c362",
    335: "79da8462ed32fe2ea46677637f51923cd6e4abc31fe94e7b816e3599aeba0d57",
    348: "b21fdf675e2415c203bdf5e578f2221c296dc3ee1d46f9fe80c2d2e820b6f2d5",
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TfIdfVectorizer", "CategoryMapper", "LabelEncoder"}

sys.path.insert(0, str(ROOT))
from scripts.lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_guard() -> dict[str, Any]:
    result = {
        "authority": sha256(AUTHORITY),
        "root_submission": sha256(ROOT_SUBMISSION),
        "all_scores": sha256(ROOT_SCORES),
    }
    result["authority_unchanged"] = result["authority"] == EXPECTED_AUTHORITY
    result["root_submission_unchanged"] = result["root_submission"] == EXPECTED_AUTHORITY
    result["all_scores_unchanged"] = result["all_scores"] == EXPECTED_SCORES
    if not all(result[key] for key in result if key.endswith("_unchanged")):
        raise RuntimeError(f"protected root artifact changed: {result}")
    return result


def exact_rank(matrix: np.ndarray) -> int:
    """Gaussian-elimination rank over exact binary-rational float values."""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(array.shape)
    work = [
        [Fraction.from_float(float(value)) for value in row]
        for row in array.astype(np.float64)
    ]
    rows, cols = array.shape
    rank = 0
    for col in range(cols):
        pivot = next((row for row in range(rank, rows) if work[row][col]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][col]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(rows):
            if row == rank or not work[row][col]:
                continue
            factor = work[row][col]
            work[row] = [a - factor * b for a, b in zip(work[row], work[rank])]
        rank += 1
        if rank == rows:
            break
    return rank


def initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def initializer_audit(task: int, model: onnx.ModelProto) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in initializers(model).items():
        row: dict[str, Any] = {
            "name": name,
            "shape": list(value.shape),
            "elements": int(value.size),
            "nonzero": int(np.count_nonzero(value)),
        }
        if value.ndim == 2:
            rank = exact_rank(value)
            m, n = value.shape
            row.update(
                exact_matrix_rank=rank,
                dense_elements=int(m * n),
                exact_rank_factor_elements=int(m * rank + rank * n),
                factor_delta=int(m * rank + rank * n - m * n),
            )
            if task == 348 and name == "D":
                row["singular_values_for_diagnostic_only"] = [
                    float(item) for item in np.linalg.svd(value.astype(np.float64), compute_uv=False)
                ]
                row["distinct_columns"] = len({tuple(column) for column in value.T})
        elif value.ndim == 3:
            mode_ranks = []
            for axis in range(3):
                unfolded = np.moveaxis(value, axis, 0).reshape(value.shape[axis], -1)
                mode_ranks.append(exact_rank(unfolded))
            row["exact_mode_ranks"] = mode_ranks
            row["rank1_possible"] = all(rank == 1 for rank in mode_ranks)
            row["rank1_cp_elements"] = int(sum(value.shape))
            row["rank2_cp_elements"] = int(2 * sum(value.shape))
        rows.append(row)
    return rows


def params(model: onnx.ModelProto) -> int:
    value = scoring.calculate_params(model)
    if value is None:
        raise RuntimeError("unscorable parameters")
    return int(value)


def nested_graph_count(model: onnx.ModelProto) -> int:
    return sum(
        1
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}
    )


def dim_list(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    checker = strict = None
    inferred = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = repr(exc)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        strict = True
    except Exception as exc:  # noqa: BLE001
        strict = repr(exc)
    nodes = [node.op_type for node in model.graph.node]
    other_outputs = [
        output
        for node in model.graph.node
        for output in node.output
        if output and output != "output"
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "checker_full": checker,
        "strict_inference": strict,
        "nodes": nodes,
        "node_count": len(nodes),
        "initializer_count": len(model.graph.initializer),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "function_count": len(model.functions),
        "nested_graph_count": nested_graph_count(model),
        "domains": sorted({item.domain for item in model.opset_import}),
        "banned_ops": sorted({item for item in nodes if item.upper() in BANNED or "Sequence" in item}),
        "lookup_ops": sorted({item for item in nodes if item in LOOKUP}),
        "params": params(model),
        "non_output_intermediates": other_outputs,
        "inferred_input_shape": dim_list(inferred.graph.input[0]) if inferred else None,
        "inferred_output_shape": dim_list(inferred.graph.output[0]) if inferred else None,
    }


def make_session(model: onnx.ModelProto, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def examples(task: int) -> list[dict[str, Any]]:
    loaded = scoring.load_examples(task)
    return [*loaded["train"], *loaded["test"], *loaded["arc-gen"]]


def known_audit(task: int, path: Path, level: ort.GraphOptimizationLevel) -> dict[str, Any]:
    model = onnx.load(path)
    try:
        session = make_session(model, level)
    except Exception as exc:  # noqa: BLE001
        return {"right": 0, "wrong": 0, "errors": 1, "session_error": repr(exc)}
    right = wrong = errors = nonfinite = 0
    runtime_shapes: set[tuple[int, ...]] = set()
    for example in examples(task):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            raw = session.run(["output"], {"input": benchmark["input"]})[0]
            runtime_shapes.add(tuple(int(item) for item in raw.shape))
            nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
            if np.array_equal(raw > 0, benchmark["output"] > 0):
                right += 1
            else:
                wrong += 1
        except Exception:  # noqa: BLE001
            errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": right + wrong + errors,
        "nonfinite_values": nonfinite,
        "runtime_shapes": [list(item) for item in sorted(runtime_shapes)],
    }


def differential(task: int, base: Path, candidate: Path) -> dict[str, Any]:
    base_session = make_session(onnx.load(base), ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    cand_session = make_session(onnx.load(candidate), ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    total = bit_equal = sign_equal = errors = nonfinite = 0
    max_abs_delta = 0.0
    for example in examples(task):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            left = base_session.run(["output"], {"input": benchmark["input"]})[0]
            right = cand_session.run(["output"], {"input": benchmark["input"]})[0]
            total += 1
            bit_equal += int(np.array_equal(left, right))
            sign_equal += int(np.array_equal(left > 0, right > 0))
            nonfinite += int(np.count_nonzero(~np.isfinite(right)))
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                max_abs_delta = max(max_abs_delta, float(np.max(np.abs(left[finite] - right[finite]))))
        except Exception:  # noqa: BLE001
            errors += 1
    return {
        "total": total,
        "raw_bitwise_equal": bit_equal,
        "threshold_equal": sign_equal,
        "errors": errors,
        "nonfinite_values": nonfinite,
        "max_abs_finite_delta": max_abs_delta,
    }


def task_from_name(path: Path) -> int:
    return int(path.name[4:7])


def model_audit(path: Path, baseline_cost: int) -> dict[str, Any]:
    task = task_from_name(path)
    structure_result = structure(path)
    disabled = known_audit(task, path, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    default = known_audit(task, path, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    profile = scoring.score_and_verify(
        onnx.load(path),
        task,
        str(TRACES),
        label=path.stem,
        require_correct=False,
    )
    actual_cost = profile["cost"] if profile else None
    return {
        "task": task,
        "structure": structure_result,
        "known_disable_all": disabled,
        "known_default": default,
        "official_like_profile": profile,
        "baseline_cost": baseline_cost,
        "strictly_lower": actual_cost is not None and 0 < actual_cost < baseline_cost,
    }


def contraction_inventory() -> dict[str, Any]:
    return {
        "task246_task335": [
            {
                "family": "couple_three_S_bonds",
                "identity": "prod_j sum_n C[n,j]S[n] = sum_n S[n]prod_j C[n,j] because support(S)={2} and S[2]=-1",
                "old_params": 109,
                "new_params": 109,
                "delta": 0,
                "result": "exact_equal_cost_control",
            },
            {
                "family": "precompute_S_times_C",
                "removed_unique": {"S": 3},
                "retained_unique": {"C": 30},
                "added_unique": {"T=S@C": 10},
                "old_params": 109,
                "new_params": 116,
                "delta": 7,
                "result": "reject_higher",
            },
            {
                "family": "reuse_C_column_as_S",
                "task246_identity": "S == C[:,3]",
                "task335_identity": "S == C[:,8]",
                "required_operation": "fix the second C index to a constant color column",
                "minimum_dense_fixed_column_selector_or_extracted_vector": 3,
                "net_saving_upper_bound": 0,
                "note": "Einsum labels do not encode a literal index; identifying the color label with a dynamic input/output label changes the polynomial",
                "result": "no_lower_exact_reuse",
            },
            {
                "family": "remove_S_by_implicit_row_sum",
                "sum_over_color_of_each_C_row": [0.0, -1.0, 0.0],
                "S": [0.0, 0.0, -1.0],
                "exact_equal": False,
                "result": "reject_not_equivalent",
            },
            {
                "family": "precompute_duplicate_C_square",
                "removed_unique": {},
                "retained_unique": {"C": 30},
                "added_unique": {"Csq=C*C": 30},
                "old_params": 109,
                "new_params": 139,
                "delta": 30,
                "result": "reject_higher",
            },
            {
                "family": "dynamic_MatMul_S_C_then_reuse",
                "params": 109,
                "minimum_new_runtime_bytes": 40,
                "minimum_cost": 149,
                "delta": 40,
                "result": "reject_higher_memory",
            },
            {
                "family": "B_or_M01_rank_factorization",
                "direct_elements_each": 8,
                "exact_mode_ranks": [2, 2, 2],
                "rank1_elements": 6,
                "rank1_possible": False,
                "rank2_CP_elements": 12,
                "result": "no_lower_exact_factor",
            },
            {
                "family": "B_M01_local_precontraction",
                "smallest_pair_contraction_elements": 16,
                "source_unique_elements": 16,
                "sources_used_elsewhere": True,
                "net_delta_lower_bound": 16,
                "result": "reject_higher",
            },
        ],
        "task348": [
            {
                "family": "precompute_C1D_and_C2D",
                "removed_unique": {"D": 90, "C1": 6, "C2": 6},
                "added_unique": {"C1D": 60, "C2D": 60},
                "old_params": 130,
                "new_params": 148,
                "delta": 18,
                "result": "reject_higher",
            },
            {
                "family": "dynamic_MatMul_C1D_C2D_then_reuse",
                "params": 130,
                "minimum_new_runtime_bytes": 480,
                "minimum_cost": 610,
                "delta": 480,
                "result": "reject_higher_memory",
            },
            {
                "family": "exact_D_matrix_factorization",
                "shape": [3, 30],
                "exact_rank": 3,
                "direct_elements": 90,
                "factor_elements": 99,
                "delta": 9,
                "result": "no_lower_exact_factor",
            },
            {
                "family": "share_C1_for_C2_via_D_row_swap",
                "C2_equals_C1": False,
                "required_extra_row_permuted_D_elements": 90,
                "removed_C2_elements": 6,
                "delta": 84,
                "result": "reject_higher",
            },
            {
                "family": "D_tail_common_additive_decomposition",
                "distinct_columns": 10,
                "exact_rank": 3,
                "pure_product_lower_than_dense": False,
                "note": "an additive decomposition needs an explicit component bond or runtime Add; neither removes the rank-3 storage floor without extra parameters/memory",
                "result": "no_lower_exact_product",
            },
        ],
    }


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    guard_before = root_guard()

    baselines: dict[int, dict[str, Any]] = {}
    for task, expected in EXPECTED_MEMBERS.items():
        path = BASELINE / f"task{task:03d}.onnx"
        if sha256(path) != expected:
            raise RuntimeError(f"task{task:03d} baseline changed")
        baselines[task] = model_audit(path, {246: 109, 335: 109, 348: 130}[task])
        baselines[task]["initializer_audit"] = initializer_audit(task, onnx.load(path))

    controls: list[dict[str, Any]] = []
    for path in sorted(CONTROLS.glob("task*.onnx")):
        task = task_from_name(path)
        row = model_audit(path, {246: 109, 335: 109, 348: 130}[task])
        row["differential_vs_baseline"] = differential(
            task, BASELINE / f"task{task:03d}.onnx", path
        )
        row["classification"] = (
            "PRE_FRESH" if row["strictly_lower"] else "REJECT_NOT_STRICTLY_LOWER"
        )
        controls.append(row)

    eligible = [row for row in controls if row["strictly_lower"]]
    if eligible:
        raise RuntimeError("unexpected strict-lower control requires fresh>=10000 audit")

    payload = {
        "status": "COMPLETE_SAFE0",
        "authority_score": 8009.46,
        "root_guard_before": guard_before,
        "baselines": baselines,
        "controls": controls,
        "contraction_inventory": contraction_inventory(),
        "fresh_gate": {
            "required_per_seed": 10000,
            "eligible_strict_lower_candidates": 0,
            "fresh_runs": 0,
            "reason": "all exact rewrites failed the strict-lower actual-cost pre-gate",
        },
        "safe_candidates": [],
        "probe_only_candidates": [],
        "policy": {
            "sparse_initializer": "excluded: docs/golf/BANNED_STRUCTURES.md records grader ERROR",
            "private_zero_or_lookup": False,
            "shape_cloak": False,
            "full_strict_required": True,
        },
        "root_guard_after": root_guard(),
    }
    (HERE / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    (AUDIT / "selector_factor_audit.json").write_text(
        json.dumps(
            {
                "baselines": baselines,
                "controls": controls,
                "contraction_inventory": payload["contraction_inventory"],
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "status": "NO_WINNER",
                "authority_score": 8009.46,
                "safe_candidates": [],
                "probe_only_candidates": [],
                "integration_performed": False,
                "protected_files_modified": False,
                "reason": "no exact strict-lower actual-cost candidate",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": payload["status"], "controls": len(controls), "eligible": 0}))


if __name__ == "__main__":
    main()
