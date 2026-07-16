#!/usr/bin/env python3
"""Independent, fail-closed review of the task192 selected-mask factorization.

This lane never promotes or edits a submission.  It independently re-measures
the immutable authority, the prior exact-polynomial control, and the proposed
factorized candidate.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8008.14.zip"
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_task192_exact111/task192_selected_masks.onnx"
EXACT_POLY = ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/candidates/task192_exact_poly.onnx"

EXPECTED = {
    "authority_zip_sha256": "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6",
    "authority_task192_sha256": "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c",
    "candidate_sha256": "40244ab462644481407ebb7200984dfdff1475c0d8e6ff731ba2d588ec92ea09",
    "exact_poly_sha256": "c3cbaf44d962ca72e15514da1b32c121ee489d153ef39d38b7101f09576e92b6",
    "authority_cost": 1609,
    "candidate_cost": 1197,
    "exact_poly_cost": 1307,
}

# These seeds are deliberately distinct from the prior exact-poly audit's
# 192800661 and 192930007 streams.
FRESH_SEEDS = (112192071, 112192072)
FRESH_COUNT = 2000
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    tensor = value.type.tensor_type
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in tensor.shape.dim]


def attribute(node: onnx.NodeProto, name: str) -> Any:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    raise KeyError(f"{node.name or node.op_type} lacks attribute {name}")


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}


def official_cost(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="review192_112_", dir="/tmp") as work:
        path = Path(work) / f"{label}.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def static_review(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "full_check": False,
        "strict_data_prop": False,
        "truthful_declared_output": False,
        "standard_domains": False,
        "finite_initializers": False,
        "conv_bias_ub0": False,
    }
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_data_prop_error"] = f"{type(exc).__name__}: {exc}"
        inferred = model

    standard = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    finite = all(
        array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
        for array in initializer_map(model).values()
    )
    findings: list[Any]
    try:
        findings = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        findings = [{"check_error": f"{type(exc).__name__}: {exc}"}]

    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    nonstatic: list[str] = []
    for node in inferred.graph.node:
        for name in node.output:
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                nonstatic.append(name)
                continue
            shape = dims(value)
            if any(dim is None or dim <= 0 for dim in shape):
                nonstatic.append(name)

    output_shape = dims(inferred.graph.output[0])
    result.update(
        {
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "standard_domains": standard,
            "finite_initializers": finite,
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nonstatic_node_outputs": sorted(set(nonstatic)),
            "all_node_outputs_static_positive": not nonstatic,
            "declared_output_shape": output_shape,
            "truthful_declared_output": output_shape == [1, 10, 30, 30],
            "conv_bias_findings": findings,
            "conv_bias_ub0": not findings,
        }
    )
    result["pass"] = all(
        (
            result["full_check"],
            result["strict_data_prop"],
            result["truthful_declared_output"],
            result["standard_domains"],
            result["finite_initializers"],
            result["conv_bias_ub0"],
            result["all_node_outputs_static_positive"],
            result["functions"] == 0,
            result["sparse_initializers"] == 0,
        )
    )
    return result


def factorization_review(exact_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    exact = onnx.load_model_from_string(exact_data)
    candidate = onnx.load_model_from_string(candidate_data)
    exact_init = initializer_map(exact)
    candidate_init = initializer_map(candidate)

    exact_output = exact.graph.node[-1]
    candidate_output = candidate.graph.node[-1]
    selected_concat = next(node for node in candidate.graph.node if node.output == ["selected_masks"])
    common_names = ("adj", "color_masks", "hist_select", "route", "background", "depth", "onehot_values")
    common_equal = {
        name: name in exact_init and name in candidate_init and np.array_equal(exact_init[name], candidate_init[name])
        for name in common_names
    }

    relation = exact_init.get("relation")
    all_colors = candidate_init.get("all_colors")
    per_color: list[dict[str, Any]] = []
    if relation is not None and all_colors is not None:
        for selected_color in range(10):
            selected = np.zeros((10,), dtype=np.float32)
            selected[selected_color] = 1.0
            contracted = np.einsum("rda,a->rd", relation, selected)
            factored = np.concatenate((all_colors, selected[None, :]), axis=0)
            per_color.append(
                {
                    "selected_color": selected_color,
                    "equal": bool(np.array_equal(contracted, factored)),
                    "max_abs_difference": float(np.max(np.abs(contracted - factored))),
                }
            )

    exact_equation = attribute(exact_output, "equation").decode("utf-8")
    candidate_equation = attribute(candidate_output, "equation").decode("utf-8")
    graph_contract = {
        "exact_equation": exact_equation,
        "candidate_equation": candidate_equation,
        "exact_relation_used_twice": list(exact_output.input).count("relation") == 2,
        "exact_selected_used_twice": list(exact_output.input).count("selected") == 2,
        "candidate_selected_masks_used_twice": list(candidate_output.input).count("selected_masks") == 2,
        "concat_inputs": list(selected_concat.input),
        "concat_axis": int(attribute(selected_concat, "axis")),
        "relation_removed": "relation" not in candidate_init,
        "all_colors_added": all_colors is not None and list(all_colors.shape) == [1, 10],
    }
    graph_contract["pass"] = all(
        (
            graph_contract["exact_equation"]
            == "bchw,rc,bdhq,rda,za,qw,bepw,ref,zf,ph,ru,uo->bohw",
            graph_contract["candidate_equation"]
            == "bchw,rc,bdhq,rd,qw,bepw,re,ph,ru,uo->bohw",
            graph_contract["exact_relation_used_twice"],
            graph_contract["exact_selected_used_twice"],
            graph_contract["candidate_selected_masks_used_twice"],
            graph_contract["concat_inputs"] == ["all_colors", "selected"],
            graph_contract["concat_axis"] == 0,
            graph_contract["relation_removed"],
            graph_contract["all_colors_added"],
        )
    )
    proof = {
        "identity": (
            "For selected one-hot s, sum_a relation[0,d,a]*s[a] = sum_a s[a] = 1, "
            "while sum_a relation[1,d,a]*s[a] = sum_a I[d,a]*s[a] = s[d]. "
            "Thus contracting relation with selected is exactly Concat(all_colors, selected)."
        ),
        "relation_shape": None if relation is None else list(relation.shape),
        "all_colors_shape": None if all_colors is None else list(all_colors.shape),
        "all_ten_onehot_selections": per_color,
        "all_onehot_equal": len(per_color) == 10 and all(row["equal"] for row in per_color),
        "shared_initializer_equal": common_equal,
        "graph_contract": graph_contract,
    }
    proof["pass"] = (
        proof["relation_shape"] == [2, 10, 10]
        and proof["all_colors_shape"] == [1, 10]
        and proof["all_onehot_equal"]
        and all(common_equal.values())
        and graph_contract["pass"]
    )
    return proof


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def run_one(session: ort.InferenceSession, benchmark: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: benchmark["input"]},
        )[0]
    )


def empty_runtime_row(total: int) -> dict[str, Any]:
    return {
        "total": total,
        "candidate_right": 0,
        "exact_poly_right": 0,
        "authority_right": 0,
        "reference_rule_right": 0,
        "runtime_errors": {"candidate": 0, "exact_poly": 0, "authority": 0},
        "nonfinite_values": {"candidate": 0, "exact_poly": 0, "authority": 0},
        "candidate_vs_exact_raw_equal": 0,
        "candidate_vs_exact_threshold_equal": 0,
        "candidate_vs_authority_raw_equal": 0,
        "candidate_vs_authority_threshold_equal": 0,
        "candidate_vs_exact_max_abs_difference": 0.0,
        "candidate_vs_authority_max_abs_difference": 0.0,
        "first_authority_threshold_difference": None,
        "candidate_min_positive": None,
        "candidate_max_nonpositive": -math.inf,
        "output_shapes": {"candidate": [], "exact_poly": [], "authority": []},
        "first_failure": None,
    }


def task192_rule(grid: list[list[int]]) -> list[list[int]]:
    counts = [sum(row.count(color) for row in grid) for color in range(10)]
    selected = max(range(1, 10), key=lambda color: counts[color])
    height, width = len(grid), len(grid[0])
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            horizontal = any(
                grid[row][other] == selected
                for other in range(max(0, col - 1), min(width, col + 2))
            )
            vertical = any(
                grid[other][col] == selected
                for other in range(max(0, row - 1), min(height, row + 2))
            )
            if grid[row][col] != 0 and horizontal and vertical:
                output[row][col] = selected
    return output


def update_row(
    row: dict[str, Any],
    example: dict[str, Any],
    benchmark: dict[str, np.ndarray],
    sessions: dict[str, ort.InferenceSession],
    case: dict[str, Any],
) -> None:
    row["reference_rule_right"] += int(task192_rule(example["input"]) == example["output"])
    expected = benchmark["output"].astype(bool)
    outputs: dict[str, np.ndarray] = {}
    for name, session in sessions.items():
        try:
            raw = run_one(session, benchmark)
        except Exception as exc:  # noqa: BLE001
            row["runtime_errors"][name] += 1
            row["first_failure"] = row["first_failure"] or {
                **case,
                "model": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        outputs[name] = raw
        shape = list(raw.shape)
        if shape not in row["output_shapes"][name]:
            row["output_shapes"][name].append(shape)
        finite = np.isfinite(raw)
        row["nonfinite_values"][name] += int(raw.size - np.count_nonzero(finite))
        row[f"{name}_right"] += int(np.array_equal(raw > 0, expected))

    candidate = outputs.get("candidate")
    exact = outputs.get("exact_poly")
    authority = outputs.get("authority")
    if candidate is not None:
        positives = candidate[expected]
        negatives = candidate[~expected]
        if positives.size:
            value = float(positives.min())
            row["candidate_min_positive"] = (
                value if row["candidate_min_positive"] is None else min(row["candidate_min_positive"], value)
            )
        if negatives.size:
            row["candidate_max_nonpositive"] = max(row["candidate_max_nonpositive"], float(negatives.max()))
    if candidate is not None and exact is not None:
        raw_equal = np.array_equal(candidate, exact)
        threshold_equal = np.array_equal(candidate > 0, exact > 0)
        row["candidate_vs_exact_raw_equal"] += int(raw_equal)
        row["candidate_vs_exact_threshold_equal"] += int(threshold_equal)
        row["candidate_vs_exact_max_abs_difference"] = max(
            row["candidate_vs_exact_max_abs_difference"],
            float(np.max(np.abs(candidate - exact), initial=0.0)),
        )
        if not (raw_equal and threshold_equal):
            row["first_failure"] = row["first_failure"] or {
                **case,
                "comparison": "candidate_vs_exact_poly",
                "raw_equal": bool(raw_equal),
                "threshold_equal": bool(threshold_equal),
            }
    if candidate is not None and authority is not None:
        row["candidate_vs_authority_raw_equal"] += int(np.array_equal(candidate, authority))
        threshold_equal = np.array_equal(candidate > 0, authority > 0)
        row["candidate_vs_authority_threshold_equal"] += int(threshold_equal)
        row["candidate_vs_authority_max_abs_difference"] = max(
            row["candidate_vs_authority_max_abs_difference"],
            float(np.max(np.abs(candidate - authority), initial=0.0)),
        )
        if not threshold_equal:
            row["first_authority_threshold_difference"] = row[
                "first_authority_threshold_difference"
            ] or {
                **case,
                "comparison": "candidate_vs_authority_threshold",
            }


def finalize_runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    total = row["total"]
    required_counts = (
        row["candidate_right"],
        row["exact_poly_right"],
        row["reference_rule_right"],
        row["candidate_vs_exact_raw_equal"],
        row["candidate_vs_exact_threshold_equal"],
    )
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())
    row["perfect"] = (
        all(value == total for value in required_counts)
        and row["runtime_errors_total"] == 0
        and row["nonfinite_values_total"] == 0
        and row["candidate_min_positive"] is not None
        and row["candidate_min_positive"] > 0
        and row["candidate_max_nonpositive"] <= 0
    )
    return row


def reclassify_recorded_runtime(report: dict[str, Any]) -> dict[str, Any]:
    """Reapply gates to already-recorded outputs without rerunning inference."""
    for row in report.get("configs", {}).values():
        failure = row.get("first_failure")
        if failure and failure.get("comparison") == "candidate_vs_authority_threshold":
            row["first_authority_threshold_difference"] = failure
            row["first_failure"] = None
        finalize_runtime_row(row)
    for row in report.get("rows", []):
        failure = row.get("first_failure")
        if failure and failure.get("comparison") == "candidate_vs_authority_threshold":
            row["first_authority_threshold_difference"] = failure
            row["first_failure"] = None
        finalize_runtime_row(row)
    if "configs" in report:
        report["pass"] = all(row.get("perfect", False) for row in report["configs"].values())
    elif "rows" in report:
        report["pass"] = len(report["rows"]) == 4 and all(
            row.get("perfect", False) for row in report["rows"]
        )
    return report


def known_four(candidate: bytes, exact: bytes, authority: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(192)
    ordered = [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[split])
    ]
    report: dict[str, Any] = {"known_total": len(ordered), "configs": {}}
    for disable, threads, label in CONFIGS:
        row = empty_runtime_row(len(ordered))
        try:
            sessions = {
                "candidate": make_session(candidate, disable, threads),
                "exact_poly": make_session(exact, disable, threads),
                "authority": make_session(authority, disable, threads),
            }
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["perfect"] = False
            report["configs"][label] = row
            continue
        for split, index, example in ordered:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                row["first_failure"] = row["first_failure"] or {
                    "split": split,
                    "index": index,
                    "error": "convert_to_numpy returned None",
                }
                continue
            update_row(row, example, benchmark, sessions, {"split": split, "index": index})
        finalize_runtime_row(row)
        report["configs"][label] = row
        print(
            f"known {label}: candidate={row['candidate_right']}/{row['total']} "
            f"exact_raw={row['candidate_vs_exact_raw_equal']}/{row['total']} "
            f"authority_threshold={row['candidate_vs_authority_threshold_equal']}/{row['total']} "
            f"perfect={row['perfect']}",
            flush=True,
        )
    report["pass"] = all(row.get("perfect", False) for row in report["configs"].values())
    return report


def fresh_dual(candidate: bytes, exact: bytes, authority: bytes) -> dict[str, Any]:
    generator = importlib.import_module("task_7e0986d6")
    rows: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS:
        per_mode: dict[str, tuple[dict[str, Any], dict[str, ort.InferenceSession] | None]] = {}
        for disable, label in ((True, "disable_all"), (False, "default")):
            row = empty_runtime_row(FRESH_COUNT)
            row.update({"seed": seed, "mode": label})
            try:
                sessions = {
                    "candidate": make_session(candidate, disable, 1),
                    "exact_poly": make_session(exact, disable, 1),
                    "authority": make_session(authority, disable, 1),
                }
            except Exception as exc:  # noqa: BLE001
                row["session_error"] = f"{type(exc).__name__}: {exc}"
                row["perfect"] = False
                sessions = None
            per_mode[label] = (row, sessions)

        random.seed(seed)
        for index in range(FRESH_COUNT):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            for label, (row, sessions) in per_mode.items():
                if benchmark is None or sessions is None:
                    row["first_failure"] = row["first_failure"] or {
                        "index": index,
                        "error": (
                            "convert_to_numpy returned None"
                            if benchmark is None
                            else "session construction failed"
                        ),
                    }
                    continue
                update_row(row, example, benchmark, sessions, {"index": index})
            if (index + 1) % 500 == 0:
                print(f"fresh seed={seed}: generated_and_checked={index + 1}/{FRESH_COUNT}", flush=True)

        for label, (row, sessions) in per_mode.items():
            finalize_runtime_row(row)
            rows.append(row)
            print(
                f"fresh seed={seed} mode={label}: candidate={row['candidate_right']}/{FRESH_COUNT} "
                f"exact_raw={row['candidate_vs_exact_raw_equal']}/{FRESH_COUNT} "
                f"authority_threshold={row['candidate_vs_authority_threshold_equal']}/{FRESH_COUNT} "
                f"perfect={row['perfect']}",
                flush=True,
            )
    return {
        "seeds": list(FRESH_SEEDS),
        "count_per_seed_per_mode": FRESH_COUNT,
        "prior_audit_seeds_excluded": [192800661, 192930007],
        "rows": rows,
        "pass": len(rows) == 4 and all(row.get("perfect", False) for row in rows),
    }


def runtime_shape_trace(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    existing = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name not in typed or name in names:
                continue
            names.append(name)
            if name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    benchmark = scoring.convert_to_numpy(scoring.load_examples(192)["train"][0])
    if benchmark is None:
        raise RuntimeError("first train example is not convertible")
    arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches: list[dict[str, Any]] = []
    nonfinite = 0
    traced_shapes: dict[str, list[int]] = {}
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        actual = list(value.shape)
        traced_shapes[name] = actual
        declared = dims(typed[name])
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "traced_shapes": traced_shapes,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-recorded-runtime",
        action="store_true",
        help="Reuse this lane's completed known/fresh raw counts and only reapply gates.",
    )
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    authority_zip_data = AUTHORITY.read_bytes()
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task192.onnx")
    candidate_data = CANDIDATE.read_bytes()
    exact_data = EXACT_POLY.read_bytes()

    hashes = {
        "authority_zip_sha256": sha256(authority_zip_data),
        "authority_task192_sha256": sha256(authority_data),
        "candidate_sha256": sha256(candidate_data),
        "exact_poly_sha256": sha256(exact_data),
    }
    hash_pass = all(hashes[name] == EXPECTED[name] for name in hashes)
    costs = {
        "authority": official_cost(authority_data, "authority_task192"),
        "exact_poly": official_cost(exact_data, "exact_poly_task192"),
        "candidate": official_cost(candidate_data, "selected_masks_task192"),
    }
    cost_pass = (
        costs["authority"]["cost"] == EXPECTED["authority_cost"]
        and costs["exact_poly"]["cost"] == EXPECTED["exact_poly_cost"]
        and costs["candidate"]["cost"] == EXPECTED["candidate_cost"]
        and costs["candidate"]["cost"] < costs["exact_poly"]["cost"] < costs["authority"]["cost"]
    )
    static = {
        "candidate": static_review(candidate_data),
        "exact_poly": static_review(exact_data),
        "authority": static_review(authority_data),
    }
    factorization = factorization_review(exact_data, candidate_data)
    if args.reuse_recorded_runtime:
        prior = json.loads((HERE / "review.json").read_text(encoding="utf-8"))
        if prior.get("hashes") != hashes:
            raise RuntimeError("recorded runtime hashes do not match current immutable inputs")
        known = reclassify_recorded_runtime(prior["known_four_configs"])
        fresh = reclassify_recorded_runtime(prior["fresh"])
        if fresh.get("seeds") != list(FRESH_SEEDS) or fresh.get("count_per_seed_per_mode") != FRESH_COUNT:
            raise RuntimeError("recorded fresh configuration does not match review constants")
    else:
        known = known_four(candidate_data, exact_data, authority_data)
        fresh = fresh_dual(candidate_data, exact_data, authority_data)
    try:
        trace = runtime_shape_trace(candidate_data)
    except Exception as exc:  # noqa: BLE001
        trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}

    gates = {
        "immutable_hashes": hash_pass,
        "official_costs": cost_pass,
        "candidate_static": bool(static["candidate"].get("pass")),
        "exact_poly_static_control": bool(static["exact_poly"].get("pass")),
        "factorization_math_and_graph": bool(factorization.get("pass")),
        "known_four_configs": bool(known.get("pass")),
        "fresh_two_new_seeds_dual_ort": bool(fresh.get("pass")),
        "runtime_shape_truthful": bool(trace.get("truthful")),
    }
    accepted = all(gates.values())
    review = {
        "review_scope": "independent fail-closed task192 selected-mask factorization",
        "runtime_evidence_mode": (
            "reclassified immediately preceding full independent run after immutable hash and config checks"
            if args.reuse_recorded_runtime
            else "full independent execution"
        ),
        "paths": {
            "authority": rel(AUTHORITY),
            "candidate": rel(CANDIDATE),
            "exact_poly": rel(EXACT_POLY),
        },
        "expected": EXPECTED,
        "hashes": hashes,
        "costs": costs,
        "cost_reduction_vs_authority": costs["authority"]["cost"] - costs["candidate"]["cost"],
        "cost_reduction_vs_exact_poly": costs["exact_poly"]["cost"] - costs["candidate"]["cost"],
        "static": static,
        "factorization": factorization,
        "known_four_configs": known,
        "fresh": fresh,
        "runtime_shape_trace": trace,
        "gates": gates,
        "verdict": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "authority_equivalence_note": (
            "Candidate-vs-authority raw and threshold equality are informational because the authority is an "
            "approximate prior implementation. Candidate-vs-exact-poly raw and threshold equality, generator/reference "
            "correctness, zero runtime errors, and finite outputs are mandatory."
        ),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    review_path = HERE / "review.json"
    review_path.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "lane": rel(HERE),
        "review": rel(review_path),
        "verdict": review["verdict"],
        "accepted": accepted,
        "candidate": {
            "path": rel(CANDIDATE),
            "sha256": hashes["candidate_sha256"],
            "official_cost": costs["candidate"],
        },
        "comparators": {
            "authority": {
                "path": rel(AUTHORITY),
                "zip_sha256": hashes["authority_zip_sha256"],
                "task192_sha256": hashes["authority_task192_sha256"],
                "official_cost": costs["authority"],
            },
            "exact_poly": {
                "path": rel(EXACT_POLY),
                "sha256": hashes["exact_poly_sha256"],
                "official_cost": costs["exact_poly"],
            },
        },
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_count_per_seed_per_mode": FRESH_COUNT,
        "gates": gates,
        "promotion_performed": False,
        "runtime_evidence_mode": review["runtime_evidence_mode"],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"verdict={review['verdict']} candidate_cost={costs['candidate']['cost']} "
        f"vs_exact={costs['exact_poly']['cost']} vs_authority={costs['authority']['cost']} gates={gates}",
        flush=True,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
