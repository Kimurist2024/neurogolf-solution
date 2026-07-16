#!/usr/bin/env python3
"""Exhaustively audit the finite generator support of task254 candidate cost42."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_b12/candidates/task254_r01_static42.onnx"
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
DATA = ROOT / "inputs/neurogolf-2026/task254.json"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_a61f2674.py"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
EXPECTED_SUPPORT_COUNTS = {"offset0_n4": 3024, "offset0_n5": 15120, "offset1_n4": 3024}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def iter_support() -> Iterator[tuple[str, int, tuple[int, ...]]]:
    values = tuple(range(1, 10))
    for offset, n, label in (
        (0, 4, "offset0_n4"),
        (0, 5, "offset0_n5"),
        (1, 4, "offset1_n4"),
    ):
        for heights in itertools.permutations(values, n):
            yield label, offset, heights


def encode(offset: int, heights: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Encode exact generate(vals=heights, offset=offset, size=9) I/O."""
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    y = np.zeros_like(x)
    x[0, 0, :9, :9] = 1.0
    y[0, 0, :9, :9] = 1.0
    low = min(heights)
    high = max(heights)
    for index, height in enumerate(heights):
        col = 2 * index + offset
        rows = slice(9 - height, 9)
        x[0, 0, rows, col] = 0.0
        x[0, 5, rows, col] = 1.0
        if height == low:
            y[0, 0, rows, col] = 0.0
            y[0, 2, rows, col] = 1.0
        elif height == high:
            y[0, 0, rows, col] = 0.0
            y[0, 1, rows, col] = 1.0
    return x, y


def structural_audit(model: onnx.ModelProto) -> dict[str, Any]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    all_values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    dynamic = {
        value.name: dims(value)
        for value in all_values
        if any(not isinstance(d, int) or d <= 0 for d in dims(value))
    }
    nested = [
        {"node": node.op_type, "attribute": attr.name}
        for node in inferred.graph.node
        for attr in node.attribute
        if attr.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}
    ]
    nonfinite_initializers = []
    initializers = []
    for init in inferred.graph.initializer:
        array = numpy_helper.to_array(init)
        finite = bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True
        if not finite:
            nonfinite_initializers.append(init.name)
        initializers.append(
            {
                "name": init.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "finite": finite,
                "sha256": sha256_bytes(array.tobytes()),
            }
        )
    custom_domains = sorted(
        item.domain for item in inferred.opset_import if item.domain not in {"", "ai.onnx"}
    )
    banned = sorted(
        {
            node.op_type
            for node in inferred.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    lookup = sorted(
        {node.op_type for node in inferred.graph.node if node.op_type in {"TfIdfVectorizer", "Hardmax"}}
    )
    node_rows = []
    for index, node in enumerate(inferred.graph.node):
        attrs: dict[str, Any] = {}
        for attr in node.attribute:
            value = helper.get_attribute_value(attr)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            attrs[attr.name] = value
        node_rows.append(
            {
                "index": index,
                "op": node.op_type,
                "domain": node.domain,
                "input_count": len(node.input),
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": attrs,
            }
        )
    return {
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "standard_domains": not custom_domains,
        "custom_domains": custom_domains,
        "functions": len(inferred.functions),
        "sparse_initializers": len(inferred.graph.sparse_initializer),
        "nested_graph_attributes": nested,
        "banned_or_sequence_ops": banned,
        "dynamic_or_nonpositive_shapes": dynamic,
        "nonfinite_initializers": nonfinite_initializers,
        "lookup_ops": lookup,
        "op_histogram": dict(Counter(node.op_type for node in inferred.graph.node)),
        "input_shapes": {item.name: dims(item) for item in inferred.graph.input},
        "output_shapes": {item.name: dims(item) for item in inferred.graph.output},
        "initializers": initializers,
        "nodes": node_rows,
    }


def make_sessions(model: onnx.ModelProto) -> dict[str, ort.InferenceSession]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected candidate")
    sessions: dict[str, ort.InferenceSession] = {}
    for graph_mode, graph_level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        for threads in (1, 4):
            options = ort.SessionOptions()
            options.graph_optimization_level = graph_level
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = threads
            options.log_severity_level = 4
            name = f"{graph_mode}_threads{threads}"
            sessions[name] = ort.InferenceSession(
                sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
    return sessions


def empty_mode_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_cases": 0,
        "nonfinite_elements": 0,
        "near_positive_elements_0_lt_x_lt_0_25": 0,
        "minimum_true_cell_raw": math.inf,
        "maximum_false_cell_raw": -math.inf,
        "minimum_positive_raw": math.inf,
        "maximum_absolute_raw": 0.0,
        "output_shapes": Counter(),
        "first_failure": None,
        "minimum_true_witness": None,
        "maximum_false_witness": None,
    }


def witness(label: str, offset: int, heights: tuple[int, ...], value: float) -> dict[str, Any]:
    return {"support_class": label, "offset": offset, "vals": list(heights), "raw": value}


def exhaustive(sessions: dict[str, ort.InferenceSession]) -> dict[str, Any]:
    stats = {name: empty_mode_stats() for name in sessions}
    support_counts: Counter[str] = Counter()
    total = 0
    start = time.monotonic()
    for label, offset, heights in iter_support():
        support_counts[label] += 1
        total += 1
        x, y = encode(offset, heights)
        truth = y > 0
        for name, session in sessions.items():
            row = stats[name]
            try:
                raw = np.asarray(session.run(["output"], {"input": x})[0])
            except Exception as exc:  # fail closed and retain complete support traversal
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "kind": "runtime",
                        "support_class": label,
                        "offset": offset,
                        "vals": list(heights),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue
            row["output_shapes"][str(list(raw.shape))] += 1
            finite = np.isfinite(raw)
            nonfinite_count = int(np.count_nonzero(~finite))
            if nonfinite_count:
                row["nonfinite_cases"] += 1
                row["nonfinite_elements"] += nonfinite_count
            mask = raw > 0.0
            if np.array_equal(mask, truth):
                row["right"] += 1
            else:
                row["wrong"] += 1
                if row["first_failure"] is None:
                    coords = np.argwhere(mask != truth)
                    row["first_failure"] = {
                        "kind": "wrong_mask",
                        "support_class": label,
                        "offset": offset,
                        "vals": list(heights),
                        "mismatch_count": int(coords.shape[0]),
                        "first_mismatch": coords[0].tolist() if coords.size else None,
                    }
            finite_raw = raw[finite]
            if finite_raw.size:
                row["maximum_absolute_raw"] = max(
                    row["maximum_absolute_raw"], float(np.max(np.abs(finite_raw)))
                )
                positives = finite_raw[finite_raw > 0]
                if positives.size:
                    row["minimum_positive_raw"] = min(
                        row["minimum_positive_raw"], float(np.min(positives))
                    )
            row["near_positive_elements_0_lt_x_lt_0_25"] += int(
                np.count_nonzero((raw > 0.0) & (raw < 0.25))
            )
            on_values = raw[truth]
            off_values = raw[~truth]
            local_min_on = float(np.min(on_values))
            local_max_off = float(np.max(off_values))
            if local_min_on < row["minimum_true_cell_raw"]:
                row["minimum_true_cell_raw"] = local_min_on
                row["minimum_true_witness"] = witness(label, offset, heights, local_min_on)
            if local_max_off > row["maximum_false_cell_raw"]:
                row["maximum_false_cell_raw"] = local_max_off
                row["maximum_false_witness"] = witness(label, offset, heights, local_max_off)
        if total % 500 == 0:
            elapsed = time.monotonic() - start
            print(f"support {total}/21168 elapsed={elapsed:.1f}s", flush=True)

    serializable: dict[str, Any] = {}
    for name, row in stats.items():
        row["output_shapes"] = dict(row["output_shapes"])
        for key in ("minimum_true_cell_raw", "maximum_false_cell_raw", "minimum_positive_raw"):
            if not math.isfinite(row[key]):
                row[key] = None
        row["total"] = row["right"] + row["wrong"] + row["runtime_errors"]
        row["perfect_mask"] = row["right"] == total and row["wrong"] == 0 and row["runtime_errors"] == 0
        row["finite_all"] = row["nonfinite_cases"] == 0
        row["positive_margin_ge_0_25"] = (
            row["minimum_true_cell_raw"] is not None and row["minimum_true_cell_raw"] >= 0.25
        )
        row["false_cells_nonpositive"] = (
            row["maximum_false_cell_raw"] is not None and row["maximum_false_cell_raw"] <= 0.0
        )
        serializable[name] = row
    return {
        "support_derivation": {
            "size": 9,
            "heights": "ordered samples without replacement from range(1,10)",
            "classes": EXPECTED_SUPPORT_COUNTS,
            "formula": "P(9,4)+P(9,5)+P(9,4)",
            "total": 21168,
            "generator_default_path": "offset randint(0,1); offset=1 => num=4; offset=0 => num randint(4,5)",
        },
        "observed_support_counts": dict(support_counts),
        "observed_total": total,
        "modes": serializable,
        "elapsed_seconds": time.monotonic() - start,
    }


def known_corpus(sessions: dict[str, ort.InferenceSession]) -> dict[str, Any]:
    payload = json.loads(DATA.read_text())
    result = {}
    for name, session in sessions.items():
        right = wrong = errors = nonfinite = near = 0
        first_failure = None
        split_counts: Counter[str] = Counter()
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(payload[split]):
                split_counts[split] += 1
                bench = scoring.convert_to_numpy(example)
                assert bench is not None
                try:
                    raw = np.asarray(session.run(["output"], {"input": bench["input"]})[0])
                    nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
                    near += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                    if np.array_equal(raw > 0, bench["output"] > 0):
                        right += 1
                    else:
                        wrong += 1
                        if first_failure is None:
                            first_failure = {"split": split, "index": index, "kind": "wrong"}
                except Exception as exc:
                    errors += 1
                    if first_failure is None:
                        first_failure = {
                            "split": split,
                            "index": index,
                            "kind": "runtime",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        result[name] = {
            "right": right,
            "wrong": wrong,
            "runtime_errors": errors,
            "total": right + wrong + errors,
            "split_counts": dict(split_counts),
            "nonfinite_elements": nonfinite,
            "near_positive_elements_0_lt_x_lt_0_25": near,
            "first_failure": first_failure,
        }
    return result


def profile_cost(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def main() -> None:
    candidate_data = CANDIDATE.read_bytes()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_data = archive.read("task254.onnx")
    with tempfile.TemporaryDirectory(prefix="task254_deep70_", dir="/tmp") as tmp:
        base_path = Path(tmp) / "task254.onnx"
        base_path.write_bytes(baseline_data)
        baseline_cost = profile_cost(base_path)
    candidate_cost = profile_cost(CANDIDATE)
    model = onnx.load(CANDIDATE)
    structure = structural_audit(model)
    sessions = make_sessions(model)
    known = known_corpus(sessions)
    support = exhaustive(sessions)
    all_runtime_modes_perfect = all(
        row["perfect_mask"]
        and row["finite_all"]
        and row["positive_margin_ge_0_25"]
        and row["false_cells_nonpositive"]
        and row["near_positive_elements_0_lt_x_lt_0_25"] == 0
        for row in support["modes"].values()
    )
    structural_pass = (
        structure["checker_full"]
        and structure["strict_shape_inference_data_prop"]
        and structure["standard_domains"]
        and not structure["functions"]
        and not structure["sparse_initializers"]
        and not structure["nested_graph_attributes"]
        and not structure["banned_or_sequence_ops"]
        and not structure["dynamic_or_nonpositive_shapes"]
        and not structure["nonfinite_initializers"]
        and not structure["lookup_ops"]
    )
    result = {
        "task": 254,
        "generator_hash": "a61f2674",
        "generator_path": str(GENERATOR.relative_to(ROOT)),
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256_bytes(candidate_data),
            "file_bytes": len(candidate_data),
            "official_profiled_cost": candidate_cost,
        },
        "baseline": {
            "zip": BASE_ZIP.name,
            "zip_sha256": sha256_bytes(BASE_ZIP.read_bytes()),
            "member": "task254.onnx",
            "member_sha256": sha256_bytes(baseline_data),
            "official_profiled_cost": baseline_cost,
        },
        "score_gain": math.log(baseline_cost["cost"] / candidate_cost["cost"]),
        "structure": structure,
        "known_corpus_four_modes": known,
        "exhaustive_parameter_support": support,
        "gate": {
            "candidate_strictly_cheaper": candidate_cost["cost"] < baseline_cost["cost"],
            "structural_pass_except_explicit_giant_einsum_exception": structural_pass,
            "giant_einsum_exception_basis": "finite generator support exhaustively evaluated in four ORT configurations",
            "all_21168_cases_four_modes_perfect": all_runtime_modes_perfect,
            "lookup0": not structure["lookup_ops"],
            "ub0": not structure["nonfinite_initializers"] and support["observed_total"] == 21168,
            "truthful_output_shape": all(
                set(row["output_shapes"]) == {"[1, 10, 30, 30]"} for row in support["modes"].values()
            ),
        },
    }
    result["gate"]["all_pass"] = all(
        bool(value)
        for key, value in result["gate"].items()
        if key not in {"giant_einsum_exception_basis"}
    )
    (HERE / "exhaustive_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gate": result["gate"], "score_gain": result["score_gain"]}, indent=2))


if __name__ == "__main__":
    main()
