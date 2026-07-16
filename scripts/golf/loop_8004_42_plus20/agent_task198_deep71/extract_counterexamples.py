#!/usr/bin/env python3
"""Extract one legal task198 generator counterexample per lower candidate.

One legal counterexample is a complete disproof of finite-generator coverage.
The script also reruns that exact input in the four requested ORT
configuration combinations (disabled/default optimizations, 1/4 threads) and
records raw sign-margin and period-selector diagnostics.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HIGH47 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_high47"
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS))
from lib import scoring  # noqa: E402


def load_high47():
    path = HIGH47 / "fresh_reference_one.py"
    spec = importlib.util.spec_from_file_location("task198_high47_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session(model: onnx.ModelProto, disable: bool, threads: int):
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def run_one(sess, example):
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise RuntimeError("convert_to_numpy returned None")
    raw = sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: benchmark["input"]},
    )[0]
    expected = benchmark["output"] > 0
    predicted = raw > 0
    return raw, predicted, expected


def sign_stats(raw: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(raw)
    positives = raw[raw > 0]
    nonpositives = raw[raw <= 0]
    on = raw[expected]
    off = raw[~expected]
    return {
        "nonfinite": int(np.size(raw) - np.count_nonzero(finite)),
        "near_positive_0_to_0_25": int(np.count_nonzero((raw > 0) & (raw < 0.25))),
        "positive_min": float(np.min(positives)) if positives.size else None,
        "positive_max": float(np.max(positives)) if positives.size else None,
        "nonpositive_max": float(np.max(nonpositives)) if nonpositives.size else None,
        "expected_on_min": float(np.min(on)) if on.size else None,
        "expected_off_max": float(np.max(off)) if off.size else None,
    }


def infer_legal_parameters(example: dict) -> dict[str, object]:
    grid = np.asarray(example["input"], dtype=np.int8)
    h, w = grid.shape
    values, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(values) != 1:
        raise AssertionError((values, counts))
    color = int(values[0])
    row_scores = np.count_nonzero(grid == color, axis=1)
    line_rows = [int(x) for x in np.flatnonzero(row_scores >= w // 2)]
    if not line_rows:
        raise AssertionError("no line rows")
    minisize = line_rows[0]
    period = minisize + 1
    size = (h + 1) // period
    line_mask = (
        np.arange(h)[:, None] % period == minisize
    ) | (np.arange(w)[None, :] % period == minisize)
    holes = [list(map(int, rc)) for rc in np.argwhere(line_mask & (grid == 0))]
    regenerated = importlib.import_module("task_83302e8f").generate(
        size=size,
        minisize=minisize,
        color=color,
        rows=[r for r, _ in holes],
        cols=[c for _, c in holes],
    )
    if regenerated != example:
        raise AssertionError("inferred parameters do not exactly regenerate the example")
    return {
        "size": size,
        "minisize": minisize,
        "period": period,
        "color": color,
        "height": h,
        "width": w,
        "hole_count": len(holes),
        "allowed_hole_count_min": size + minisize,
        "allowed_hole_count_max": size * minisize,
        "holes": holes,
        "exact_regeneration": True,
    }


def intermediate_diagnostics(model: onnx.ModelProto, example: dict) -> dict[str, object]:
    diagnostic = copy.deepcopy(model)
    names = []
    for node in diagnostic.graph.node[:-1]:
        for name in node.output:
            names.append(name)
            diagnostic.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [3])
            )
    if not names:
        return {}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    # Do not call scoring.sanitize_model here: it intentionally collapses the
    # graph back to the official single output and would remove these temporary
    # diagnostic outputs.
    sess = ort.InferenceSession(
        diagnostic.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(example)
    assert benchmark is not None
    values = sess.run(
        None, {sess.get_inputs()[0].name: benchmark["input"]}
    )
    output_names = [item.name for item in sess.get_outputs()]
    result = {}
    for name in names:
        value = values[output_names.index(name)]
        result[name] = {
            "values": [float(x) for x in np.asarray(value).reshape(-1)],
            "argmax": int(np.argmax(value)),
        }
    return result


def load_official_evidence() -> dict[str, dict]:
    rows = {}
    for path in sorted((HIGH47 / "evidence").glob("task198_history_*.json")):
        row = json.loads(path.read_text())
        rows[row["sha256"]] = row
    return rows


def main() -> None:
    high47 = load_high47()
    generator = importlib.import_module("task_83302e8f")
    seed = 47_000_199
    random.seed(seed)
    examples = [generator.generate() for _ in range(1000)]
    reference = high47.solve_198
    assert all(reference(copy.deepcopy(x["input"])) == x["output"] for x in examples)

    evidence = load_official_evidence()
    results = []
    paths = sorted((HIGH47 / "candidates").glob("task198*.onnx"))
    for path in paths:
        model = onnx.load(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row = evidence[sha]
        base_session = session(model, True, 1)
        first = None
        for index, example in enumerate(examples):
            raw, predicted, expected = run_one(base_session, example)
            if not np.array_equal(predicted, expected):
                first = (index, example, raw, predicted, expected)
                break
        if first is None:
            raise AssertionError(f"expected a counterexample for {path}")
        index, example, raw, predicted, expected = first
        params = infer_legal_parameters(example)
        expected_period_index = params["minisize"] - 3
        mismatch = np.argwhere(predicted != expected)
        first_pos = [int(x) for x in mismatch[0]]

        configs = {}
        baseline_bits = np.packbits(predicted.reshape(-1)).tobytes()
        for threads in (1, 4):
            for mode, disable in (("disable_all", True), ("default", False)):
                raw2, pred2, expected2 = run_one(session(model, disable, threads), example)
                key = f"{mode}_threads{threads}"
                configs[key] = {
                    "correct": bool(np.array_equal(pred2, expected2)),
                    "wrong_bits": int(np.count_nonzero(pred2 != expected2)),
                    "same_threshold_output_as_disabled_threads1": bool(
                        np.packbits(pred2.reshape(-1)).tobytes() == baseline_bits
                    ),
                    "stats": sign_stats(raw2, expected2),
                }

        equations = []
        for node in model.graph.node:
            equation = None
            for attr in node.attribute:
                if attr.name == "equation":
                    equation = helper.get_attribute_value(attr).decode()
            equations.append(
                {
                    "op": node.op_type,
                    "arity": len(node.input),
                    "equation": equation,
                    "outputs": list(node.output),
                }
            )
        arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
        nonzero_abs = np.concatenate(
            [np.abs(x.reshape(-1))[np.abs(x.reshape(-1)) > 0] for x in arrays]
        )
        inter = intermediate_diagnostics(model, example)
        selector = inter.get("period_gate") or inter.get("period_raw") or {}
        results.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": sha,
                "official_like": row["official_like_score"],
                "known_disable_all_total": row["known_disable_all"]["total"],
                "known_default_total": row["known_default"]["total"],
                "full_check": row["full_check"],
                "strict_shape_data_prop": row["strict_shape_data_prop"],
                "runtime_shape_mismatches": row["runtime_shape_trace"]["declared_actual_mismatches"],
                "op_histogram": row["op_histogram"],
                "max_node_inputs": row["max_node_inputs"],
                "equations": equations,
                "initializer_numeric_range": {
                    "min_nonzero_abs": float(np.min(nonzero_abs)),
                    "max_abs": float(max(np.max(np.abs(x)) for x in arrays)),
                },
                "counterexample": {
                    "seed": seed,
                    "index": index,
                    "parameters": params,
                    "expected_period_index": expected_period_index,
                    "selector": selector,
                    "mismatch_bits": int(len(mismatch)),
                    "first_mismatch_nqrc": first_pos,
                    "first_expected": bool(expected[tuple(first_pos)]),
                    "first_predicted": bool(predicted[tuple(first_pos)]),
                    "first_raw": float(raw[tuple(first_pos)]),
                    "configurations": configs,
                },
            }
        )

    costs = Counter(row["official_like"]["cost"] for row in results)
    output = {
        "task": 198,
        "generator_hash": "83302e8f",
        "seed": seed,
        "examples_generated": len(examples),
        "reference_correct": len(examples),
        "candidate_count": len(results),
        "cost_histogram": {str(k): v for k, v in sorted(costs.items())},
        "all_candidates_have_legal_counterexample": len(results) == len(paths),
        "candidates": results,
    }
    out = HERE / "counterexamples.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out}: {len(results)} legal counterexamples")


if __name__ == "__main__":
    main()
