#!/usr/bin/env python3
"""Four-runtime exhaustive-support audit for the task328 exact553 candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import random
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = HERE / "candidates/task328_exact553_split_p0.onnx"
CONFIGS = (
    (True, 1, "disable_all_t1"),
    (True, 4, "disable_all_t4"),
    (False, 1, "default_t1"),
    (False, 4, "default_t4"),
)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def make_session(path: Path, disabled: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def canonical_cases() -> list[tuple[dict, dict[str, Any]]]:
    generator = importlib.import_module("task_d22278a0")
    cases = []
    for size in range(6, 19):
        corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        for count in range(2, 5):
            for selected in itertools.combinations(corners, count):
                rows, cols = zip(*selected)
                colors = tuple(range(1, count + 1))
                example = generator.generate(
                    size=size, rows=rows, cols=cols, colors=colors
                )
                cases.append(
                    (
                        example,
                        {
                            "size": size,
                            "corners": [list(item) for item in selected],
                            "colors": list(colors),
                        },
                    )
                )
    return cases


def known_cases() -> list[tuple[dict, dict[str, Any]]]:
    rows = []
    examples = scoring.load_examples(328)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            rows.append((example, {"split": split, "index": index}))
    return rows


def empty_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values_0_to_0_25": 0,
        "true_below_0_25": 0,
        "false_positive_values": 0,
        "min_positive": None,
        "max_false": None,
        "max_abs": 0.0,
        "first_failure": None,
    }


def audit_config(
    config: tuple[bool, int, str],
    cases: list[tuple[dict, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    disabled, threads, label = config
    session = make_session(CANDIDATE, disabled, threads)
    stats = empty_stats()
    for example, meta in cases:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            stats["runtime_errors"] += 1
            if stats["first_failure"] is None:
                stats["first_failure"] = {**meta, "error": "convert_to_numpy returned None"}
            continue
        expected = benchmark["output"].astype(bool)
        try:
            raw = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: benchmark["input"]},
            )[0]
        except Exception as exc:  # noqa: BLE001
            stats["runtime_errors"] += 1
            if stats["first_failure"] is None:
                stats["first_failure"] = {
                    **meta,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            continue
        finite = np.isfinite(raw)
        stats["nonfinite_values"] += int((~finite).sum())
        if np.any(finite):
            stats["max_abs"] = max(
                float(stats["max_abs"]), float(np.abs(raw[finite]).max())
            )
        positive = raw[finite & (raw > 0)]
        if positive.size:
            value = float(positive.min())
            stats["min_positive"] = (
                value
                if stats["min_positive"] is None
                else min(float(stats["min_positive"]), value)
            )
        near = finite & (raw > 0) & (raw < 0.25)
        stats["near_positive_values_0_to_0_25"] += int(near.sum())
        true_raw = raw[expected]
        false_raw = raw[~expected]
        stats["true_below_0_25"] += int(np.count_nonzero(true_raw < 0.25))
        stats["false_positive_values"] += int(np.count_nonzero(false_raw > 0))
        if false_raw.size and np.isfinite(false_raw).all():
            value = float(false_raw.max())
            stats["max_false"] = (
                value
                if stats["max_false"] is None
                else max(float(stats["max_false"]), value)
            )
        correct = np.array_equal(raw > 0, expected)
        stats["right" if correct else "wrong"] += 1
        if not correct and stats["first_failure"] is None:
            stats["first_failure"] = {
                **meta,
                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
            }
    stats["perfect_sign"] = bool(
        stats["right"] == len(cases)
        and stats["wrong"] == 0
        and stats["runtime_errors"] == 0
        and stats["nonfinite_values"] == 0
    )
    stats["strict_margin"] = bool(
        stats["perfect_sign"]
        and stats["near_positive_values_0_to_0_25"] == 0
        and stats["true_below_0_25"] == 0
    )
    print(
        label,
        stats["right"],
        stats["wrong"],
        "near", stats["near_positive_values_0_to_0_25"],
        "min", stats["min_positive"],
        flush=True,
    )
    return label, stats


def four_config(cases: list[tuple[dict, dict[str, Any]]]) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda cfg: audit_config(cfg, cases), CONFIGS))
    return dict(rows)


def fresh_orbit_coverage() -> list[dict[str, Any]]:
    generator = importlib.import_module("task_d22278a0")
    expected_keys = {
        (size, selected)
        for size in range(6, 19)
        for count in range(2, 5)
        for selected in itertools.combinations(range(4), count)
    }
    runs = []
    for seed in (328_175_101, 328_175_102):
        counts: Counter[tuple[int, tuple[int, ...]]] = Counter()
        invalid = 0
        for case in range(10_000):
            random.seed(seed + case)
            example = generator.generate()
            grid = np.asarray(example["input"])
            size = int(grid.shape[0])
            corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
            selected = tuple(index for index, rc in enumerate(corners) if grid[rc] != 0)
            colors = [int(grid[rc]) for rc in corners if grid[rc] != 0]
            key = (size, selected)
            if (
                key not in expected_keys
                or len(colors) not in (2, 3, 4)
                or len(set(colors)) != len(colors)
                or max(colors, default=0) > 9
            ):
                invalid += 1
            else:
                counts[key] += 1
        runs.append(
            {
                "seed": seed,
                "requested": 10_000,
                "mapped": int(sum(counts.values())),
                "invalid": invalid,
                "distinct_orbits_seen": len(counts),
                "all_seen_orbits_belong_to_exhaustive_set": invalid == 0,
            }
        )
    return runs


def shape_of(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def runtime_shapes(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name in typed:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(scoring.load_examples(328)["train"][0])
    assert benchmark is not None
    values = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    rows = [
        {
            "tensor": name,
            "declared": shape_of(typed[name]),
            "runtime": list(np.asarray(value).shape),
        }
        for name, value in zip(names, values)
        if shape_of(typed[name]) != list(np.asarray(value).shape)
    ]
    return {
        "traced_outputs": len(names),
        "mismatch_count": len(rows),
        "mismatches": rows,
    }


def structural() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    ops = Counter(node.op_type for node in model.graph.node)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    domains = {item.domain for item in model.opset_import} | {
        node.domain for node in model.graph.node
    }
    nested = [
        node.op_type
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    nonstatic = [value.name for value in values if shape_of(value) is None]
    color_axes = [
        {"name": name, "axis": axis}
        for name, array in arrays.items()
        for axis, width in enumerate(array.shape)
        if width == 10
    ]
    ssel = arrays["Ssel"]
    with tempfile.TemporaryDirectory(prefix="task328_exact175_", dir="/tmp") as workdir:
        temp = Path(workdir) / "candidate.onnx"
        temp.write_bytes(CANDIDATE.read_bytes())
        memory, params, cost = cost_of(str(temp))
    return {
        "sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "full_checker": True,
        "strict_data_prop": True,
        "nonstatic_tensors": nonstatic,
        "runtime_shapes": runtime_shapes(model),
        "op_histogram": dict(sorted(ops.items())),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "standard_domains": domains <= {"", "ai.onnx"},
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        ],
        "nested_graphs": nested,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or np.isfinite(array).all()
            for array in arrays.values()
        ),
        "conv_bias_findings": check_conv_bias(model),
        "lookup_ops": {
            name: ops.get(name, 0)
            for name in ("TfIdfVectorizer", "Gather", "GatherND", "ScatterND", "Hardmax")
            if ops.get(name, 0)
        },
        "profile": {"memory": int(memory), "params": int(params), "cost": int(cost)},
        "color_axis_initializers": color_axes,
        "nonzero_color_columns_identical": bool(
            np.all(ssel[:, 1:] == ssel[:, 1:2])
        ),
        "color_equivariance_proof": (
            "MaxPool is channelwise, ReduceL2 is channel-permutation invariant, "
            "and Ssel is the only initializer with a 10-color axis; its columns "
            "1..9 are identical. Therefore every permutation of nonzero colors "
            "permutes the free output color axis and preserves logits/signs."
        ),
    }


def main() -> None:
    support = canonical_cases()
    known = known_cases()
    result = {
        "task": 328,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "structural": structural(),
        "finite_support": {
            "orbit_representatives": len(support),
            "full_generator_states": 71_136,
            "configs": four_config(support),
        },
        "known": {"total": len(known), "configs": four_config(known)},
        "fresh_orbit_mapping": fresh_orbit_coverage(),
    }
    result["finite_support"]["all_four_sign_perfect"] = all(
        row["perfect_sign"]
        for row in result["finite_support"]["configs"].values()
    )
    result["finite_support"]["all_four_strict_margin"] = all(
        row["strict_margin"]
        for row in result["finite_support"]["configs"].values()
    )
    result["known"]["all_four_sign_perfect"] = all(
        row["perfect_sign"] for row in result["known"]["configs"].values()
    )
    (HERE / "final_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "profile": result["structural"]["profile"],
        "finite_support": result["finite_support"],
        "known_all_four": result["known"]["all_four_sign_perfect"],
        "fresh_orbit_mapping": result["fresh_orbit_mapping"],
    }, indent=2))


if __name__ == "__main__":
    main()
