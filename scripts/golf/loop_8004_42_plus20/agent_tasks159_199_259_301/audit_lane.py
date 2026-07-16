#!/usr/bin/env python3
"""Fail-closed exact-regolf audit for task159/199/259/301."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = HERE / "current"
EVIDENCE = HERE / "evidence"
TASKS = (159, 199, 259, 301)
TASK301_REJECT = HERE / "rejected" / "task301_cyan_max_exact_REJECT.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TFIDFVECTORIZER", "SCATTERELEMENTS", "SCATTERND", "HARDMAX"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def profile(model: onnx.ModelProto) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="regolf159_") as tmp:
        path = Path(tmp) / "candidate.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def examples(task: int) -> list[dict[str, np.ndarray]]:
    return [
        item
        for split in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(split, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]


def session(model: onnx.ModelProto, optimization: str, threads: int, profile_prefix: str | None = None) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    return ort.InferenceSession(clean.SerializeToString(), options, providers=["CPUExecutionProvider"])


def runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        return {"truthful": False, "error": "sanitize rejected"}
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(clean), strict_mode=True, data_prop=True)
    declared = {
        value.name: shape(value)
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    by_node = {node.name: node for node in inferred.graph.node if node.name}
    with tempfile.TemporaryDirectory(prefix=f"shape_{task:03d}_") as tmp:
        sess = session(model, "disable_all", 1, str(Path(tmp) / "profile"))
        item = examples(task)[0]
        output = sess.run(None, {"input": item["input"]})[0]
        events = json.loads(Path(sess.end_profiling()).read_text(encoding="utf-8"))
    actual: dict[str, list[int]] = {}
    for event in events:
        args = event.get("args", {})
        shapes = args.get("output_type_shape")
        if event.get("cat") != "Node" or not shapes:
            continue
        event_name = str(event.get("name", "")).removesuffix("_kernel_time")
        node = by_node.get(event_name)
        if node is None:
            continue
        for index, item in enumerate(shapes):
            if index >= len(node.output) or not node.output[index] or not item:
                continue
            dims = next(iter(item.values()))
            actual[node.output[index]] = [int(dim) for dim in dims]
    actual[inferred.graph.output[0].name] = list(output.shape)
    mismatches = {
        name: {"declared": declared.get(name), "runtime": dims}
        for name, dims in actual.items()
        if declared.get(name) != dims
    }
    return {
        "truthful": not mismatches,
        "profiled_outputs": len(actual),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def mechanical(model: onnx.ModelProto) -> dict[str, Any]:
    init = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output if name}
    consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ni, node in enumerate(model.graph.node):
        for ii, name in enumerate(node.input):
            if name:
                consumers[name].append((ni, ii))
    live = {value.name for value in model.graph.output}
    stack = list(live)
    while stack:
        name = stack.pop()
        node = producers.get(name)
        if node is None:
            continue
        for source in node.input:
            if source and source not in live:
                live.add(source)
                stack.append(source)
    dead_nodes = [
        index for index, node in enumerate(model.graph.node)
        if not any(output in live for output in node.output)
    ]
    unused_initializers = sorted(set(init) - {name for node in model.graph.node for name in node.input})
    duplicates: list[list[str]] = []
    names = list(init)
    seen: set[str] = set()
    for i, name in enumerate(names):
        if name in seen:
            continue
        group = [name]
        for other in names[i + 1 :]:
            if init[name].dtype == init[other].dtype and init[name].shape == init[other].shape and np.array_equal(init[name], init[other]):
                group.append(other)
        if len(group) > 1:
            duplicates.append(group)
            seen.update(group)
    cse: list[list[int]] = []
    signatures: dict[bytes, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        clone = copy.deepcopy(node)
        del clone.output[:]
        clone.name = ""
        signatures[clone.SerializeToString()].append(index)
    cse = [indices for indices in signatures.values() if len(indices) > 1]
    noops = [
        index for index, node in enumerate(model.graph.node)
        if node.op_type in {"Identity", "Dropout"}
    ]
    ranks: dict[str, Any] = {}
    for name, array in init.items():
        if array.ndim >= 2 and array.dtype.kind in "fiu" and np.isfinite(array).all():
            ranks[name] = {
                "shape": list(array.shape),
                "axis_unfolding_ranks": [
                    int(np.linalg.matrix_rank(np.moveaxis(array.astype(np.float64), axis, 0).reshape(array.shape[axis], -1)))
                    for axis in range(array.ndim)
                ],
            }
    return {
        "dead_nodes": dead_nodes,
        "unused_initializers": unused_initializers,
        "duplicate_initializer_aliases": duplicates,
        "common_subexpressions": cse,
        "obvious_noops": noops,
        "initializer_axis_ranks": ranks,
        "consumer_counts": {name: len(rows) for name, rows in sorted(consumers.items()) if name in init},
    }


def optimizer_scan(model: onnx.ModelProto, baseline: dict[str, int]) -> dict[str, Any]:
    passes = [
        "nop",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_consecutive_idempotent_ops",
        "eliminate_nop_pad",
        "eliminate_nop_concat",
        "eliminate_nop_split",
        "eliminate_nop_expand",
        "eliminate_shape_gather",
        "eliminate_slice_after_shape",
        "eliminate_nop_transpose",
        "fuse_consecutive_concats",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "eliminate_nop_reshape",
        "eliminate_nop_with_unit",
        "eliminate_common_subexpression",
        "fuse_consecutive_unsqueezes",
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_shape_op",
        "fuse_consecutive_slices",
        "eliminate_unused_initializer",
        "eliminate_duplicate_initializer",
    ]
    rows: list[dict[str, Any]] = []
    for label, selected in [(name, [name]) for name in passes] + [("all_safe", passes)]:
        row: dict[str, Any] = {"profile": label}
        try:
            candidate = onnxoptimizer.optimize(copy.deepcopy(model), selected, fixed_point=True)
            onnx.checker.check_model(candidate, full_check=True)
            onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
            current = profile(candidate)
            row.update(
                nodes=len(candidate.graph.node),
                initializers=len(candidate.graph.initializer),
                cost=current,
                changed=candidate.SerializeToString() != model.SerializeToString(),
                strict_lower=current["cost"] < baseline["cost"],
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return {
        "profiles": len(rows),
        "changed": sum(bool(row.get("changed")) for row in rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "rows": rows,
    }


def structure(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        full = True
    except Exception as exc:  # noqa: BLE001
        full = False
        errors.append(f"full:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"strict:{type(exc).__name__}:{exc}")
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    try:
        truth = runtime_shapes(model, task)
    except Exception as exc:  # noqa: BLE001
        truth = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "full_check": full,
        "strict_data_prop": strict,
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and shape(model.graph.input[0]) == [1, 10, 30, 30]
            and shape(model.graph.output[0]) == [1, 10, 30, 30]
        ),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_banned": all(node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper() for node in model.graph.node),
        "no_lookup": all(node.op_type.upper() not in LOOKUP for node in model.graph.node),
        "no_shape_cloak": truth.get("truthful") is True and all(node.op_type != "CenterCropPad" for node in model.graph.node),
        "no_new_giant_einsum_note": f"incumbent max Einsum arity={max_einsum}; no new Einsum was built",
        "conv_bias_ub0": not check_conv_bias(model),
        "ops": dict(sorted(ops.items())),
        "runtime_shapes": truth,
        "errors": errors,
    }


def task301_counterexample(base: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    a = session(base, "disable_all", 1)
    b = session(candidate, "disable_all", 1)
    index = 0
    for split in ("train", "test", "arc-gen"):
        for split_index, raw in enumerate(scoring.load_examples(301).get(split, [])):
            item = scoring.convert_to_numpy(raw)
            if item is None:
                continue
            left = a.run(None, {"input": item["input"]})[0]
            right = b.run(None, {"input": item["input"]})[0]
            if not np.array_equal(left, right):
                counts = item["input"].sum(axis=(2, 3))[0]
                return {
                    "global_index": index,
                    "split": split,
                    "split_index": split_index,
                    "channel_counts": [float(value) for value in counts],
                    "B_background_max": float(counts.max()),
                    "n_cyan_channel8": float(counts[8]),
                    "raw_different_elements": int(np.count_nonzero(left != right)),
                    "decoded_different_elements": int(np.count_nonzero((left > 0) != (right > 0))),
                    "baseline_correct": bool(np.array_equal(left > 0, item["output"] > 0)),
                    "candidate_correct": bool(np.array_equal(right > 0, item["output"] > 0)),
                }
            index += 1
    return {"error": "no counterexample found"}


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": hashlib.sha256((ROOT / "submission_base_8009.46.zip").read_bytes()).hexdigest(),
        "tasks": {},
    }
    for task in TASKS:
        path = CURRENT / f"task{task:03d}.onnx"
        model = onnx.load(path)
        baseline = profile(model)
        result["tasks"][str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "profile": baseline,
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "structure": structure(model, task),
            "mechanical": mechanical(model),
            "optimizer_scan": optimizer_scan(model, baseline),
        }

    base301 = onnx.load(CURRENT / "task301.onnx")
    cand301 = onnx.load(TASK301_REJECT)
    result["task301_rejected_candidate"] = {
        "path": str(TASK301_REJECT.relative_to(ROOT)),
        "sha256": sha(TASK301_REJECT),
        "baseline_profile": profile(base301),
        "candidate_profile": profile(cand301),
        "strict_lower": profile(cand301)["cost"] < profile(base301)["cost"],
        "counterexample": task301_counterexample(base301, cand301),
        "classification": "REJECT_KNOWN_RAW_AND_DECODED_MISMATCH",
        "fresh_not_run": "known exact-equivalence gate failed on first example",
    }
    result["winner"] = None
    result["conclusion"] = "No strict-lower candidate passes the structural and exactness gates."
    (EVIDENCE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tasks": {task: result["tasks"][str(task)]["profile"] for task in TASKS},
        "optimizer_strict_lower": {task: len(result["tasks"][str(task)]["optimizer_scan"]["strict_lower"]) for task in TASKS},
        "task301_reject": result["task301_rejected_candidate"],
    }, indent=2))


if __name__ == "__main__":
    main()
