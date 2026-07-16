#!/usr/bin/env python3
"""Strict, non-promoting regolf audit for tasks 189/263/304/383.

All generated files stay beside this script.  The root submission archive and
score ledgers are read-only inputs.  No call is made to try_candidate.py.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = HERE / "authority"
CANDIDATES = HERE / "candidates"
EVIDENCE = HERE / "evidence"

TASKS = (189, 263, 304, 383)
HASHES = {
    189: "7c008303",
    263: "a87f7484",
    304: "c3e719e8",
    383: "f1cefba8",
}
AUTHORITY_COSTS = {189: 183, 263: 181, 304: 180, 383: 172}
FRESH_SEEDS = (189_263_304, 383_304_263)
FRESH_PER_SEED = 1000
CONFIGS = (
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1, "disable_all_threads1"),
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4, "disable_all_threads4"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1, "default_threads1"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4, "default_threads4"),
)

PASS_SETS: dict[str, list[str]] = {
    "dead": ["eliminate_deadend"],
    "cse": ["eliminate_common_subexpression"],
    "initializer_aliases": [
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
    ],
    "idempotent": ["eliminate_consecutive_idempotent_ops"],
    "noops": [
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_concat",
        "eliminate_nop_dropout",
        "eliminate_nop_expand",
        "eliminate_nop_flatten",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_split",
        "eliminate_nop_transpose",
        "eliminate_nop_with_unit",
    ],
    "safe_cleanup": [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
        "eliminate_common_subexpression",
        "eliminate_consecutive_idempotent_ops",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_concat",
        "eliminate_nop_dropout",
        "eliminate_nop_expand",
        "eliminate_nop_flatten",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_split",
        "eliminate_nop_transpose",
        "eliminate_nop_with_unit",
    ],
    "conv_fusions": [
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_pad_into_conv",
        "fuse_pad_into_pool",
    ],
    "shape_folds": [
        "eliminate_shape_gather",
        "eliminate_slice_after_shape",
        "eliminate_shape_op",
        "fuse_consecutive_slices",
    ],
    "einsum_matmul": ["replace_einsum_with_matmul"],
    "rewrite_where": ["rewrite_where"],
    "adjust_add": ["adjust_add"],
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf import check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tensor_shape(item: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in item.type.tensor_type.shape.dim
    ]


def load_known(task: int) -> list[dict[str, Any]]:
    examples = scoring.load_examples(task)
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in examples.get(split, [])
    ]


def load_rule(task: int):
    path = ROOT / "inputs/sakana-gcg-2025/raw" / f"task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"raw_rule_{task}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def make_fresh(task: int, seed: int) -> list[dict[str, Any]]:
    generator = importlib.import_module(f"task_{HASHES[task]}")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [generator.generate() for _ in range(FRESH_PER_SEED)]


def independent_truth(task: int, examples: list[dict[str, Any]]) -> dict[str, Any]:
    rule = load_rule(task)
    mismatch_count = 0
    error_count = 0
    first: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        try:
            observed = rule(copy.deepcopy(example["input"]))
            if [list(row) for row in observed] != example["output"]:
                mismatch_count += 1
                if len(first) < 5:
                    first.append({"index": index, "kind": "mismatch"})
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            if len(first) < 5:
                first.append(
                    {"index": index, "kind": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
    return {
        "attempts": len(examples),
        "right": len(examples) - mismatch_count - error_count,
        "mismatch_count": mismatch_count,
        "error_count": error_count,
        "first_failures": first,
    }


def make_session(data: bytes, level: ort.GraphOptimizationLevel, threads: int):
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected payload")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def runtime_group(
    data: bytes,
    examples: list[dict[str, Any]],
    authority_data: bytes | None = None,
) -> dict[str, Any]:
    sessions: dict[str, ort.InferenceSession] = {}
    authorities: dict[str, ort.InferenceSession] = {}
    session_errors: dict[str, str] = {}
    for level, threads, label in CONFIGS:
        try:
            sessions[label] = make_session(data, level, threads)
            if authority_data is not None:
                authorities[label] = make_session(authority_data, level, threads)
        except Exception as exc:  # noqa: BLE001
            session_errors[label] = f"{type(exc).__name__}: {exc}"

    metrics: dict[str, dict[str, Any]] = {
        label: {
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "shape_mismatches": 0,
            "nonfinite_values": 0,
            "bad_margin_values": 0,
            "raw_equal_to_authority": 0,
            "raw_equal_to_disable_all_threads1": 0,
            "output_shapes": set(),
            "min_positive": None,
            "max_nonpositive": None,
        }
        for _, _, label in CONFIGS
    }
    valid = 0
    first_failure: dict[str, Any] | None = None
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        valid += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            row = metrics[label]
            try:
                value = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                first_failure = first_failure or {
                    "index": index,
                    "config": label,
                    "kind": "runtime_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            outputs[label] = value
            row["output_shapes"].add(tuple(int(x) for x in value.shape))
            row["nonfinite_values"] += int(value.size - np.isfinite(value).sum())
            row["bad_margin_values"] += int(np.count_nonzero((value > 0) & (value < 0.25)))
            positive = value[value > 0]
            nonpositive = value[value <= 0]
            if positive.size:
                current = float(np.min(positive))
                row["min_positive"] = (
                    current if row["min_positive"] is None else min(row["min_positive"], current)
                )
            if nonpositive.size:
                current = float(np.max(nonpositive))
                row["max_nonpositive"] = (
                    current
                    if row["max_nonpositive"] is None
                    else max(row["max_nonpositive"], current)
                )
            shape_ok = value.shape == expected.shape
            if not shape_ok:
                row["shape_mismatches"] += 1
            correct = shape_ok and np.array_equal(value > 0, expected)
            row["right" if correct else "wrong"] += 1
            if not correct and first_failure is None:
                first_failure = {
                    "index": index,
                    "config": label,
                    "kind": "wrong",
                    "shape": list(value.shape),
                    "expected_shape": list(expected.shape),
                    "threshold_differences": (
                        int(np.count_nonzero((value > 0) != expected)) if shape_ok else None
                    ),
                }
            authority_session = authorities.get(label)
            if authority_session is not None:
                try:
                    authority_value = np.asarray(
                        authority_session.run(
                            [authority_session.get_outputs()[0].name],
                            {authority_session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    if (
                        value.dtype == authority_value.dtype
                        and value.shape == authority_value.shape
                        and np.ascontiguousarray(value).tobytes()
                        == np.ascontiguousarray(authority_value).tobytes()
                    ):
                        row["raw_equal_to_authority"] += 1
                except Exception:  # noqa: BLE001
                    pass
        reference = outputs.get("disable_all_threads1")
        if reference is not None:
            raw = np.ascontiguousarray(reference).tobytes()
            for label, value in outputs.items():
                if (
                    value.dtype == reference.dtype
                    and value.shape == reference.shape
                    and np.ascontiguousarray(value).tobytes() == raw
                ):
                    metrics[label]["raw_equal_to_disable_all_threads1"] += 1

    for row in metrics.values():
        row["output_shapes"] = [list(x) for x in sorted(row["output_shapes"])]
    return {
        "attempts": len(examples),
        "valid": valid,
        "session_errors": session_errors,
        "configs": metrics,
        "first_failure": first_failure,
    }


def runtime_shape_trace(data: bytes, example: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        typed = {
            item.name: item
            for item in list(inferred.graph.value_info) + list(inferred.graph.output)
            if item.type.HasField("tensor_type")
        }
        expected = {name: tuple(tensor_shape(item)) for name, item in typed.items()}
        names: list[str] = []
        seen: set[str] = set()
        for node in inferred.graph.node:
            for name in node.output:
                if name and name in typed and name not in seen:
                    names.append(name)
                    seen.add(name)
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        session = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError("witness is not scorer-convertible")
        values = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
        mismatches = [
            {
                "tensor": name,
                "declared": list(expected[name]),
                "runtime": list(value.shape),
            }
            for name, value in zip(names, values)
            if tuple(value.shape) != expected[name]
        ]
        return {
            "traced": len(names),
            "truthful": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def structural_audit(task: int, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    memory, params, cost = cost_of(str(path))
    produced = {name for node in model.graph.node for name in node.output if name}
    consumed = {name for node in model.graph.node for name in node.input if name}
    outputs = {item.name for item in model.graph.output}
    initializer_names = {item.name for item in model.graph.initializer}
    return {
        "task": task,
        "sha256": sha256_bytes(data),
        "serialized_size": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "profile": {"memory": int(memory), "params": int(params), "cost": int(cost)},
        "authority_cost": AUTHORITY_COSTS[task],
        "profile_matches_authority": int(cost) == AUTHORITY_COSTS[task],
        "checker_full": True,
        "strict_data_prop": True,
        "declared_outputs": [tensor_shape(item) for item in model.graph.output],
        "inferred_outputs": [tensor_shape(item) for item in inferred.graph.output],
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "standard_domains": all(x.domain in {"", "ai.onnx"} for x in model.opset_import)
        and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper()
            in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
            or "Sequence" in node.op_type
        ],
        "lookup_or_policy_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND", "ScatterElements"}
        ],
        "center_crop_pad": sum(node.op_type == "CenterCropPad" for node in model.graph.node),
        "giant_nodes": [
            {"op": node.op_type, "inputs": len(node.input)}
            for node in model.graph.node
            if len(node.input) > 16
        ],
        "conv_bias_ub": [list(item) for item in check_conv_bias.check_model(model)],
        "dead_node_outputs": sorted(produced - consumed - outputs),
        "unused_initializers": sorted(initializer_names - consumed),
    }


def optimizer_scan(task: int, data: bytes) -> list[dict[str, Any]]:
    source = onnx.load_model_from_string(data)
    rows: list[dict[str, Any]] = []
    for label, passes in PASS_SETS.items():
        row: dict[str, Any] = {"label": label, "passes": passes}
        try:
            candidate = onnxoptimizer.optimize(source, passes, fixed_point=True)
            encoded = candidate.SerializeToString()
            row["changed"] = encoded != data
            onnx.checker.check_model(candidate, full_check=True)
            onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
            with tempfile.TemporaryDirectory(prefix=f"task{task}_{label}_") as temp:
                path = Path(temp) / f"task{task:03d}.onnx"
                onnx.save(candidate, path)
                memory, params, cost = cost_of(str(path))
            row["profile"] = {
                "memory": int(memory),
                "params": int(params),
                "cost": int(cost),
            }
            row["strict_lower"] = int(cost) < AUTHORITY_COSTS[task]
            if row["strict_lower"]:
                path = CANDIDATES / f"task{task:03d}_optimizer_{label}.onnx"
                onnx.save(candidate, path)
                row["path"] = str(path.relative_to(ROOT))
                row["sha256"] = sha256_file(path)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    old = next(item for item in model.graph.initializer if item.name == name)
    index = list(model.graph.initializer).index(old)
    model.graph.initializer.remove(old)
    model.graph.initializer.insert(index, numpy_helper.from_array(array, name))


def build_task304_rank_drop(component: int) -> Path:
    model = onnx.load(AUTHORITY / "task304.onnx")
    for name in ("Basis", "Tcol"):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item)
        replace_initializer(model, name, np.delete(array, component, axis=1))
    path = CANDIDATES / f"task304_drop_color_factor_{component}.onnx"
    onnx.save(model, path)
    return path


def build_task304_t_drop(component: int) -> Path:
    model = onnx.load(AUTHORITY / "task304.onnx")
    for name, axis in (("D", 0), ("Tcol", 0)):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item)
        replace_initializer(model, name, np.delete(array, component, axis=axis))
    path = CANDIDATES / f"task304_drop_state_factor_{component}.onnx"
    onnx.save(model, path)
    return path


def build_task304_precontract() -> Path:
    """Symbolically contract every adjacent SF/SG,H pair inside the Einsum."""
    model = onnx.load(AUTHORITY / "task304.onnx")
    node = model.graph.node[0]
    equation = next(
        onnx.helper.get_attribute_value(attr).decode()
        for attr in node.attribute
        if attr.name == "equation"
    )
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    names = list(node.input)
    arrays = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    hf = np.einsum("xd,dA->xA", arrays["H"], arrays["SF"]).astype(np.float32)
    hg = np.einsum("xd,dA->xA", arrays["H"], arrays["SG"]).astype(np.float32)

    new_terms: list[str] = []
    new_names: list[str] = []
    index = 0
    while index < len(names):
        if (
            index + 1 < len(names)
            and names[index] in {"SF", "SG"}
            and names[index + 1] == "H"
        ):
            left, right = terms[index], terms[index + 1]
            shared = [char for char in left if char in right]
            if len(shared) != 1:
                raise RuntimeError(f"unexpected pair labels {left},{right}")
            contracted = shared[0]
            result = "".join(char for char in right if char != contracted)
            result += "".join(char for char in left if char != contracted)
            new_terms.append(result)
            new_names.append("HF" if names[index] == "SF" else "HG")
            index += 2
        else:
            new_terms.append(terms[index])
            new_names.append(names[index])
            index += 1

    del node.input[:]
    node.input.extend(new_names)
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = (",".join(new_terms) + "->" + rhs).encode()
    for name in ("H", "SF", "SG"):
        item = next(x for x in model.graph.initializer if x.name == name)
        model.graph.initializer.remove(item)
    model.graph.initializer.extend(
        [numpy_helper.from_array(hf, "HF"), numpy_helper.from_array(hg, "HG")]
    )
    path = CANDIDATES / "task304_precontract_H_selectors.onnx"
    onnx.save(model, path)
    return path


def build_task263_truthful_bypass() -> Path:
    """Replace the GroupNorm booleanization trick with an honest Cast(input)."""
    model = onnx.load(AUTHORITY / "task263.onnx")
    nodes = list(model.graph.node)
    if nodes[0].op_type != "GroupNormalization" or nodes[1].op_type != "CastLike":
        raise RuntimeError("unexpected task263 prefix")
    direct = onnx.helper.make_node("Cast", ["input"], ["q"], to=onnx.TensorProto.UINT8)
    del model.graph.node[:]
    model.graph.node.extend([direct, *nodes[2:]])
    scale = next(item for item in model.graph.initializer if item.name == "scale")
    model.graph.initializer.remove(scale)
    # Remove the authority's cloaked [1,1,1,1] annotations so strict inference
    # can derive the honest full-grid intermediates from the replacement Cast.
    del model.graph.value_info[:]
    path = CANDIDATES / "task263_truthful_cast_bypass.onnx"
    onnx.save(model, path)
    return path


def audit_probe(task: int, path: Path, known: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
    }
    try:
        row["structural"] = structural_audit(task, path)
        data = path.read_bytes()
        authority_data = (AUTHORITY / f"task{task}.onnx").read_bytes()
        row["runtime_shape"] = runtime_shape_trace(data, known[0])
        row["known_four_configs"] = runtime_group(data, known, authority_data)
        cost = row["structural"]["profile"]["cost"]
        row["strict_lower"] = cost < AUTHORITY_COSTS[task]
        row["score_gain_if_correct"] = math.log(AUTHORITY_COSTS[task] / cost)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def guard_snapshot() -> dict[str, Any]:
    paths = [ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "others/71407"]
    result: dict[str, Any] = {}
    for path in paths:
        if path.is_file():
            result[str(path.relative_to(ROOT))] = {
                "kind": "file",
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        elif path.is_dir():
            members = []
            for member in sorted(x for x in path.rglob("*") if x.is_file()):
                members.append(
                    {
                        "path": str(member.relative_to(path)),
                        "sha256": sha256_file(member),
                        "size": member.stat().st_size,
                    }
                )
            digest = hashlib.sha256(
                json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            result[str(path.relative_to(ROOT))] = {
                "kind": "directory_manifest",
                "sha256": digest,
                "members": len(members),
            }
        else:
            result[str(path.relative_to(ROOT))] = {"kind": "absent"}
    return result


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(4)
    output = EVIDENCE / "audit.json"
    report: dict[str, Any] = {
        "authority_source": "root submission.zip extracted members",
        "authority_archive_sha256": sha256_file(ROOT / "submission.zip"),
        "authority_costs": AUTHORITY_COSTS,
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_per_seed": FRESH_PER_SEED,
        "configs": [label for _, _, label in CONFIGS],
        "guard_before": guard_snapshot(),
        "tasks": {},
        "manual_probes": [],
    }

    known_by_task: dict[int, list[dict[str, Any]]] = {}
    for task in TASKS:
        path = AUTHORITY / f"task{task}.onnx"
        data = path.read_bytes()
        known = load_known(task)
        known_by_task[task] = known
        task_row: dict[str, Any] = {
            "generator": f"inputs/arc-gen-repo/tasks/task_{HASHES[task]}.py",
            "generator_sha256": sha256_file(
                ROOT / "inputs/arc-gen-repo/tasks" / f"task_{HASHES[task]}.py"
            ),
            "raw_rule": f"inputs/sakana-gcg-2025/raw/task{task:03d}.py",
            "structural": structural_audit(task, path),
            "runtime_shape": runtime_shape_trace(data, known[0]),
            "known_independent_truth": independent_truth(task, known),
            "known_runtime": runtime_group(data, known),
            "optimizer_profiles": optimizer_scan(task, data),
            "fresh": {},
        }
        report["tasks"][str(task)] = task_row
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"task{task:03d} known complete", flush=True)
        for seed in FRESH_SEEDS:
            fresh = make_fresh(task, seed)
            task_row["fresh"][str(seed)] = {
                "independent_truth": independent_truth(task, fresh),
                "runtime": runtime_group(data, fresh),
            }
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"task{task:03d} fresh seed={seed} complete", flush=True)

    probe_paths = [
        (263, build_task263_truthful_bypass()),
        (304, build_task304_precontract()),
        *[(304, build_task304_rank_drop(index)) for index in range(4)],
        *[(304, build_task304_t_drop(index)) for index in range(2)],
    ]
    for task, path in probe_paths:
        probe = audit_probe(task, path, known_by_task[task])
        report["manual_probes"].append(probe)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"probe {path.name} complete", flush=True)

    optimizer_lower = [
        {"task": int(task), **profile}
        for task, task_row in report["tasks"].items()
        for profile in task_row["optimizer_profiles"]
        if profile.get("strict_lower")
    ]
    manual_lower = [
        probe
        for probe in report["manual_probes"]
        if probe.get("strict_lower")
    ]
    report["summary"] = {
        "optimizer_profiles": len(TASKS) * len(PASS_SETS),
        "optimizer_strict_lower": optimizer_lower,
        "manual_strict_lower_count": len(manual_lower),
        "manual_strict_lower": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["sha256"],
                "cost": row.get("structural", {}).get("profile", {}).get("cost"),
                "known_first_failure": row.get("known_four_configs", {}).get("first_failure"),
            }
            for row in manual_lower
        ],
        "accepted_winners": [],
        "exact_cost_gain": 0,
        "exact_score_gain": 0.0,
        "decision": "NO_STRICT_LOWER_SUPPORT_SAFE_WINNER",
    }
    report["guard_after"] = guard_snapshot()
    report["guards_unchanged"] = report["guard_before"] == report["guard_after"]
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
