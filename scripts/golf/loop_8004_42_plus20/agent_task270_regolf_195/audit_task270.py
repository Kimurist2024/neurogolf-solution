#!/usr/bin/env python3
"""Audit task270 current-derived regolf probes without promoting anything."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "inputs/arc-gen-repo/tasks"),
    str(ROOT / "scripts/golf/loop_7999_13/lane_c11"),
]
from lib import scoring  # noqa: E402
from audit_candidates import runtime_shape_trace  # noqa: E402


ROOT_GUARDS = {
    ROOT / "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    ROOT / "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    ROOT / "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}
PATHS = {
    "authority587": HERE / "baseline/task270_authority.onnx",
    "truthful_direct608": HERE / "candidates/task270_truthful_direct_cost608.onnx",
    "truthful_packed595": HERE / "candidates/task270_truthful_packed_cost595.onnx",
    "truthful_shared_scale592": HERE / "candidates/task270_truthful_shared_scale_cost592.onnx",
    "unsafe_pr2_quantize588": HERE / "candidates/task270_unsafe_pr2_saturating_cost588.onnx",
}
EXPECTED_SHA = {
    "authority587": "0d848124abafda1daf24fe5f779ed5249c9b8b2054854264dde838b05e27a443",
    "truthful_direct608": "77ecc3c5be720d304482c4c49380c29dc60235b94284d8c5b9c2e0031fbe5cba",
    "truthful_packed595": "046e15662b85364584c598cb5b00f21ef9bfadbc48a78b72240259962bf1caac",
    "truthful_shared_scale592": "3d98850eabbb3383d372f31255bfe2a33967d43d56b275c9767e9a1c0cfce4ec",
    "unsafe_pr2_quantize588": "9154c38aa6bb6c4937956b305df9b1340090d7cd64700c031cd3e852d6ce621b",
}
MODES = {
    "disable_all_threads1": (True, 1),
    "disable_all_threads4": (True, 4),
    "default_threads1": (False, 1),
    "default_threads4": (False, 4),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards() -> None:
    for path, expected in ROOT_GUARDS.items():
        if digest(path) != expected:
            raise RuntimeError(f"root guard failed: {path}")
    for label, path in PATHS.items():
        if digest(path) != EXPECTED_SHA[label]:
            raise RuntimeError(f"model guard failed: {label}")


def session(path: Path, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def profile(label: str, path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    scored = scoring.score_and_verify(
        model, 270, str(HERE / "audit/profile_tmp"), label=label, require_correct=False
    )
    if scored is None:
        raise RuntimeError(f"profile failed: {label}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "official_profile": scored,
        "full_checker": True,
        "strict_data_prop": True,
        "runtime_shape": runtime_shape_trace(270, model),
        "op_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "standard_domains": all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node for attr in node.attribute
        ),
        "nonfinite_initializers": int(
            sum(np.count_nonzero(~np.isfinite(array)) for array in arrays if array.dtype.kind in "fc")
        ),
        "conv_bias_ub0": not any(
            node.op_type in {"Conv", "ConvTranspose", "QLinearConv"}
            for node in model.graph.node
        ),
    }


def known_four() -> dict[str, Any]:
    examples = scoring.load_examples(270)
    rows = examples["train"] + examples["test"] + examples["arc-gen"]
    result: dict[str, Any] = {label: {} for label in PATHS}
    for mode, (disabled, threads) in MODES.items():
        sessions = {label: session(path, disabled, threads) for label, path in PATHS.items()}
        for label, runner in sessions.items():
            stats = {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0, "first_wrong": None}
            for index, example in enumerate(rows):
                benchmark = scoring.convert_to_numpy(example)
                assert benchmark is not None
                try:
                    raw = runner.run(None, {"input": benchmark["input"]})[0]
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    stats.setdefault("first_error", f"{type(exc).__name__}: {exc}")
                    continue
                stats["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
                    if stats["first_wrong"] is None:
                        stats["first_wrong"] = index
            result[label][mode] = stats
    return result


def renderer_domain(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    final = next(node for node in model.graph.node if list(node.output) == ["output"])
    inits = {item.name: item for item in model.graph.initializer}
    graph = helper.make_graph(
        [copy.deepcopy(final)],
        "task270_renderer_complete_domain",
        [
            helper.make_tensor_value_info("R", TensorProto.FLOAT16, [2, 30]),
            helper.make_tensor_value_info("C", TensorProto.FLOAT16, [2, 30]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        initializer=[copy.deepcopy(inits["A"]), copy.deepcopy(inits["K"])],
    )
    isolated = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runner = ort.InferenceSession(isolated.SerializeToString(), options, providers=["CPUExecutionProvider"])

    names = ("O", "B", "C", "P")
    codes = {"O": (0.0, 0.0), "B": (1.0, 1.0), "C": (-0.25, 0.5), "P": (-1.0, 0.75)}
    states = [("O/O", codes["O"])]
    states.extend(
        (f"{left}/{right}", (codes[left][0], codes[right][1]))
        for left in names[1:] for right in names[1:] if (left, right) != ("C", "C")
    )
    profiles = np.zeros((2, 30), dtype=np.float16)
    for index, (_, values) in enumerate(states):
        profiles[:, index] = values
    raw = runner.run(None, {"R": profiles, "C": profiles})[0]
    checked = collisions = wrong = nonfinite = 0
    positives: list[float] = []
    for ri, (rn, _) in enumerate(states):
        rs = rn.split("/")
        for ci, (cn, _) in enumerate(states):
            cs = cn.split("/")
            objects = []
            if "O" not in rs + cs:
                for flower, (center, petal) in enumerate(((2, 3), (1, 7))):
                    pair = (rs[flower], cs[flower])
                    if pair == ("C", "C"):
                        objects.append(center)
                    elif pair in {("C", "P"), ("P", "C")}:
                        objects.append(petal)
            if len(objects) > 1:
                collisions += 1
                continue
            want = np.zeros(10, dtype=bool)
            if objects:
                want[objects[0]] = True
            elif "O" not in rs + cs:
                want[0] = True
            cell = raw[0, :, ri, ci]
            wrong += int(not np.array_equal(cell > 0, want))
            nonfinite += int(np.count_nonzero(~np.isfinite(cell)))
            if want.any():
                positives.append(float(cell[np.flatnonzero(want)[0]]))
            checked += 1
    return {
        "axis_states": 9,
        "checked_states_including_padding": checked,
        "reachable_in_grid_states": 62,
        "generator_rejected_collision_states": collisions,
        "wrong": wrong,
        "nonfinite": nonfinite,
        "minimum_intended_positive": min(positives),
    }


def fresh_shared_scale(count: int = 1000) -> dict[str, Any]:
    generator = importlib.import_module("task_ae3edfdc")
    labels = ("authority587", "truthful_shared_scale592")
    result: dict[str, Any] = {}
    for seed in (27019501, 27019502):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        examples = [generator.generate() for _ in range(count)]
        seed_result: dict[str, Any] = {}
        for mode, (disabled, threads) in MODES.items():
            runners = {label: session(PATHS[label], disabled, threads) for label in labels}
            stats = {
                label: {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0, "raw_different_from_authority": 0}
                for label in labels
            }
            for example in examples:
                benchmark = scoring.convert_to_numpy(example)
                assert benchmark is not None
                raws: dict[str, np.ndarray] = {}
                for label, runner in runners.items():
                    try:
                        raw = runner.run(None, {"input": benchmark["input"]})[0]
                    except Exception:  # noqa: BLE001
                        stats[label]["errors"] += 1
                        continue
                    raws[label] = raw
                    stats[label]["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
                    if np.array_equal(raw > 0, benchmark["output"] > 0):
                        stats[label]["right"] += 1
                    else:
                        stats[label]["wrong"] += 1
                if len(raws) == 2:
                    stats["truthful_shared_scale592"]["raw_different_from_authority"] += int(
                        not np.array_equal(raws["authority587"], raws["truthful_shared_scale592"])
                    )
            seed_result[mode] = stats
        result[str(seed)] = seed_result
    return {"count_per_seed": count, "seeds": result}


def presence_masks() -> dict[str, Any]:
    generator = importlib.import_module("task_ae3edfdc")
    rows, cols = [4, 10], [4, 10]
    maximum = [3, 9, 9, 3, 9, 3, 3, 9]
    result: dict[str, Any] = {}
    for mode, (disabled, threads) in MODES.items():
        runner = session(PATHS["truthful_shared_scale592"], disabled, threads)
        stats = {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0}
        for mask in range(256):
            deltas = [value if mask & (1 << index) else -1 for index, value in enumerate(maximum)]
            benchmark = scoring.convert_to_numpy(generator.generate(rows=rows, cols=cols, deltas=deltas))
            assert benchmark is not None
            try:
                raw = runner.run(None, {"input": benchmark["input"]})[0]
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
                continue
            stats["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
            if np.array_equal(raw > 0, benchmark["output"] > 0):
                stats["right"] += 1
            else:
                stats["wrong"] += 1
        result[mode] = stats
    return result


def main() -> int:
    guards()
    profiles = {label: profile(label, path) for label, path in PATHS.items()}
    result = {
        "task": 270,
        "task_hash": "ae3edfdc",
        "decision": "NO_ADMISSIBLE_STRICT_LOWER_CANDIDATE",
        "profiles": profiles,
        "known_four_mode": known_four(),
        "truthful_shared_scale_complete_renderer_domain": renderer_domain(
            PATHS["truthful_shared_scale592"]
        ),
        "truthful_shared_scale_presence_masks_four_mode": presence_masks(),
        "truthful_shared_scale_fresh": fresh_shared_scale(),
        "finite_detector_proof": {
            "size": 15,
            "center_coordinate_domain": [2, 12],
            "center_rows_and_columns_distinct": True,
            "presence_masks": 256,
            "identities": [
                "sr=sum(petal_row-center_row)",
                "qr=sum((petal_row-center_row)^2)",
                "qr>sr^2 iff both vertical directions are present",
                "sign(sr) selects the sole vertical direction",
                "petal_count-up-down gives horizontal count",
                "sign(sum(petal_col)-count*center_col) selects the sole horizontal direction",
            ],
            "packed_scale": 2048,
            "float32_integer_exact": True,
            "uint8_lane_arithmetic": "exhausted by all 256 presence masks plus fresh coordinate/delta coverage",
        },
        "renderer_rank_search": {
            "rank6_params": {"A": 12, "K": 60},
            "rank5_projected_param_saving": 12,
            "runs": [
                {"seed": 270195, "steps": 12000, "best_fp16_wrong": 184, "note": "pre-loss correction"},
                {"seed": 270196, "steps": 12000, "best_fp16_wrong": 15},
                {"seed": 270197, "steps": 16000, "best_fp16_wrong": 25},
                {"seed": 270198, "steps": 10000, "best_fp16_wrong": 15},
            ],
            "accepted": False,
        },
        "mechanical_disposition": {
            "unused_initializers": 0,
            "byte_identical_initializer_aliases": 0,
            "dead_nodes": 0,
            "removable_optional_outputs": 0,
            "castlike_to_cast": "cost626 in prior/current diagnostic because truthful [2,3] propagation becomes charged",
            "concat_scatter": "two [2,3] int32 index tensors and two [2,30] fp16 profiles are already element-minimal for this formulation",
            "shared_center_quantize_scale": "valid, saves 3 cost versus truthful packed595, terminal cost592",
            "pr2_quantize": "cost588 but incorrect because uint8 QuantizeLinear saturates rather than modulo-wraps",
        },
        "root_modified": False,
        "others_71407_observe_only_sha256": digest(ROOT / "others/71407/MANIFEST.json"),
    }
    out = HERE / "audit/result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    guards()
    print(json.dumps({
        "decision": result["decision"],
        "costs": {label: row["official_profile"]["cost"] for label, row in profiles.items()},
        "shape_mismatches": {
            label: len(row["runtime_shape"]["declared_actual_mismatches"])
            for label, row in profiles.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
