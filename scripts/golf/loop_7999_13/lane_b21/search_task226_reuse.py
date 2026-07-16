#!/usr/bin/env python3
"""Search exact same-shape initializer reuse for the Wave15 task226 net.

The generator domain is finite.  For every proposed equality between two
same-shaped code initializers, this script refits the existing two-dimensional
QLinearConv decoder and exhaustively checks every generator-reachable
``wides``/``talls`` pair under both ORT modes.  It never promotes a model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE.parent / "lane_a5" / "baseline" / "task226.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))

from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


TASK_HASH = "941d9a10"
SCALAR_CODES = ("q_xz", "R_B", "x_zp", "C_F", "C_M", "C_L")
VECTOR_CODES = ("R_F", "R_M", "R_L")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compositions(length: int, total: int, low: int, high: int):
    for values in itertools.product(range(low, high + 1), repeat=length):
        if sum(values) == total:
            yield values


def domain_cases() -> list[tuple[dict[str, object], np.ndarray, np.ndarray]]:
    generator = importlib.import_module(f"task_{TASK_HASH}")
    wides = list(compositions(3, 8, 1, 4)) + list(compositions(5, 6, 1, 4))
    talls = list(compositions(3, 8, 1, 3)) + list(compositions(5, 6, 1, 3))
    cases = []
    for wide in wides:
        for tall in talls:
            example = generator.generate(wides=list(wide), talls=list(tall))
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError("finite domain conversion failed")
            cases.append((example, benchmark["input"], benchmark["output"] > 0.0))
    return cases


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {item.name: item for item in model.graph.initializer}


def tie_initializer(
    original: onnx.ModelProto, target: str, source: str
) -> onnx.ModelProto:
    model = copy.deepcopy(original)
    initializers = initializer_map(model)
    if target not in initializers or source not in initializers:
        raise KeyError((target, source))
    left = numpy_helper.to_array(initializers[target])
    right = numpy_helper.to_array(initializers[source])
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"incompatible tie {target}->{source}")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == target:
                node.input[index] = source
    kept = [item for item in model.graph.initializer if item.name != target]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def expose_feat(model: onnx.ModelProto) -> onnx.ModelProto:
    probe = copy.deepcopy(model)
    probe.graph.output.append(
        helper.make_tensor_value_info("feat", TensorProto.UINT8, [1, 2, 10, 10])
    )
    onnx.checker.check_model(probe, full_check=True)
    return probe


def feature_dataset(
    model: onnx.ModelProto,
    cases: list[tuple[dict[str, object], np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    session = make_session(expose_feat(model), True)
    observations: set[tuple[int, int, int]] = set()
    for example, input_array, _ in cases:
        feat = session.run(["feat"], {"input": input_array})[0]
        labels = np.asarray(example["output"], dtype=np.int64)
        if feat.shape != (1, 2, 10, 10) or labels.shape != (10, 10):
            raise RuntimeError((feat.shape, labels.shape))
        for row in range(10):
            for col in range(10):
                observations.add(
                    (int(feat[0, 0, row, col]), int(feat[0, 1, row, col]), int(labels[row, col]))
                )
    feature_to_labels: dict[tuple[int, int], set[int]] = {}
    for first, second, label in observations:
        feature_to_labels.setdefault((first, second), set()).add(label)
    conflicts = {
        str(feature): sorted(labels)
        for feature, labels in feature_to_labels.items()
        if len(labels) != 1
    }
    if conflicts:
        raise RuntimeError(f"feature conflicts: {conflicts}")
    ordered = sorted((first, second, label) for first, second, label in observations)
    points = np.asarray([[first, second] for first, second, _ in ordered], dtype=np.int32)
    labels = np.asarray([label for _, _, label in ordered], dtype=np.int64)
    return points, labels


def qlinear_inputs(model: onnx.ModelProto) -> tuple[str, str, str, str]:
    nodes = [node for node in model.graph.node if node.op_type == "QLinearConv"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one QLinearConv, got {len(nodes)}")
    node = nodes[0]
    return node.input[2], node.input[3], node.input[5], node.input[7]


def scalar_value(model: onnx.ModelProto, name: str) -> int | float:
    value = numpy_helper.to_array(initializer_map(model)[name])
    return value.reshape(-1)[0].item()


def fit_decoder(
    model: onnx.ModelProto, points: np.ndarray, labels: np.ndarray
) -> tuple[onnx.ModelProto, dict[str, object]]:
    x_zp_name, weight_name, w_zp_name, y_zp_name = qlinear_inputs(model)
    x_zp = int(scalar_value(model, x_zp_name))
    w_zp = int(scalar_value(model, w_zp_name))
    y_zp = int(scalar_value(model, y_zp_name))
    if y_zp != 0:
        raise RuntimeError(f"positive output zero point is inadmissible: {y_zp}")
    centered = points.astype(np.int32) - x_zp
    values = np.arange(256, dtype=np.int32)
    weights = np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1).reshape(-1, 2)
    deltas = weights - w_zp
    scores = centered @ deltas.T
    scale_name = next(node.input[1] for node in model.graph.node if node.op_type == "QLinearConv")
    scale = float(scalar_value(model, scale_name))
    # QLinearConv rounds the scaled integer accumulator to uint8.  With the
    # incumbent scale, integer scores through 34 map exactly to zero while 35
    # is the first positive output.  This is still a clean post-quantization
    # margin: the graph output itself is uint8 {0, >=1}.
    max_off = math.ceil(0.5 / scale) - 1
    min_on = max_off + 1
    fitted = np.full((10, 2, 1, 1), w_zp, dtype=np.uint8)
    details: dict[str, object] = {
        "x_zero_point": x_zp,
        "weight_zero_point": w_zp,
        "output_zero_point": y_zp,
        "quant_scale": scale,
        "minimum_integer_on_score": min_on,
        "maximum_integer_off_score": max_off,
        "unique_feature_label_pairs": int(len(labels)),
        "classes": {},
    }
    for label in (0, 1, 2, 3, 5):
        on = labels == label
        off = ~on
        feasible = np.all(scores[on] >= min_on, axis=0) & np.all(scores[off] <= max_off, axis=0)
        indices = np.flatnonzero(feasible)
        if not len(indices):
            raise RuntimeError(f"class {label} is not robustly homogeneous-linearly separable")
        on_min = scores[on][:, indices].min(axis=0)
        off_max = scores[off][:, indices].max(axis=0)
        quality = on_min - off_max
        best_index = int(indices[int(np.argmax(quality))])
        fitted[label, :, 0, 0] = weights[best_index].astype(np.uint8)
        details["classes"][str(label)] = {
            "weights": weights[best_index].tolist(),
            "deltas": deltas[best_index].tolist(),
            "minimum_on_score": int(scores[on, best_index].min()),
            "maximum_off_score": int(scores[off, best_index].max()),
        }
    result = copy.deepcopy(model)
    initializers = initializer_map(result)
    initializers[weight_name].CopyFrom(numpy_helper.from_array(fitted, weight_name))
    return result, details


def exhaustive_runtime(
    model: onnx.ModelProto,
    cases: list[tuple[dict[str, object], np.ndarray, np.ndarray]],
    disabled: bool,
) -> dict[str, object]:
    try:
        session = make_session(model, disabled)
    except Exception as exc:  # noqa: BLE001
        return {"right": 0, "wrong": 0, "errors": len(cases), "session_error": repr(exc)}
    right = wrong = errors = 0
    first_failure = None
    for index, (_, input_array, expected) in enumerate(cases):
        try:
            actual = session.run(["output"], {"input": input_array})[0] > 0.0
            if np.array_equal(actual, expected):
                right += 1
            else:
                wrong += 1
                if first_failure is None:
                    first_failure = {
                        "case": index,
                        "differing_elements": int(np.count_nonzero(actual != expected)),
                    }
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"case": index, "error": repr(exc)}
    return {"right": right, "wrong": wrong, "errors": errors, "first_failure": first_failure}


def main() -> None:
    ort.set_default_logger_severity(4)
    cases = domain_cases()
    baseline = onnx.load(BASELINE)
    baseline_cost = int(cost_of(str(BASELINE))[2])
    baseline_checks = {
        "disabled": exhaustive_runtime(baseline, cases, True),
        "default": exhaustive_runtime(baseline, cases, False),
    }
    attempts: list[dict[str, object]] = []
    winners: list[dict[str, object]] = []
    groups = (SCALAR_CODES, VECTOR_CODES)
    for group in groups:
        for target, source in itertools.permutations(group, 2):
            item: dict[str, object] = {"target": target, "source": source}
            try:
                tied = tie_initializer(baseline, target, source)
                points, labels = feature_dataset(tied, cases)
                candidate, fit = fit_decoder(tied, points, labels)
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                disabled = exhaustive_runtime(candidate, cases, True)
                default = exhaustive_runtime(candidate, cases, False)
                temp = HERE / f"task226_tie_{target}_to_{source}.onnx"
                onnx.save(candidate, temp)
                memory, params, cost = cost_of(str(temp))
                item.update(
                    {
                        "status": "checked",
                        "fit": fit,
                        "cost": int(cost),
                        "memory": int(memory),
                        "params": int(params),
                        "disabled": disabled,
                        "default": default,
                        "sha256": sha256(temp),
                    }
                )
                good = (
                    cost < baseline_cost
                    and disabled["right"] == len(cases)
                    and disabled["errors"] == 0
                    and default["right"] == len(cases)
                    and default["errors"] == 0
                )
                if good:
                    winners.append({**item, "path": str(temp.relative_to(ROOT))})
                else:
                    temp.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                item.update({"status": "rejected", "reason": repr(exc)})
            attempts.append(item)
    payload = {
        "task": 226,
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": sha256(BASELINE),
        "baseline_cost": baseline_cost,
        "finite_domain": {
            "case_count": len(cases),
            "wides_count": len(list(compositions(3, 8, 1, 4)))
            + len(list(compositions(5, 6, 1, 4))),
            "talls_count": len(list(compositions(3, 8, 1, 3)))
            + len(list(compositions(5, 6, 1, 3))),
        },
        "baseline_checks": baseline_checks,
        "attempts": attempts,
        "winners": winners,
    }
    (HERE / "task226_reuse_search.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "attempts": len(attempts), "winners": winners}, indent=2))


if __name__ == "__main__":
    main()
