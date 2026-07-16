#!/usr/bin/env python3
"""Audit the requested exact algebraic/factor reconstruction avenues."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


def load(name: str, filename: str):
    path = HERE.parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initializer_inventory(model: onnx.ModelProto) -> dict[str, object]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    identical = []
    for index, (left_name, left) in enumerate(initializers.items()):
        for right_name, right in list(initializers.items())[index + 1 :]:
            if left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right):
                identical.append([left_name, right_name])
    return {
        "uses": {name: uses[name] for name in initializers},
        "unused": [name for name in initializers if uses[name] == 0],
        "identical_same_shape": identical,
    }


def task224_factor_audit() -> dict[str, object]:
    model = onnx.load(HERE / "baseline" / "task224.onnx")
    shared = load("b22_esof", "einsum_shared_operand_fusion.py")
    identity = load("b22_identity", "einsum_remove_identity_operand.py")
    inline = load("b22_inline", "einsum_inline_single_use.py")
    counts, locations = shared.uses(model.graph)
    plans = {
        tensor.name: bool(shared.plan_source(model, tensor.name, counts, locations))
        for tensor in model.graph.initializer
        if counts[tensor.name] >= 2
    }
    return {
        "initializer_inventory": initializer_inventory(model),
        "shared_operand_exact_fusion_plans": plans,
        "shared_operand_exact_fusion_candidates": sum(plans.values()),
        "identity_operand_removals": identity.opportunities(model),
        "single_use_einsum_inline_cap128": inline.opportunities(model, 128),
        "same_shape_reuse_attempts": [
            {
                "pair": "H0B/H1B",
                "parameter_saving": 6,
                "known_result_both_directions": "0/266 in both ORT modes",
            },
            {
                "pair": "Cdiag/Csum",
                "parameter_saving": 4,
                "known_result_both_directions": "0/266 in both ORT modes",
            },
        ],
        "decision": "no_exact_factor_or_reuse_candidate",
    }


def make_session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def task400_feature_codes() -> dict[int, int]:
    model = onnx.load(HERE / "baseline" / "task400.onnx")
    probe = copy.deepcopy(model)
    probe.graph.output.append(
        helper.make_tensor_value_info("patch3", TensorProto.INT8, [1, 1, 5, 5])
    )
    session = make_session(probe)
    mapping: dict[int, set[int]] = {}
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(400)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError("known conversion failed")
            patch = session.run(["patch3"], {"input": benchmark["input"]})[0][0, 0]
            labels = np.asarray(example["output"], dtype=np.int64)
            for code, label in zip(patch.reshape(-1), labels.reshape(-1)):
                mapping.setdefault(int(code), set()).add(int(label))
    conflicts = {str(code): sorted(labels) for code, labels in mapping.items() if len(labels) != 1}
    if conflicts:
        raise RuntimeError(f"code conflicts: {conflicts}")
    return {code: next(iter(labels)) for code, labels in sorted(mapping.items())}


def task400_one_feature_search() -> dict[str, object]:
    model = onnx.load(HERE / "baseline" / "task400.onnx")
    arrays = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    mapping = task400_feature_codes()
    codes = np.asarray(list(mapping), dtype=np.int32)
    labels = np.asarray(list(mapping.values()), dtype=np.int64)
    scale = float(arrays["scale"])
    decoder_values = np.arange(-128, 128, dtype=np.int32)
    survivors = []
    class_feasible_counts: dict[str, int] = {str(label): 0 for label in range(10)}
    for multiplier in range(-128, 128):
        # The existing Mul has int8 output, so multiplication wraps modulo 256.
        feature = ((codes * multiplier + 128) % 256) - 128
        all_classes = True
        for label in range(10):
            desired = labels == label
            accumulator = feature[:, None] * decoder_values[None, :]
            prediction = np.rint(scale * accumulator).clip(-128, 127) > 0
            feasible = np.all(prediction == desired[:, None], axis=0)
            count = int(np.count_nonzero(feasible))
            if count:
                class_feasible_counts[str(label)] += 1
            else:
                all_classes = False
                break
        if all_classes:
            survivors.append(multiplier)
    return {
        "initializer_inventory": initializer_inventory(model),
        "observed_generator_code_to_color": {str(code): label for code, label in mapping.items()},
        "multipliers_enumerated": 256,
        "decoder_weights_per_class_enumerated": 256,
        "one_feature_survivors": survivors,
        "one_feature_survivor_count": len(survivors),
        "current_two_feature_intermediate_elements": 50,
        "folding_would_save_intermediate_elements": 25,
        "proof_note": (
            "A one-dimensional QLinearConv decoder is a half-line test after a shared zero point; "
            "it cannot make all nine reachable non-blue colors separate one-vs-rest classes. "
            "The exhaustive int8 multiplier/weight check confirms zero survivors."
        ),
        "decision": "two_wraparound_features_are_semantically_essential",
    }


def main() -> None:
    payload = {
        "task224": task224_factor_audit(),
        "task400": task400_one_feature_search(),
    }
    (HERE / "algebra_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
