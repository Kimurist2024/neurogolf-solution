#!/usr/bin/env python3
"""Build and audit the exact J2 = J1 @ T factorization for task051."""

from __future__ import annotations

import json
import sys
import copy
import itertools
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
BASE = LANE / "baseline" / "task051.onnx"
OUT = LANE / "candidates" / "task051_j1_factor.onnx"
WORK = LANE / "score_work_task051_factor"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import structure_gate  # noqa: E402
from lib import scoring  # noqa: E402


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def main() -> None:
    model = onnx.load(BASE)
    arrays = initializer_map(model)
    j1 = arrays["J1"].astype(np.float32, copy=False)
    j2 = arrays["J2"].astype(np.float32, copy=False)
    transform = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    reconstructed = j1 @ transform
    if not np.array_equal(reconstructed, j2):
        raise AssertionError("J2 != J1 @ transform in float32")

    final = model.graph.node[-1]
    if final.op_type != "Einsum" or len(final.input) != 65:
        raise AssertionError("unexpected task051 final Einsum")
    equation_attr = next(attr for attr in final.attribute if attr.name == "equation")
    equation = equation_attr.s.decode("utf-8")
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    if terms[20:24] != ["jz", "jZ", "kd", "kD"]:
        raise AssertionError(f"unexpected J1/J2 terms: {terms[20:24]}")
    if list(final.input[20:24]) != ["J1", "J2", "J1", "J2"]:
        raise AssertionError(f"unexpected J1/J2 inputs: {list(final.input[20:24])}")

    transform_name = "J12_transform"
    final.input[21] = transform_name
    final.input[23] = transform_name
    terms[21] = "zZ"
    terms[23] = "dD"
    equation_attr.s = (",".join(terms) + "->" + rhs).encode("utf-8")

    kept = [init for init in model.graph.initializer if init.name != "J2"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(transform, transform_name))

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)

    profile = scoring.score_and_verify(
        copy.deepcopy(model),
        51,
        str(WORK),
        label="task051_j1_factor",
        require_correct=False,
    )
    _, structure_reason, static_floor = structure_gate(OUT.read_bytes())

    existing_3x3 = {
        name: value
        for name, value in arrays.items()
        if tuple(value.shape) == (3, 3)
    }
    exact_existing = [
        name
        for name, value in existing_3x3.items()
        if np.array_equal(value.astype(np.float32, copy=False), transform)
    ]

    diagonal_matches: list[dict[str, object]] = []
    for name, value in arrays.items():
        if tuple(value.shape) != (3, 3, 3):
            continue
        for axis in range(3):
            diag = np.diagonal(value, axis1=(axis + 1) % 3, axis2=(axis + 2) % 3)
            for perm in ((0, 1), (1, 0)):
                candidate = np.transpose(diag, perm).astype(np.float32, copy=False)
                if np.array_equal(candidate, transform):
                    diagonal_matches.append(
                        {"initializer": name, "axis": axis, "transpose": list(perm)}
                    )

    rank3 = {
        name: value.astype(np.float32, copy=False)
        for name, value in arrays.items()
        if tuple(value.shape) == (3, 3, 3)
    }
    absorption_matches: list[dict[str, object]] = []
    # The only other users of final-Einsum labels Z and D are rank-3 tensors.
    # Absorbing T into such a tensor creates a 27-element tensor. Check whether
    # that tensor already exists, including axis permutations.
    for source_name, source in rank3.items():
        absorbed = np.einsum("ab,bcd->acd", transform, source)
        for existing_name, existing in rank3.items():
            for permutation in itertools.permutations(range(3)):
                if np.array_equal(absorbed, np.transpose(existing, permutation)):
                    absorption_matches.append(
                        {
                            "source": source_name,
                            "existing": existing_name,
                            "existing_axis_permutation": list(permutation),
                        }
                    )

    result = {
        "task": 51,
        "baseline_cost": 283,
        "candidate": str(OUT.relative_to(ROOT)),
        "factorization": {
            "identity": "J2 = J1 @ J12_transform",
            "float32_exact": True,
            "removed_parameters": int(j2.size),
            "added_parameters": int(transform.size),
            "parameter_delta": int(transform.size - j2.size),
            "final_einsum_inputs_before": 65,
            "final_einsum_inputs_after": len(final.input),
        },
        "profile": profile,
        "structure": {
            "pass": structure_reason == "pass",
            "reason": structure_reason,
            "static_floor": static_floor,
        },
        "existing_3x3_exact_transform_matches": exact_existing,
        "existing_3x3x3_diagonal_transform_matches": diagonal_matches,
        "existing_label_absorption": {
            "rank3_existing_matches": absorption_matches,
            "removed_parameters": int(j2.size),
            "required_new_rank3_parameters": 27,
            "parameter_delta": 27 - int(j2.size),
            "final_einsum_inputs_after": 65,
            "decision": "reject_not_cheaper",
        },
        "decision": (
            "reject_structure_giant_einsum"
            if structure_reason != "pass"
            else "eligible_for_fresh_validation"
        ),
    }
    (LANE / "task051_factor_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
