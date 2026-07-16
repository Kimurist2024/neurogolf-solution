#!/usr/bin/env python3
"""Build algebraically exact, root-isolated Wave B4 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def build_task333_reuse() -> Path:
    """Reuse zero-padded D5/D2 in place of their C5/C2 prefixes."""
    model = onnx.load(HERE / "baseline_task333.onnx")
    node = model.graph.node[0]
    if node.op_type != "Einsum":
        raise RuntimeError("task333 baseline is no longer the expected one-node Einsum")

    replacements = {"C5": "D5", "C2": "D2"}
    for index, name in enumerate(node.input):
        if name in replacements:
            node.input[index] = replacements[name]

    kept = [init for init in model.graph.initializer if init.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidate_task333_reuse_d.onnx"
    onnx.save(model, output)
    return output


def build_task328_sum_coreb() -> Path:
    """Pre-sum CoreB's private first axis in every final-Einsum use."""
    model = onnx.load(HERE / "baseline_task328.onnx")
    node = model.graph.node[-1]
    if node.op_type != "Einsum":
        raise RuntimeError("task328 baseline no longer ends in the expected Einsum")

    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    equation = equation_attr.s.decode("ascii")
    lhs, rhs = equation.split("->")
    subscripts = lhs.split(",")
    if len(subscripts) != len(node.input):
        raise RuntimeError("task328 Einsum operand/equation mismatch")
    for index, name in enumerate(node.input):
        if name == "CoreB":
            if len(subscripts[index]) != 3:
                raise RuntimeError(f"unexpected CoreB subscript: {subscripts[index]}")
            subscripts[index] = subscripts[index][1:]
    equation_attr.s = (",".join(subscripts) + "->" + rhs).encode("ascii")

    for index, init in enumerate(model.graph.initializer):
        if init.name == "CoreB":
            core = numpy_helper.to_array(init)
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(core.sum(axis=0), name="CoreB")
            )
            break
    else:
        raise RuntimeError("CoreB initializer missing")

    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidate_task328_sum_coreb.onnx"
    onnx.save(model, output)
    return output


def build_task107_shared_coefficients() -> Path:
    """Factor Acoef/Bcoef through their three shared 5-D slices.

    Acoef and Bcoef are 2x2x2x2x2x2 tensors.  Their four first-axis
    slices contain only three distinct tensors: Acoef[0], the common
    Acoef[1] == Bcoef[0], and Bcoef[1].  Replacing each six-index operand
    by a one-hot selector and a shared three-slice bank is an exact tensor
    factorization and saves 20 initializer elements without introducing a
    runtime tensor.
    """
    model = onnx.load(HERE / "baseline_task107.onnx")
    node = model.graph.node[-1]
    if node.op_type != "Einsum":
        raise RuntimeError("task107 baseline no longer ends in the expected Einsum")

    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    acoef = initializers["Acoef"]
    bcoef = initializers["Bcoef"]
    if not (acoef.shape == bcoef.shape == (2, 2, 2, 2, 2, 2)):
        raise RuntimeError("unexpected task107 coefficient shapes")
    if not (acoef[1] == bcoef[0]).all():
        raise RuntimeError("task107 shared coefficient slice no longer matches")

    bank = numpy_helper.from_array(
        np.stack((acoef[0], acoef[1], bcoef[1]), axis=0), name="ABbank"
    )
    selector_a = numpy_helper.from_array(
        np.array([[1, 0, 0], [0, 1, 0]], dtype=acoef.dtype),
        name="ABselA",
    )
    selector_b = numpy_helper.from_array(
        np.array([[0, 1, 0], [0, 0, 1]], dtype=acoef.dtype),
        name="ABselB",
    )

    kept = [init for init in model.graph.initializer if init.name not in {"Acoef", "Bcoef"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend((*kept, bank, selector_a, selector_b))

    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    equation = equation_attr.s.decode("ascii")
    lhs, rhs = equation.split("->")
    subscripts = lhs.split(",")
    inputs = list(node.input)
    if len(subscripts) != len(inputs):
        raise RuntimeError("task107 Einsum operand/equation mismatch")

    # Every occurrence needs a private contraction label; sharing it would
    # couple independent sums inside the monolithic Einsum.
    private_labels = iter(("q", "z", "U", "V"))
    new_inputs: list[str] = []
    new_subscripts: list[str] = []
    replacements = 0
    for name, subscript in zip(inputs, subscripts):
        if name not in {"Acoef", "Bcoef"}:
            new_inputs.append(name)
            new_subscripts.append(subscript)
            continue
        label = next(private_labels)
        selector = "ABselA" if name == "Acoef" else "ABselB"
        new_inputs.extend((selector, "ABbank"))
        new_subscripts.extend((subscript[0] + label, label + subscript[1:]))
        replacements += 1
    if replacements != 4:
        raise RuntimeError(f"expected four task107 coefficient uses, got {replacements}")

    del node.input[:]
    node.input.extend(new_inputs)
    equation_attr.s = (",".join(new_subscripts) + "->" + rhs).encode("ascii")

    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidate_task107_shared_coefficients.onnx"
    onnx.save(model, output)
    return output


def build_task107_shared_coefficients_rank4() -> Path:
    """Further factor the shared coefficient bank across a zero support cut.

    ABbank has shape 3x2x2x2x2x2.  After grouping its first three and last
    three axes, the 12x8 matrix has nonzero columns only at 3, 5, 6, and 7.
    It therefore factors exactly into a 12x4 value bank and a 4x8 one-hot
    support tensor.  This saves another 16 parameters with no runtime node.
    """
    shared_path = build_task107_shared_coefficients()
    model = onnx.load(shared_path)
    node = model.graph.node[-1]
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    bank = initializers["ABbank"]
    if bank.shape != (3, 2, 2, 2, 2, 2):
        raise RuntimeError(f"unexpected ABbank shape: {bank.shape}")
    matrix = bank.reshape(12, 8)
    pivots = (3, 5, 6, 7)
    left = matrix[:, pivots].reshape(3, 2, 2, 4)
    right = np.zeros((4, 8), dtype=bank.dtype)
    right[np.arange(4), pivots] = 1
    if not np.array_equal(left.reshape(12, 4) @ right, matrix):
        raise RuntimeError("task107 ABbank rank-4 reconstruction failed")

    left_init = numpy_helper.from_array(left, name="ABleft")
    right_init = numpy_helper.from_array(right.reshape(4, 2, 2, 2), name="ABright")
    kept = [init for init in model.graph.initializer if init.name != "ABbank"]
    del model.graph.initializer[:]
    model.graph.initializer.extend((*kept, left_init, right_init))

    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    lhs, rhs = equation_attr.s.decode("ascii").split("->")
    subscripts = lhs.split(",")
    inputs = list(node.input)
    if len(subscripts) != len(inputs):
        raise RuntimeError("factorized task107 Einsum operand/equation mismatch")

    private_labels = iter(("j", "k", "l", "W"))
    new_inputs: list[str] = []
    new_subscripts: list[str] = []
    replacements = 0
    for name, subscript in zip(inputs, subscripts):
        if name != "ABbank":
            new_inputs.append(name)
            new_subscripts.append(subscript)
            continue
        if len(subscript) != 6:
            raise RuntimeError(f"unexpected ABbank subscript: {subscript}")
        label = next(private_labels)
        new_inputs.extend(("ABleft", "ABright"))
        new_subscripts.extend((subscript[:3] + label, label + subscript[3:]))
        replacements += 1
    if replacements != 4:
        raise RuntimeError(f"expected four ABbank uses, got {replacements}")
    del node.input[:]
    node.input.extend(new_inputs)
    equation_attr.s = (",".join(new_subscripts) + "->" + rhs).encode("ascii")

    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidate_task107_shared_coefficients_rank4.onnx"
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "333-reuse",
            "328-sum-coreb",
            "107-shared-coefficients",
            "107-shared-coefficients-rank4",
        ],
    )
    args = parser.parse_args()
    builders = {
        "333-reuse": build_task333_reuse,
        "328-sum-coreb": build_task328_sum_coreb,
        "107-shared-coefficients": build_task107_shared_coefficients,
        "107-shared-coefficients-rank4": build_task107_shared_coefficients_rank4,
    }
    print(builders[args.task]())


if __name__ == "__main__":
    main()
