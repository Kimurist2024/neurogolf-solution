#!/usr/bin/env python3
"""Finite-domain proofs and counterexamples for lane 132 rewrites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent


def float16_proofs() -> dict[str, object]:
    bits = np.arange(65536, dtype=np.uint16)
    values = bits.view(np.float16)
    with np.errstate(all="ignore"):
        div2 = np.divide(values, np.float16(2.0))
        mulhalf = np.multiply(values, np.float16(0.5))
        ln2 = np.float16(0.66015625)
        staged = np.divide(np.multiply(values, ln2), np.float16(2.0))
        reassociated = np.multiply(values, np.float16(ln2 / np.float16(2.0)))
    direct_diff = np.flatnonzero(div2.view(np.uint16) != mulhalf.view(np.uint16))
    reassoc_diff = np.flatnonzero(staged.view(np.uint16) != reassociated.view(np.uint16))
    examples = [
        {
            "input_bits": f"0x{int(bits[index]):04x}",
            "staged_bits": f"0x{int(staged.view(np.uint16)[index]):04x}",
            "reassociated_bits": f"0x{int(reassociated.view(np.uint16)[index]):04x}",
        }
        for index in reassoc_diff[:8]
    ]
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [65536])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [65536])

    def binary_model(op: str, constant: float):
        graph = helper.make_graph(
            [helper.make_node(op, ["x", "constant"], ["y"])],
            op,
            [input_info],
            [output_info],
            [numpy_helper.from_array(np.asarray(constant, dtype=np.float16), "constant")],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    ort_differences = {}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        div_session = ort.InferenceSession(
            binary_model("Div", 2.0).SerializeToString(), options,
            providers=["CPUExecutionProvider"],
        )
        mul_session = ort.InferenceSession(
            binary_model("Mul", 0.5).SerializeToString(), options,
            providers=["CPUExecutionProvider"],
        )
        div_output = div_session.run(None, {"x": values})[0]
        mul_output = mul_session.run(None, {"x": values})[0]
        ort_differences[mode] = int(
            np.count_nonzero(div_output.view(np.uint16) != mul_output.view(np.uint16))
        )
    return {
        "domain": "all 65536 IEEE-754 binary16 bit patterns including NaNs",
        "div_x_by_2_vs_mul_x_by_half_bit_differences": int(direct_diff.size),
        "ort_div_vs_mul_bit_differences": ort_differences,
        "decision_div_to_mul": "equivalent_but_no_cost_reduction_so_not_emitted",
        "staged_x_mul_ln2_div2_vs_reassociated_x_mul_ln2half_bit_differences": int(
            reassoc_diff.size
        ),
        "reassociation_counterexamples": examples,
        "decision_reassociation": "reject_not_bitwise_exact_on_full_float16_domain",
    }


def boolean_proofs() -> dict[str, object]:
    flip_rows = []
    for xpose in (False, True):
        for high_r in (False, True):
            for high_c in (False, True):
                original = high_r ^ (xpose and (high_r ^ high_c))
                mux = high_c if xpose else high_r
                flip_rows.append(
                    {
                        "xpose": xpose,
                        "high_r": high_r,
                        "high_c": high_c,
                        "original": original,
                        "mux": mux,
                    }
                )
    box_rows = [
        {
            "Db": value,
            "original_2D_minus_1": 2 * int(value) - 1,
            "select": 1 if value else -1,
        }
        for value in (False, True)
    ]
    return {
        "task054_flip_truth_table": flip_rows,
        "task054_flip_equal": all(row["original"] == row["mux"] for row in flip_rows),
        "task054_box_select_truth_table": box_rows,
        "task054_box_select_equal": all(
            row["original_2D_minus_1"] == row["select"] for row in box_rows
        ),
        "runtime_decision": (
            "both rewrites rejected because this ORT build has no Where kernel for the "
            "required bool/int8 output types"
        ),
    }


def uint8_polynomial_proof() -> dict[str, object]:
    # Chunk the 256^3 proof by s to avoid holding multiple 16M arrays.
    r = np.arange(256, dtype=np.uint16)[:, None]
    h = np.arange(256, dtype=np.uint16)[None, :]
    mismatches = 0
    for shift in range(256):
        original = (((9 * r * r + 12 * h + 13) & 255) * shift + 4) & 15
        distributed = (9 * r * r * shift + 12 * h * shift + 13 * shift + 4) & 15
        mismatches += int(np.count_nonzero(original != distributed))
    return {
        "domain": "all 256^3 uint8 triples (r,h,shift)",
        "mismatches": mismatches,
        "arithmetic": "Z/256Z followed by projection to low four bits (Z/16Z)",
        "runtime_decision": (
            "identity proved, but candidate rejected because ORT has no uint8 Einsum kernel"
        ),
    }


def task204_counterexample() -> dict[str, object]:
    mask32 = (1 << 32) - 1
    vertical_sides = (1 << 0) | (1 << 2)  # one legal length-3 box interval
    p6 = vertical_sides ^ ((vertical_sides * 6) & mask32)
    p48 = p6 ^ ((p6 * 8) & mask32)
    p3072 = p48 ^ ((p48 * 64) & mask32)
    chained = p3072 ^ ((p3072 * 4096) & mask32)
    tempting_factor = 7 * 9 * 65 * 4097
    factored = (vertical_sides * tempting_factor) & mask32
    return {
        "candidate": "replace four Mul/Xor stages by multiplication by 16777215",
        "legal_boundary_mask": f"0x{vertical_sides:08x}",
        "interval": [0, 2],
        "chained_result_mod_2pow32": f"0x{chained:08x}",
        "factored_result_mod_2pow32": f"0x{factored:08x}",
        "equal": chained == factored,
        "decision": (
            "reject: XOR is not integer addition when shifted copies overlap; signed int32 "
            "intermediates also cross 2^31, so associativity cannot be assumed"
        ),
    }


def main() -> None:
    result = {
        "float16": float16_proofs(),
        "boolean": boolean_proofs(),
        "uint8_polynomial": uint8_polynomial_proof(),
        "task204_integer_prefix": task204_counterexample(),
    }
    (HERE / "audit/domain_proofs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
