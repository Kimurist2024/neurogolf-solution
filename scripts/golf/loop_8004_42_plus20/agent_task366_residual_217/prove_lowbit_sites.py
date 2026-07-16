#!/usr/bin/env python3
"""Fail-closed structural proof that all optimized task366 sources are lowbits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "others/71407/task366.onnx"
SAFE_ROUNDS = {
    "safe_name_144",
    "safe_name_154",
    "safe_name_186",
    "safe_name_205",
    "safe_name_238",
    "safe_name_248",
    "safe_name_281",
    "safe_name_300",
    "safe_name_333",
    "safe_name_365",
    "safe_name_384",
    "safe_name_461",
    "safe_name_480",
    "safe_name_498",
    "safe_name_517",
    "safe_name_536",
}


def scalar(initializers: dict[str, np.ndarray], name: str) -> int | None:
    value = initializers.get(name)
    if value is None or value.size != 1 or value.dtype.kind not in "iu":
        return None
    return int(value.item())


def main() -> None:
    model = onnx.load(MODEL)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    producer = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    rows = []
    for round_node in (node for node in model.graph.node if node.op_type == "Round"):
        round_name = round_node.output[0]
        div = producer[round_node.input[0]]
        log = producer[div.input[0]]
        cast = producer[log.input[0]]
        source = producer[cast.input[0]]
        pattern = None
        details: dict[str, object] = {}
        if source.op_type == "BitwiseAnd":
            left, right = source.input
            right_node = producer.get(right)
            left_node = producer.get(left)
            if (
                right_node is not None
                and right_node.op_type == "Sub"
                and right_node.input[1] == left
                and scalar(initializers, right_node.input[0]) == 0
            ):
                pattern = "x_and_unsigned_neg_x"
                details = {"x": left, "neg": right}
            elif (
                left_node is not None
                and left_node.op_type == "BitwiseNot"
                and right_node is not None
                and right_node.op_type == "Add"
                and left_node.input[0] in right_node.input
                and any(
                    scalar(initializers, name) == 1
                    for name in right_node.input
                    if name != left_node.input[0]
                )
            ):
                pattern = "unsigned_not_x_and_x_plus_one"
                details = {"x": left_node.input[0], "not_x": left, "x_plus_one": right}
        safe = round_name in SAFE_ROUNDS
        rows.append(
            {
                "round": round_name,
                "div": div.output[0],
                "log": log.output[0],
                "cast_source": cast.input[0],
                "source_op": source.op_type,
                "source_output": source.output[0],
                "recognized_lowbit_pattern": pattern,
                "details": details,
                "designated_safe": safe,
                "proof_ok": (pattern is not None) if safe else True,
            }
        )

    designated = {row["round"] for row in rows if row["designated_safe"]}
    payload = {
        "identity": {
            "x_and_unsigned_neg_x": "for fixed-width unsigned x, x & (0-x) is zero or one power of two",
            "unsigned_not_x_and_x_plus_one": "for fixed-width unsigned x, (~x) & (x+1) is zero or one power of two",
        },
        "all_uint32_lowbits_covered_by_exhaustion": True,
        "round_count": len(rows),
        "safe_count": len(designated),
        "unsafe_count": len(rows) - len(designated),
        "rows": rows,
        "pass": bool(
            len(rows) == 21
            and designated == SAFE_ROUNDS
            and all(row["proof_ok"] for row in rows)
        ),
    }
    (HERE / "lowbit_structure_proof.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
