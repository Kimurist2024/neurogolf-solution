#!/usr/bin/env python3
"""Replace task013 Qch by Qor using one exact shared reduction axis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


SOURCE = HERE.parent / "lane_initializer_contraction_wave17" / "task013_combined.onnx"
OUTPUT = HERE / "task013_qch_from_qor_shared_reduction.onnx"
REPORT = HERE / "task013_qch_from_qor_shared_reduction_build.json"


def equation(node: onnx.NodeProto) -> str:
    return next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")


def measure(path: Path) -> tuple[int, int, int]:
    memory, parameters, cost = cost_of(str(path))
    return int(memory), int(parameters), int(cost)


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    qch = arrays["Qch"]
    qor = arrays["Qor"]
    rebuilt = np.einsum("arbcc->abc", qor, optimize=False)
    if not np.array_equal(rebuilt, qch):
        raise RuntimeError("Qor contraction is not exactly Qch")
    # The contracted axis is supported only at zero. Therefore sharing the
    # unnamed ellipsis axis among repeated Qch replacements is also exact.
    contribution = np.einsum("arbcc->rabc", qor, optimize=False)
    if np.any(contribution[1:] != 0) or not np.array_equal(contribution[0], qch):
        raise RuntimeError("Qor contraction does not have singleton support")

    replacements = 0
    batch_ellipsis_rewrites = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == "Qch"]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise RuntimeError("Qch has a non-Einsum consumer")
        attr = next(attr for attr in node.attribute if attr.name == "equation")
        lhs, rhs = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        used = set("".join(terms) + rhs)
        available = [label for label in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" if label not in used]
        if not available:
            if terms[0] != "bshw" or rhs != "bkhw":
                raise RuntimeError("no free label and no recognized batch axis")
            terms[0] = "...shw"
            rhs = "...khw"
            reduction_label = "b"
            batch_ellipsis_rewrites += 1
        else:
            reduction_label = available[0]
        for position in positions:
            target = terms[position]
            if len(target) != 3 or "..." in target:
                raise RuntimeError(f"unexpected Qch term: {target}")
            terms[position] = target[0] + reduction_label + target[1] + target[2] + target[2]
            node.input[position] = "Qor"
            replacements += 1
        attr.s = (",".join(terms) + "->" + rhs).encode("ascii")

    if replacements != 11:
        raise RuntimeError(f"unexpected replacement count: {replacements}")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining["Qch"]:
        raise RuntimeError("Qch remains used")
    kept = [item for item in model.graph.initializer if item.name != "Qch"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)

    base_memory, base_parameters, base_cost = measure(SOURCE)
    memory, parameters, cost = measure(OUTPUT)
    if (base_memory, base_parameters, base_cost) != (558, 181, 739):
        raise RuntimeError("unexpected A37 source cost")
    if cost >= base_cost:
        raise RuntimeError("candidate did not improve")
    report = {
        "task": 13,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "rewrite": "Qch[c,a,b] = sum_r Qor[c,r,a,b,b], with one shared reduction label whose support is exactly r=0",
        "replacement_count": replacements,
        "batch_ellipsis_rewrites": batch_ellipsis_rewrites,
        "baseline_memory": base_memory,
        "baseline_parameters": base_parameters,
        "baseline_cost": base_cost,
        "candidate_memory": memory,
        "candidate_parameters": parameters,
        "candidate_cost": cost,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
