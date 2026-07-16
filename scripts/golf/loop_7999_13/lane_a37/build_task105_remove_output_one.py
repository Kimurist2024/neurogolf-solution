#!/usr/bin/env python3
"""Remove task105's redundant singleton output initializer from six Einsums."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


SOURCE = HERE.parent / "lane_initializer_contraction_wave17" / "task105_combined.onnx"
OUTPUT = HERE / "task105_remove_output_one.onnx"
REPORT = HERE / "task105_remove_output_one_build.json"


def main() -> None:
    model = onnx.load(SOURCE)
    rewrites = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == "one_f"]
        if not positions:
            continue
        if node.op_type != "Einsum" or len(positions) != 1:
            raise RuntimeError("unexpected one_f consumer")
        attr = next(attr for attr in node.attribute if attr.name == "equation")
        lhs, rhs = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        position = positions[0]
        if terms[position] != "n" or rhs != "n" or "b" not in terms[0]:
            raise RuntimeError("unexpected singleton-output equation")
        del terms[position]
        del node.input[position]
        attr.s = (",".join(terms) + "->b").encode("ascii")
        rewrites += 1
    if rewrites != 6:
        raise RuntimeError(f"unexpected rewrite count: {rewrites}")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining["one_f"]:
        raise RuntimeError("one_f remains used")
    kept = [item for item in model.graph.initializer if item.name != "one_f"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    base_memory, base_parameters, base_cost = map(int, cost_of(str(SOURCE)))
    memory, parameters, cost = map(int, cost_of(str(OUTPUT)))
    if (base_memory, base_parameters, base_cost) != (89, 106, 195):
        raise RuntimeError("unexpected source cost")
    if (memory, parameters, cost) != (89, 105, 194):
        raise RuntimeError("unexpected candidate cost")
    report = {
        "task": 105,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "rewrite": "replace the synthetic singleton output n with the real static batch axis b",
        "einsum_rewrites": rewrites,
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
