#!/usr/bin/env python3
"""Prove the task333 sign absorption term-by-term for arbitrary input tensors."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx"


def equation(model: onnx.ModelProto) -> str:
    return next(
        helper.get_attribute_value(attr).decode()
        for attr in model.graph.node[0].attribute
        if attr.name == "equation"
    )


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        base = onnx.load_from_string(archive.read("task333.onnx"))
    candidate = onnx.load(CANDIDATE)
    assert len(base.graph.node) == len(candidate.graph.node) == 1
    assert base.graph.node[0].op_type == candidate.graph.node[0].op_type == "Einsum"
    before = arrays(base)
    after = arrays(candidate)
    ge = before["GE"].astype(np.float32)
    hc = before["HC"].astype(np.float32)
    ghht = before["GHHT"].astype(np.float32)
    hc_after = after["HC"].astype(np.float32)
    ghht_after = after["GHHT"].astype(np.float32)

    unchanged = sorted(set(before) - {"GE", "HC", "GHHT"})
    assert set(after) == set(unchanged) | {"HC", "GHHT"}
    assert all(np.array_equal(before[name], after[name]) for name in unchanged)
    assert np.array_equal(ge, np.asarray([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(hc_after, ge[:, None] * hc)
    assert np.array_equal(ghht_after, ghht * ge[None, :])

    first_use = np.zeros_like(hc)
    second_use = np.zeros((3, 2, 10), dtype=np.float32)
    for z in range(2):
        for d in range(10):
            first_use[z, d] = hc[z, d] * ge[z]
            assert first_use[z, d] == hc_after[z, d]
    for t in range(3):
        for u in range(2):
            for c in range(10):
                old = ghht[t, u] * hc[u, c]
                new = ghht_after[t, u] * hc_after[u, c]
                second_use[t, u, c] = new
                assert old == new

    base_inputs = list(base.graph.node[0].input)
    candidate_inputs = list(candidate.graph.node[0].input)
    assert base_inputs[7] == "GE"
    assert candidate_inputs == base_inputs[:7] + base_inputs[8:]
    old_terms = equation(base).split("->")[0].split(",")
    new_terms = equation(candidate).split("->")[0].split(",")
    assert old_terms[7] == "Z"
    assert new_terms == old_terms[:7] + old_terms[8:]
    assert equation(base).split("->")[1] == equation(candidate).split("->")[1]

    payload = {
        "task": 333,
        "baseline": "submission_base_8005.17.zip::task333.onnx",
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "proved": True,
        "rewrite": {
            "removed_operand": "GE[Z]=[1,-1]",
            "first_HC_use": "HC_new[Z,d] = GE[Z] * HC_old[Z,d]",
            "compensation": "GHHT_new[t,U] = GHHT_old[t,U] * GE[U]",
            "second_HC_use": "GHHT_new[t,U]*HC_new[U,c] = GHHT_old[t,U]*HC_old[U,c] because GE[U]^2=1",
        },
        "exhaustive_changed_factor_checks": {
            "first_use_entries": int(first_use.size),
            "second_use_entries": int(second_use.size),
            "total": int(first_use.size + second_use.size),
            "all_float32_exact": True,
        },
        "unchanged_initializer_count": len(unchanged),
        "all_other_initializers_byte_equal": True,
        "node_and_output_count_unchanged": True,
        "einsum_operand_count": {"baseline": len(base_inputs), "candidate": len(candidate_inputs)},
        "termwise_statement": "For every assignment of all Einsum indices, including every batch/output/source/green/current-cell index, the candidate monomial equals the baseline monomial exactly in real arithmetic. Therefore the full output tensor is equal for every possible input tensor; no generator distribution assumption is used.",
        "platform_residual_needed": "The operand-count/contraction plan changes 36->35, so four-configuration execution of the complete trilinear coefficient support is still required before adoption.",
    }
    (HERE / "sign_equivalence_proof.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
