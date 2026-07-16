#!/usr/bin/env python3
"""Four-config exact-pass-through audit for task066 residual lane 208."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / "others/71407/task066.onnx"
CANDIDATE = HERE / "task066_residual_cost551.onnx"
EXPECTED_PARENT_SHA = "2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b"
EXPECTED_CANDIDATE_SHA = "622b3b28271806949bb18e8b9517335d49cb0383410caf36a19e064d95798dd3"
FRESH = ((66_208_001, 2000), (66_208_002, 2000))
TRACE = ("Gv", "Gh", "Gf", "G", "Ov", "Oh", "O", "aMask", "bMask", "selF", "selQ", "ti")

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_review_task066_206"))
import audit_review as review206  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def graph_delta(parent_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    parent = onnx.load_model_from_string(parent_data)
    candidate = onnx.load_model_from_string(candidate_data)
    changed = []
    for index, (left, right) in enumerate(zip(parent.graph.node, candidate.graph.node, strict=True)):
        if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True):
            changed.append(
                {
                    "index": index,
                    "output": list(right.output),
                    "parent_inputs": list(left.input),
                    "candidate_inputs": list(right.input),
                    "parent_equation": next(
                        onnx.helper.get_attribute_value(attr).decode()
                        for attr in left.attribute
                        if attr.name == "equation"
                    ),
                    "candidate_equation": next(
                        onnx.helper.get_attribute_value(attr).decode()
                        for attr in right.attribute
                        if attr.name == "equation"
                    ),
                }
            )
    left_init = {item.name: item for item in parent.graph.initializer}
    right_init = {item.name: item for item in candidate.graph.initializer}
    common_equal = all(
        left_init[name].SerializeToString(deterministic=True)
        == right_init[name].SerializeToString(deterministic=True)
        for name in set(left_init) & set(right_init)
    )
    left_shell = copy.deepcopy(parent)
    right_shell = copy.deepcopy(candidate)
    del left_shell.graph.node[:]
    del right_shell.graph.node[:]
    del left_shell.graph.initializer[:]
    del right_shell.graph.initializer[:]
    shell_equal = left_shell.SerializeToString(deterministic=True) == right_shell.SerializeToString(deterministic=True)
    exact = bool(
        [row["index"] for row in changed] == [22, 23]
        and [row["output"] for row in changed] == [["Gv"], ["Gh"]]
        and set(left_init) - set(right_init) == {"greenhalf10"}
        and not (set(right_init) - set(left_init))
        and common_equal
        and shell_equal
    )
    return {
        "changed_nodes": changed,
        "removed_initializers": sorted(set(left_init) - set(right_init)),
        "added_initializers": sorted(set(right_init) - set(left_init)),
        "common_initializers_proto_equal": common_equal,
        "all_other_model_fields_equal": shell_equal,
        "whitelist_exact": exact,
    }


def selector_proof(parent_data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(parent_data)
    init = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    rebuilt = np.einsum(
        "ad,ed,aj,aj,ek,k,fl,fl,fm,m->d",
        init["Uchan"],
        init["Vchan"],
        init["Trow"],
        init["Tcol"],
        init["Tcol"],
        init["z1"],
        init["Uchan"],
        init["Vchan"],
        init["Tcol"],
        init["z1"],
        optimize=True,
    )
    original = init["greenhalf10"]
    return {
        "original": original.tolist(),
        "rebuilt": rebuilt.tolist(),
        "numeric_float32_equal": np.array_equal(original, rebuilt),
        "signed_zero_only_raw_difference": bool(
            np.array_equal(original, rebuilt)
            and original.tobytes() != rebuilt.astype(np.float32, copy=False).tobytes()
        ),
        "identity": "(Uchan[0]*Vchan[2])*dot(Uchan[2],Vchan[2])=(-e3)*(-1)=e3",
        "exact_arithmetic": "all selector factors and partial sums are in {-1,0,1}",
    }


def evaluate(
    parent: ort.InferenceSession,
    candidate: ort.InferenceSession,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "parent_gold": 0,
        "candidate_gold": 0,
        "final_raw_equal": 0,
        "final_threshold_equal": 0,
        "trace_raw_equal": {name: 0 for name in TRACE},
        "runtime_errors": 0,
        "parent_final_nonfinite": 0,
        "candidate_final_nonfinite": 0,
        "candidate_trace_nonfinite": {name: 0 for name in TRACE},
        "first_difference": None,
    }
    for index, example in enumerate(cases):
        converted = review206.scoring.convert_to_numpy(example)
        if converted is None:
            continue
        result["valid"] += 1
        outputs = {}
        for label, session in (("parent", parent), ("candidate", candidate)):
            try:
                outputs[label] = [
                    np.asarray(value)
                    for value in session.run(None, {session.get_inputs()[0].name: converted["input"]})
                ]
            except Exception as exc:  # noqa: BLE001
                result["runtime_errors"] += 1
                result["first_difference"] = result["first_difference"] or {
                    "case": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) != 2:
            continue
        p_final, *p_trace = outputs["parent"]
        c_final, *c_trace = outputs["candidate"]
        expected = converted["output"].astype(bool)
        result["parent_gold"] += int(np.array_equal(p_final > 0, expected))
        result["candidate_gold"] += int(np.array_equal(c_final > 0, expected))
        final_equal = p_final.dtype == c_final.dtype and p_final.shape == c_final.shape and p_final.tobytes() == c_final.tobytes()
        result["final_raw_equal"] += int(final_equal)
        result["final_threshold_equal"] += int(np.array_equal(p_final > 0, c_final > 0))
        result["parent_final_nonfinite"] += int(p_final.size - np.count_nonzero(np.isfinite(p_final)))
        result["candidate_final_nonfinite"] += int(c_final.size - np.count_nonzero(np.isfinite(c_final)))
        trace_equal = True
        for name, left, right in zip(TRACE, p_trace, c_trace, strict=True):
            equal = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
            result["trace_raw_equal"][name] += int(equal)
            trace_equal &= equal
            if right.dtype.kind == "f":
                result["candidate_trace_nonfinite"][name] += int(
                    right.size - np.count_nonzero(np.isfinite(right))
                )
        if not (final_equal and trace_equal):
            result["first_difference"] = result["first_difference"] or {
                "case": index,
                "final_equal": final_equal,
                "different_trace": [
                    name
                    for name, left, right in zip(TRACE, p_trace, c_trace, strict=True)
                    if left.tobytes() != right.tobytes()
                ],
            }
    valid = result["valid"]
    result["pass"] = bool(
        valid == len(cases)
        and result["final_raw_equal"] == valid
        and result["final_threshold_equal"] == valid
        and all(count == valid for count in result["trace_raw_equal"].values())
        and result["runtime_errors"] == 0
        and result["parent_final_nonfinite"] == 0
        and result["candidate_final_nonfinite"] == 0
    )
    return result


def main() -> None:
    ort.set_default_logger_severity(4)
    parent_data = PARENT.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    assert sha256(parent_data) == EXPECTED_PARENT_SHA
    assert sha256(candidate_data) == EXPECTED_CANDIDATE_SHA
    selector = selector_proof(parent_data)
    assert selector["numeric_float32_equal"]

    parent_trace = review206.traced_model(parent_data, TRACE)
    candidate_trace = review206.traced_model(candidate_data, TRACE)
    known = review206.known_cases()
    fresh = [(seed, review206.fresh_cases(seed, count)) for seed, count in FRESH]
    evaluations: dict[str, Any] = {"known": {}, "fresh": [{"seed": seed, "modes": {}} for seed, _ in fresh]}
    rows = []
    for disable, threads, label in review206.CONFIGS:
        parent_session = review206.make_session(parent_trace, disable, threads)
        candidate_session = review206.make_session(candidate_trace, disable, threads)
        row = evaluate(parent_session, candidate_session, known)
        evaluations["known"][label] = row
        rows.append(row)
        print(
            f"known {label}: gold={row['candidate_gold']}/{row['valid']} "
            f"raw={row['final_raw_equal']} Gv={row['trace_raw_equal']['Gv']} Gh={row['trace_raw_equal']['Gh']}",
            flush=True,
        )
        for index, (seed, cases) in enumerate(fresh):
            row = evaluate(parent_session, candidate_session, cases)
            evaluations["fresh"][index]["modes"][label] = row
            rows.append(row)
            print(
                f"fresh {seed} {label}: gold={row['candidate_gold']}/{row['valid']} "
                f"raw={row['final_raw_equal']} Gv={row['trace_raw_equal']['Gv']} Gh={row['trace_raw_equal']['Gh']}",
                flush=True,
            )

    profiles = {
        "parent": review206.official_profile(parent_data, "parent_residual208"),
        "candidate": review206.official_profile(candidate_data, "candidate_residual208"),
    }
    static = {
        "parent": review206.static_audit(parent_data),
        "candidate": review206.static_audit(candidate_data),
    }
    shapes = [review206.runtime_shape_truth(candidate_data, disable) for disable in (True, False)]
    delta = graph_delta(parent_data, candidate_data)
    summary = {
        "strict_lower": profiles["candidate"]["cost"] < profiles["parent"]["cost"],
        "cost_delta": profiles["parent"]["cost"] - profiles["candidate"]["cost"],
        "score_gain": math.log(profiles["parent"]["cost"] / profiles["candidate"]["cost"]),
        "selector_exact": selector["numeric_float32_equal"],
        "graph_delta_exact": delta["whitelist_exact"],
        "static_pass": static["candidate"]["pass"],
        "truthful_shapes": all(row["truthful"] for row in shapes),
        "known_gold_four_configs": all(
            row["candidate_gold"] == row["valid"] for row in evaluations["known"].values()
        ),
        "all_raw_pass_through": all(row["pass"] for row in rows),
        "runtime_errors": sum(row["runtime_errors"] for row in rows),
        "final_nonfinite": sum(row["candidate_final_nonfinite"] for row in rows),
    }
    summary["accepted"] = bool(
        summary["strict_lower"]
        and summary["selector_exact"]
        and summary["graph_delta_exact"]
        and summary["static_pass"]
        and summary["truthful_shapes"]
        and summary["known_gold_four_configs"]
        and summary["all_raw_pass_through"]
        and summary["runtime_errors"] == 0
        and summary["final_nonfinite"] == 0
    )
    result = {
        "parent": {"sha256": sha256(parent_data), "profile": profiles["parent"]},
        "candidate": {"sha256": sha256(candidate_data), "profile": profiles["candidate"]},
        "selector_proof": selector,
        "graph_delta": delta,
        "static": static,
        "runtime_shapes": shapes,
        "evaluations": evaluations,
        "summary": summary,
    }
    print("AUDIT_SUMMARY")
    print(json.dumps(review206.safe(result["summary"]), indent=2))
    print("AUDIT_DETAIL")
    print(json.dumps(review206.safe(result), indent=2))
    assert summary["accepted"]


if __name__ == "__main__":
    main()
