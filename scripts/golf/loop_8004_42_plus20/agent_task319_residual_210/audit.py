#!/usr/bin/env python3
"""Fail-closed audit of the cost-975 task319 residual candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task319.onnx"
CANDIDATE = HERE / "candidates/task319_combined_best_local.onnx"
AUTHORITY_SHA256 = "ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4"
CANDIDATE_SHA256 = "a4e0531b0a3dc08355d429ba9a049f8dbd076b203a8ddb8f88c635bedf9f31cd"


def load_helpers() -> Any:
    path = HERE.parent / "agent_review_task319_207/audit.py"
    spec = importlib.util.spec_from_file_location("task319_review207_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FRESH_SEEDS = (319_210_011, 319_210_029)
    module.FRESH_PER_SEED = 1_500
    return module


H = load_helpers()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official_profiles(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for label, data in (("authority", authority_data), ("candidate", candidate_data)):
        with tempfile.TemporaryDirectory(prefix=f"task319_210_{label}_", dir=HERE) as workdir:
            profiles[label] = H.scoring.score_and_verify(
                onnx.load_model_from_string(data), 319, workdir,
                label=f"task319_210_{label}", require_correct=False,
            )
    profiles["cost_delta"] = profiles["candidate"]["cost"] - profiles["authority"]["cost"]
    profiles["score_delta"] = math.log(profiles["authority"]["cost"] / profiles["candidate"]["cost"])
    return profiles


def formal_checks(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    an = {x.output[0]: x for x in authority.graph.node}
    cn = {x.output[0]: x for x in candidate.graph.node}
    ai = {x.name: numpy_helper.to_array(x) for x in authority.graph.initializer}
    ci = {x.name: numpy_helper.to_array(x) for x in candidate.graph.initializer}

    argmax_failures = 0
    for index in range(10):
        old = np.arange(10, dtype=np.uint8) == np.uint8(index)
        new = np.arange(10, dtype=np.int64) == np.int64(index)
        argmax_failures += int(not np.array_equal(old, new))

    # eq1 has fixed runtime shape [1,1,2].  Check every boolean assignment.
    reduce_failures = 0
    for bits in range(4):
        values = np.asarray([(bits >> i) & 1 for i in range(2)], dtype=bool).reshape(1, 1, 2)
        old = np.squeeze(np.min(values, axis=2, keepdims=True))
        new = np.min(values, keepdims=False)
        reduce_failures += int(not np.array_equal(old, new))

    terminal_failures = 0
    for background in range(10):
        mask = np.arange(10) == background
        old = np.ones(10, np.uint8)
        old[background] = 0
        new = np.where(mask, np.uint8(0), np.uint8(1))
        terminal_failures += int(not np.array_equal(old, new))

    graph = {
        "authority_argmax_cast_then_equal": an["safe_name_28"].op_type == "CastLike" and an["safe_name_29"].input[1] == "safe_name_28",
        "candidate_int64_ramp_direct_equal": ci["safe_name_4"].dtype == np.dtype(np.int64) and cn["safe_name_29"].input[1] == "safe_name_27",
        "candidate_argmax_cast_removed": "safe_name_28" not in cn,
        "authority_cond_axis2_then_squeeze": an["cond1"].op_type == "ReduceMin" and "cond1s" in {o for n in authority.graph.node if n.op_type == "Squeeze" for o in n.output},
        "candidate_cond_reduce_all_scalar": cn["cond1s"].op_type == "ReduceMin" and len(cn["cond1s"].input) == 1 and helper.get_attribute_value(next(x for x in cn["cond1s"].attribute if x.name == "keepdims")) == 0,
        "candidate_bg_mask_transpose": cn["bg_mask_w"].op_type == "Transpose" and cn["bg_mask_w"].input[0] == "safe_name_29",
        "candidate_bg_where": cn["w_base2"].op_type == "Where" and list(cn["w_base2"].input) == ["bg_mask_w", "safe_name_13", "safe_name_14"],
        "terminal_second_scatter_unchanged": cn["w_u8_2"].SerializeToString() == an["w_u8_2"].SerializeToString(),
        "removed_ones_and_bg_zero": "weight_base_ones_u8" not in ci and "weight_bg_zero_u8" not in ci,
    }
    checks = {
        "argmax_equal_all_ten_indices": argmax_failures == 0,
        "condition_reduce_all_four_assignments": reduce_failures == 0,
        "terminal_background_all_ten_indices": terminal_failures == 0,
        "graph_contract": all(graph.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "argmax_equal": {"indices": 10, "failures": argmax_failures, "argument": "ArgMax is always in [0,9], so int64-to-uint8 Cast is lossless before equality."},
        "condition_reduce": {"assignments": 4, "failures": reduce_failures, "argument": "eq1 is fixed [1,1,2]; reducing axis 2 then squeezing equals reducing all axes directly."},
        "terminal_background": {"indices": 10, "failures": terminal_failures, "argument": "Transpose of the existing one-hot background mask followed by Where(0,1) equals scattering 0 into ten ones."},
        "graph_contract": graph,
        "authority_safe4_dtype": str(ai["safe_name_4"].dtype),
        "candidate_safe4_dtype": str(ci["safe_name_4"].dtype),
    }


def relation_trace(authority_data: bytes, candidate_data: bytes, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    a_names = ["safe_name_27", "safe_name_29", "cond1s", "w_base2", "w_u8_2", "output"]
    c_names = ["safe_name_27", "safe_name_29", "cond1s", "bg_mask_w", "w_base2", "w_u8_2", "output"]
    result: dict[str, Any] = {}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        aa = H.selected_trace_session(authority_data, level, a_names)
        cc = H.selected_trace_session(candidate_data, level, c_names)
        failures = {name: 0 for name in (
            "argmax", "background_mask", "condition", "background_mask_transpose",
            "base_weights", "final_weights", "output",
        )}
        errors = 0
        for row in rows:
            try:
                av = dict(zip(a_names, aa.run(a_names, {aa.get_inputs()[0].name: row["input"]}), strict=True))
                cv = dict(zip(c_names, cc.run(c_names, {cc.get_inputs()[0].name: row["input"]}), strict=True))
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            checks = {
                "argmax": np.array_equal(av["safe_name_27"], cv["safe_name_27"]),
                "background_mask": np.array_equal(av["safe_name_29"], cv["safe_name_29"]),
                "condition": np.array_equal(av["cond1s"], cv["cond1s"]),
                "background_mask_transpose": np.array_equal(cv["bg_mask_w"], np.transpose(cv["safe_name_29"], (1, 0, 2, 3))),
                "base_weights": np.array_equal(av["w_base2"], cv["w_base2"]),
                "final_weights": np.array_equal(av["w_u8_2"], cv["w_u8_2"]),
                "output": np.array_equal(av["output"], cv["output"]),
            }
            for label, passed in checks.items():
                failures[label] += int(not passed)
        result[mode] = {
            "cases": len(rows), "runtime_errors": errors,
            "failures": failures,
            "passed": errors == 0 and not any(failures.values()),
        }
    return result


def main() -> None:
    authority_data = AUTHORITY.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    hashes = {"authority": digest(authority_data), "candidate": digest(candidate_data)}
    hash_checks = {
        "authority": hashes["authority"] == AUTHORITY_SHA256,
        "candidate": hashes["candidate"] == CANDIDATE_SHA256,
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"hash drift: {hashes}")

    known = H.known_rows()
    fresh, generation = H.fresh_rows()
    structures = {
        "authority": H.model_structure(authority_data),
        "candidate": H.model_structure(candidate_data),
    }
    profiles = official_profiles(authority_data, candidate_data)
    outputs = H.compare_outputs(authority_data, candidate_data, known, fresh)
    trace_rows = known[:16] + fresh[:48]
    shapes = {
        "authority": H.trace_shapes(authority_data, trace_rows),
        "candidate": H.trace_shapes(candidate_data, trace_rows),
    }
    mismatch: dict[str, Any] = {}
    for mode in ("disable_all", "default"):
        aset = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["authority"][mode]["mismatches"]
        }
        cset = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["candidate"][mode]["mismatches"]
        }
        mismatch[mode] = {
            "authority_count": len(aset), "candidate_count": len(cset),
            "new": [list(x) for x in sorted(cset - aset)],
            "removed": [list(x) for x in sorted(aset - cset)],
            "identical_sets": aset == cset,
        }
    relations = relation_trace(authority_data, candidate_data, known[:16] + fresh[:496])
    formal = formal_checks(authority_data, candidate_data)

    output_pass = all(
        not row["session_errors"]
        and row["known"]["raw_equal"] == len(known)
        and row["known"]["raw_different"] == 0
        and row["fresh"]["raw_equal"] == len(fresh)
        and row["fresh"]["raw_different"] == 0
        and row["known"]["authority_errors"] == row["known"]["candidate_errors"] == 0
        and row["fresh"]["authority_errors"] == row["fresh"]["candidate_errors"] == 0
        and row["known"]["candidate_nonfinite"] == row["fresh"]["candidate_nonfinite"] == 0
        for row in outputs.values()
    )
    shape_pass = all(
        item["identical_sets"] and not item["new"]
        and shapes["authority"][mode]["runtime_errors"] == 0
        and shapes["candidate"][mode]["runtime_errors"] == 0
        and shapes["authority"][mode]["nonfinite_values"] == 0
        and shapes["candidate"][mode]["nonfinite_values"] == 0
        for mode, item in mismatch.items()
    )
    gates = {
        "hashes": all(hash_checks.values()),
        "authority_structure": structures["authority"]["passed"],
        "candidate_structure": structures["candidate"]["passed"],
        "official_cost_978_to_975": profiles["authority"]["cost"] == 978 and profiles["candidate"]["cost"] == 975,
        "candidate_known_correct": profiles["candidate"]["correct"] is True,
        "raw_equal_all_four_configs": output_pass,
        "same_26_inherited_mismatches": shape_pass,
        "rewrite_relations": all(x["passed"] for x in relations.values()),
        "formal_complete_support": formal["passed"],
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "task": 319, "decision": decision,
        "hashes": hashes, "hash_checks": hash_checks,
        "known_count": len(known), "fresh_count": len(fresh), "fresh_generation": generation,
        "structures": structures, "official_profiles": profiles,
        "output_equivalence": outputs, "shape_traces": shapes,
        "mismatch_comparison": mismatch, "relation_trace": relations,
        "formal_checks": formal, "gates": gates,
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": decision, "gates": gates, "profiles": profiles,
        "known": len(known), "fresh": len(fresh), "mismatch": mismatch,
        "relations": relations,
    }, indent=2))


if __name__ == "__main__":
    main()
