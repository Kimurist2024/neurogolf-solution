#!/usr/bin/env python3
"""Try exact int64 optimizer barriers after the task233 Cast removal."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
BASE_CANDIDATE = HERE / "candidates/task233_integer_carrier.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

spec = importlib.util.spec_from_file_location("carrier242_audit", HERE / "audit.py")
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


VARIANTS = {
    "identity": lambda output: helper.make_node("Identity", ["nb_ci_i64"], [output]),
    "abs": lambda output: helper.make_node("Abs", ["nb_ci_i64"], [output]),
    "add_zero": lambda output: helper.make_node("Add", ["nb_ci_i64", "si0"], [output]),
    "mul_one": lambda output: helper.make_node("Mul", ["nb_ci_i64", "si1"], [output]),
    "div_one": lambda output: helper.make_node("Div", ["nb_ci_i64", "si1"], [output]),
    "max_zero": lambda output: helper.make_node("Max", ["nb_ci_i64", "si0"], [output]),
    "bitwise_or_zero": lambda output: helper.make_node(
        "BitwiseOr", ["nb_ci_i64", "si0"], [output]
    ),
    "bitwise_xor_zero": lambda output: helper.make_node(
        "BitwiseXor", ["nb_ci_i64", "si0"], [output]
    ),
    "reshape_same": lambda output: helper.make_node("Reshape", ["nb_ci_i64", "k5"], [output]),
    "expand_same": lambda output: helper.make_node("Expand", ["nb_ci_i64", "k5"], [output]),
    "transpose_rank1": lambda output: helper.make_node(
        "Transpose", ["nb_ci_i64"], [output], perm=[0]
    ),
    "clip_min_zero": lambda output: helper.make_node("Clip", ["nb_ci_i64", "si0"], [output]),
}


def profile(model: onnx.ModelProto, label: str) -> dict[str, int]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"carrier242_barrier_{label}_") as work:
        path = Path(work) / "task233.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def insert_barrier(model: onnx.ModelProto, label: str) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    output = f"ci_barrier_{label}"
    barrier = VARIANTS[label](output)
    nodes = []
    inserted = False
    rewired = 0
    target_outputs = {"qseq_code_f16", "selr5_vec", "selc5_vec"}
    for node in candidate.graph.node:
        nodes.append(node)
        if node.output and node.output[0] == "nb_ci_i64":
            nodes.append(barrier)
            inserted = True
        if node.op_type == "Gather" and set(node.output) & target_outputs:
            if node.input[1] != "nb_ci_i64":
                raise RuntimeError(f"unexpected Gather index for {list(node.output)}")
            node.input[1] = output
            rewired += 1
    if not inserted or rewired != 3:
        raise RuntimeError(f"barrier insertion failed: inserted={inserted} rewired={rewired}")
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    return candidate


def main() -> None:
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task233.onnx")
    authority = onnx.load_from_string(authority_bytes)
    base = onnx.load(BASE_CANDIDATE)
    known = audit.known_cases()
    authority_cost = profile(authority, "authority")
    rows = []
    survivors = []

    trials = [("none", base)]
    trials.extend((label, insert_barrier(base, label)) for label in VARIANTS)
    for label, candidate in trials:
        row = {
            "label": label,
            "exact_identity_proof": {
                "input_interval": [0, 6],
                "note": "ArgMax indices are nonnegative and <7; every tested op is identity on that interval.",
            },
            "known_four_config": [],
        }
        try:
            onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
            onnx.shape_inference.infer_shapes(
                copy.deepcopy(candidate), strict_mode=True, data_prop=True
            )
            row["full_check"] = True
            row["strict_shape_inference_data_prop"] = True
            row["profile"] = profile(candidate, label)
            row["strict_lower"] = row["profile"]["cost"] < authority_cost["cost"]
            row["memory_nonincrease"] = row["profile"]["memory"] <= authority_cost["memory"]
            if row["strict_lower"] and row["memory_nonincrease"]:
                for config in audit.CONFIGS:
                    result = audit.audit_cases(authority, candidate, known, config)
                    row["known_four_config"].append(result)
                    print(
                        label, result["config"], result["raw_equal_authority"], "/", result["total"],
                        flush=True,
                    )
            row["known_pass"] = (
                len(row["known_four_config"]) == len(audit.CONFIGS)
                and all(result["pass"] for result in row["known_four_config"])
            )
            if row["known_pass"]:
                path = HERE / "candidates" / f"task233_integer_carrier_{label}.onnx"
                onnx.save(candidate, path)
                row["path"] = str(path.relative_to(ROOT))
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                survivors.append(row)
        except Exception as exc:  # fail closed
            row["full_check"] = row.get("full_check", False)
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["known_pass"] = False
        rows.append(row)

    payload = {
        "task": 233,
        "authority_member_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "authority_profile": authority_cost,
        "known_cases": len(known),
        "barriers": list(VARIANTS),
        "rows": rows,
        "survivors": survivors,
    }
    (HERE / "barrier_scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "authority_profile": authority_cost,
                "rows": [
                    {
                        "label": row["label"],
                        "profile": row.get("profile"),
                        "known_pass": row.get("known_pass"),
                        "error": row.get("error"),
                    }
                    for row in rows
                ],
                "survivors": [row["label"] for row in survivors],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
