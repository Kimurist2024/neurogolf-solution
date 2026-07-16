#!/usr/bin/env python3
"""Fail-closed full-support audit for task226 Greater cost370."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline/task226.onnx"
CANDIDATE = HERE / "candidates/task226_greater_cost370.onnx"
EXPECTED_BASELINE = "852b6091385d97df6899e21304bf194440fb5cd3343385693093c24be0cb8203"
EXPECTED_CANDIDATE = "aebca4b2e7c3ce5cb5663a6f8b88e428e9bcf53b3e9f1161728c7dd9c502389f"
FRESH_SEEDS = (226_187_501, 226_187_777)
FRESH_COUNT = 5_000
MODES = (
    ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import structure  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session(model: onnx.ModelProto, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def domain() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[dict]]:
    generator = importlib.import_module("task_941d9a10")
    wides = [
        values
        for length in (3, 5)
        for values in itertools.product(range(1, 5), repeat=length)
        if sum(values) + length - 1 == 10
    ]
    talls = [
        values
        for length in (3, 5)
        for values in itertools.product(range(1, 4), repeat=length)
        if sum(values) + length - 1 == 10
    ]
    examples = [
        generator.generate(wides=list(wide), talls=list(tall))
        for wide in wides
        for tall in talls
    ]
    return wides, talls, examples


def evaluate(
    candidate_sessions: dict[str, ort.InferenceSession],
    examples: list[dict],
    authority_sessions: dict[str, ort.InferenceSession] | None = None,
) -> dict[str, dict]:
    rows = {
        name: {
            "right": 0,
            "wrong": 0,
            "errors": 0,
            "nonfinite": 0,
            "raw_equal_to_authority": 0,
            "first_failure": None,
        }
        for name in candidate_sessions
    }
    for index, example in enumerate(examples):
        bench = scoring.convert_to_numpy(example)
        assert bench is not None
        for name, candidate_session in candidate_sessions.items():
            row = rows[name]
            try:
                raw = candidate_session.run(["output"], {"input": bench["input"]})[0]
                row["nonfinite"] += int(np.count_nonzero(~np.isfinite(raw)))
                if np.array_equal(raw > 0, bench["output"] > 0):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {"index": index, "kind": "wrong"}
                if authority_sessions is not None:
                    authority_raw = authority_sessions[name].run(
                        ["output"], {"input": bench["input"]}
                    )[0]
                    row["raw_equal_to_authority"] += int(
                        np.array_equal(raw, authority_raw, equal_nan=True)
                    )
            except Exception as error:  # noqa: BLE001
                row["errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "index": index,
                        "kind": "runtime",
                        "error": f"{type(error).__name__}: {error}",
                    }
    for row in rows.values():
        row["total"] = row["right"] + row["wrong"] + row["errors"]
        if authority_sessions is None:
            row.pop("raw_equal_to_authority")
    return rows


def runtime_shapes(model: onnx.ModelProto, example: dict) -> list[dict]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    infos = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    names = [node.output[0] for node in inferred.graph.node]
    exposed = copy.deepcopy(inferred)
    del exposed.graph.output[:]
    exposed.graph.output.extend(copy.deepcopy(infos[name]) for name in names)
    onnx.checker.check_model(exposed, full_check=True)
    bench = scoring.convert_to_numpy(example)
    assert bench is not None
    results = []
    for name, level in MODES:
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 4
        probe = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        values = probe.run(names, {"input": bench["input"]})
        mismatches = []
        nonfinite = 0
        for tensor_name, value in zip(names, values):
            expected = tuple(
                int(dim.dim_value)
                for dim in infos[tensor_name].type.tensor_type.shape.dim
            )
            if tuple(value.shape) != expected:
                mismatches.append(
                    {
                        "name": tensor_name,
                        "declared": list(expected),
                        "runtime": list(value.shape),
                    }
                )
            nonfinite += int(np.count_nonzero(~np.isfinite(value)))
        results.append(
            {
                "mode": name,
                "outputs_exposed": len(names),
                "mismatches": mismatches,
                "nonfinite": nonfinite,
                "pass": not mismatches and nonfinite == 0,
            }
        )
    return results


def team_validator() -> dict:
    path = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"
    spec = importlib.util.spec_from_file_location("task226_regolf187_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit, failures = module.audit_model_bytes(
        CANDIDATE.read_bytes(),
        226,
        ROOT / "inputs/neurogolf-2026",
        source="task226_regolf187_greater370",
        trace_dir=HERE / "traces",
    )
    return {"audit": dataclasses.asdict(audit), "failures": failures}


def main() -> None:
    ort.set_default_logger_severity(4)
    assert sha256(BASELINE) == EXPECTED_BASELINE
    assert sha256(CANDIDATE) == EXPECTED_CANDIDATE
    authority = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    candidate_sessions = {name: session(candidate, level) for name, level in MODES}
    authority_sessions = {name: session(authority, level) for name, level in MODES}

    payload = scoring.load_examples(226)
    known_examples = payload["train"] + payload["test"] + payload["arc-gen"]
    known = evaluate(candidate_sessions, known_examples, authority_sessions)
    wides, talls, exhaustive_examples = domain()
    exhaustive = evaluate(candidate_sessions, exhaustive_examples, authority_sessions)

    generator = importlib.import_module("task_941d9a10")
    fresh = {}
    for seed in FRESH_SEEDS:
        random.seed(seed)
        examples = [generator.generate() for _ in range(FRESH_COUNT)]
        fresh[str(seed)] = evaluate(candidate_sessions, examples)

    witness = exhaustive_examples[0]
    shapes = runtime_shapes(candidate, witness)
    structural = structure(copy.deepcopy(candidate), 226)
    official = scoring.score_and_verify(
        copy.deepcopy(candidate),
        226,
        str(HERE / "traces"),
        label="task226_regolf187_greater370",
    )
    team = team_validator()
    cost = cost_of(str(CANDIDATE))
    truth_table = [
        {
            "a": a,
            "b": b,
            "old": bool(a) and not bool(b),
            "new": a > b,
        }
        for a, b in itertools.product((0.0, 1.0), repeat=2)
    ]

    all_known = all(
        row["right"] == len(known_examples)
        and row["errors"] == 0
        and row["nonfinite"] == 0
        and row["raw_equal_to_authority"] == len(known_examples)
        for row in known.values()
    )
    all_exhaustive = all(
        row["right"] == 136
        and row["errors"] == 0
        and row["nonfinite"] == 0
        and row["raw_equal_to_authority"] == 136
        for row in exhaustive.values()
    )
    all_fresh = all(
        row["right"] == FRESH_COUNT
        and row["errors"] == 0
        and row["nonfinite"] == 0
        for seed_rows in fresh.values()
        for row in seed_rows.values()
    )
    static_pass = bool(
        structural["checker_full"]
        and structural["strict_data_prop"]
        and structural["static_positive"]
        and structural["standard_domains"]
        and not structural["banned_ops"]
        and not structural["conv_bias_findings"]
        and not structural["giant_einsum"]
        and not structural["huge_fanin"]
        and not structural["lookup_or_scatter"]
        and not structural["errors"]
        and all(row["pass"] for row in shapes)
    )
    winner = bool(
        cost == (331, 39, 370)
        and official is not None
        and official["correct"]
        and all_known
        and all_exhaustive
        and all_fresh
        and static_pass
        and team["audit"]["valid"]
        and not team["failures"]
        and all(row["old"] == row["new"] for row in truth_table)
    )
    result = {
        "lane": "agent_task226_regolf_187",
        "task": 226,
        "authority": {
            "lb": 8009.46,
            "path": str(BASELINE.relative_to(ROOT)),
            "sha256": sha256(BASELINE),
            "cost": {"memory": 333, "params": 39, "cost": 372},
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
            "cost": {"memory": cost[0], "params": cost[1], "cost": cost[2]},
            "strict_lower": cost[2] < 372,
            "score_gain": math.log(372 / cost[2]),
        },
        "proof": {
            "generator": "inputs/arc-gen-repo/tasks/task_941d9a10.py",
            "valid_width_sequences": len(wides),
            "valid_height_sequences": len(talls),
            "complete_support_cases": len(exhaustive_examples),
            "complete_support_formula": "17 widths x 8 heights = 136",
            "rewrite_identity": "bool(a) AND NOT bool(b) == (a > b) for a,b in {0.0,1.0}",
            "truth_table": truth_table,
            "carrier_reason": "One-hot channel-0 GatherElements values are exactly 0.0 or 1.0.",
            "lookup_or_fixture_used": False,
        },
        "official_score_and_verify": official,
        "team_validator": team,
        "structure": structural,
        "runtime_shapes_four_modes": shapes,
        "known_four_modes": known,
        "exhaustive_136_four_modes": exhaustive,
        "fresh_2x5000_four_modes": fresh,
        "gates": {
            "static": static_pass,
            "known": all_known,
            "exhaustive_full_support": all_exhaustive,
            "fresh": all_fresh,
        },
        "winner_eligible": winner,
        "verdict": "SAFE_PRIVATE_ZERO" if winner else "REJECT",
        "protected_files_modified": False,
    }
    (HERE / "audit/final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "result.json").write_text(
        json.dumps(
            {
                "lane": result["lane"],
                "task": 226,
                "verdict": result["verdict"],
                "winner": result["candidate"] if winner else None,
                "authority_cost": 372,
                "winner_cost": cost[2] if winner else None,
                "score_gain": result["candidate"]["score_gain"] if winner else 0.0,
                "root_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "candidate": result["candidate"],
                "gates": result["gates"],
                "known": known,
                "exhaustive": exhaustive,
                "fresh": fresh,
                "runtime_shapes": shapes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
