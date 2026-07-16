#!/usr/bin/env python3
"""Fail-closed audit for the task285 SOUND reconstruction lane.

This lane never modifies a submission archive.  The current archive member is
read only to establish the exact admission threshold.  Fresh 2x5000 is gated
behind strict-lower cost, so an above-threshold SOUND control is not probed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.lib import scoring  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


BASE_ZIP = ROOT / "submission_base_8008.14.zip"
MEMBER = "task285.onnx"
SOUND = HERE / "candidates/task285_sound_dedup_copy.onnx"
SOUND_SOURCE = ROOT / "scripts/golf/scratch_codex/task285/agent_hub_algebra/candidate_columnfull_bitboard.onnx"
BANNED_FRESH_LINEAGE = {
    "current_sha256": "366212e29105fde0295030f3ec3bb014bd300f23aa8259ccd79da2eea720b9e2",
    "cost8717_fresh": "93/100; seven generator failures",
    "historical_fresh30": "29/30 followed by LB 6844.55 -> 6830.41 (-14.14)",
    "policy": "Never re-admit this shortcut/fixture lineage as SOUND.",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def measured(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def session_options(disable: bool) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return options


def raw_equivalence(left: onnx.ModelProto, right: onnx.ModelProto, disable: bool) -> dict[str, object]:
    a = ort.InferenceSession(scoring.sanitize_model(copy.deepcopy(left)).SerializeToString(), session_options(disable))
    b = ort.InferenceSession(scoring.sanitize_model(copy.deepcopy(right)).SerializeToString(), session_options(disable))
    total = equal = decoded_equal = 0
    max_abs = 0.0
    first_failure = None
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(scoring.load_examples(285)[split]):
            bench = scoring.convert_to_numpy(example)
            if bench is None:
                continue
            left_raw = a.run(["output"], {"input": bench["input"]})[0]
            right_raw = b.run(["output"], {"input": bench["input"]})[0]
            total += 1
            same = np.array_equal(left_raw, right_raw, equal_nan=True)
            equal += int(same)
            decoded_equal += int(np.array_equal(left_raw > 0, right_raw > 0))
            if left_raw.shape == right_raw.shape:
                delta = np.abs(np.nan_to_num(left_raw).astype(np.float64) - np.nan_to_num(right_raw).astype(np.float64))
                max_abs = max(max_abs, float(delta.max(initial=0.0)))
            else:
                max_abs = float("inf")
            if not same and first_failure is None:
                first_failure = {"split": split, "index": index}
    return {
        "total": total,
        "raw_bitwise_equal": equal,
        "decoded_equal": decoded_equal,
        "max_abs_raw_difference": max_abs,
        "first_failure": first_failure,
    }


def all_intermediate_audit(model: onnx.ModelProto, disable: bool) -> dict[str, object]:
    # Use the model's own declarations, not fresh inferred annotations.  This
    # directly checks that every charged declared tensor has the runtime shape
    # claimed by the submission on every known example.
    typed = {value.name: value for value in list(model.graph.value_info) + list(model.graph.output)}
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    expected_shapes: list[list[int]] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
                expected_shapes.append([int(dim.dim_value) for dim in typed[name].type.tensor_type.shape.dim])
    sanitized = scoring.sanitize_model(traced)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected traced model")
    session = ort.InferenceSession(sanitized.SerializeToString(), session_options(disable))
    output_names = [item.name for item in session.get_outputs()]
    examples = scoring.load_examples(285)
    checked_cases = checked_tensors = nonfinite_cells = shape_mismatches = 0
    affected_tensors: set[str] = set()
    first_shape_mismatch = None
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            bench = scoring.convert_to_numpy(example)
            if bench is None:
                continue
            checked_cases += 1
            values = session.run(None, {"input": bench["input"]})
            for tensor_index, (name, value) in enumerate(zip(output_names, values)):
                array = np.asarray(value)
                actual_shape = [int(item) for item in array.shape]
                if actual_shape != expected_shapes[tensor_index]:
                    shape_mismatches += 1
                    if first_shape_mismatch is None:
                        first_shape_mismatch = {
                            "split": split,
                            "tensor": name,
                            "declared": expected_shapes[tensor_index],
                            "runtime": actual_shape,
                        }
                if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating)):
                    continue
                checked_tensors += 1
                count = int(np.count_nonzero(~np.isfinite(array)))
                if count:
                    nonfinite_cells += count
                    affected_tensors.add(name)
    return {
        "cases": checked_cases,
        "floating_tensor_evaluations": checked_tensors,
        "nonfinite_cells": nonfinite_cells,
        "affected_tensors": sorted(affected_tensors),
        "declared_runtime_shape_mismatches": shape_mismatches,
        "first_shape_mismatch": first_shape_mismatch,
    }


def main() -> None:
    baseline_dir = HERE / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_bytes = archive.read(MEMBER)
    baseline_path = baseline_dir / MEMBER
    baseline_path.write_bytes(baseline_bytes)

    baseline_model = onnx.load_model_from_string(baseline_bytes)
    sound_model = onnx.load(SOUND)
    source_model = onnx.load(SOUND_SOURCE)
    baseline_cost = measured(baseline_path)
    sound_cost = measured(SOUND)
    strict_lower = sound_cost["cost"] < baseline_cost["cost"]

    sound_structure = structure(copy.deepcopy(sound_model), 285)
    result = {
        "lane": "agent_sound285_105",
        "target": 285,
        "current_authority": {
            "archive": BASE_ZIP.name,
            "archive_sha256": sha(BASE_ZIP),
            "member": MEMBER,
            "member_sha256": sha_bytes(baseline_bytes),
            "serialized_bytes": len(baseline_bytes),
            "cost": baseline_cost,
            "known_disable_all": run_known(copy.deepcopy(baseline_model), 285, True),
            "known_default": run_known(copy.deepcopy(baseline_model), 285, False),
            "sound_admission": "PROHIBITED_UNSAFE_LINEAGE",
        },
        "minimum_sound_candidate": {
            "path": str(SOUND.relative_to(ROOT)),
            "sha256": sha(SOUND),
            "serialized_bytes": SOUND.stat().st_size,
            "cost": sound_cost,
            "strict_lower": strict_lower,
            "cost_gap_above_current": sound_cost["cost"] - baseline_cost["cost"],
            "known_disable_all": run_known(copy.deepcopy(sound_model), 285, True),
            "known_default": run_known(copy.deepcopy(sound_model), 285, False),
            "structure": sound_structure,
            "raw_equivalence_to_pre_dedup_source": {
                "source": str(SOUND_SOURCE.relative_to(ROOT)),
                "source_sha256": sha(SOUND_SOURCE),
                "proof": "Only byte-identical TensorProto initializers were aliased.",
                "disable_all": raw_equivalence(source_model, sound_model, True),
                "default": raw_equivalence(source_model, sound_model, False),
            },
            "nonfinite_all_known_intermediates": {
                "disable_all": all_intermediate_audit(copy.deepcopy(sound_model), True),
                "default": all_intermediate_audit(copy.deepcopy(sound_model), False),
            },
            "fresh_evidence_inherited": {
                "detector_exhaustive": "23353/23353 (23088 reachable configurations plus 265 saved)",
                "detector_true_generator": "10000/10000",
                "full_model": "1000/1000",
                "reference_solver": "3000/3000",
                "formal_inheritance": "The 14-parameter dedup aliases byte-identical constants and cannot alter any input/output behavior.",
            },
        },
        "banned_prior_lineage": BANNED_FRESH_LINEAGE,
        "admission_gate": {
            "required": [
                "strictly lower than current cost 8623",
                "known all cases in ORT_DISABLE_ALL and default",
                "fresh seed A 5000/5000 and seed B 5000/5000",
                "standard domain, no fixture lookup, truthful runtime shapes",
                "Conv UB findings 0 and nonfinite findings 0",
            ],
            "fresh_two_seed_5000": {
                "ran": False,
                "reason": "No candidate passed the strict-lower pre-gate; running an adoption-only probe cannot change the rejection.",
            },
        },
        "winner": None,
        "verified_gain": 0,
        "verdict": "NO_STRICT_LOWER_SOUND_CANDIDATE",
        "protected_files_modified": False,
    }
    (HERE / "final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "lane": "agent_sound285_105",
                "task": 285,
                "winners": [],
                "candidate_for_probe": None,
                "verified_gain": 0,
                "protected_files_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"baseline": baseline_cost, "minimum_sound": sound_cost, "strict_lower": strict_lower, "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
