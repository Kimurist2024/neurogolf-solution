#!/usr/bin/env python3
"""Deep exact-equivalence audit for the task205 rowpow Selu memshave."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
BASE = HERE / "current/task205.onnx"
CANDIDATE = HERE / "candidates/task205_rowpow_selu.onnx"
FRESH_SEEDS = (12320501, 12320502)
FRESH_PER_SEED = 5000
ARBITRARY_SEED = 12320599
ARBITRARY_CASES = 2000
MODES = ((True, "disable_all"), (False, "default"))

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "high123_deep_audit_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
SCAN = load_module(
    "high123_deep_scan_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
CONV = load_module("high123_deep_conv", ROOT / "scripts/golf/check_conv_bias.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(session, array: np.ndarray) -> np.ndarray:
    return np.asarray(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: array})[0])


def make_pair(base: bytes, candidate: bytes, disable: bool):
    return (
        AUDIT.make_session(base, disable, 1),
        AUDIT.make_session(candidate, disable, 1),
    )


def empty_mode() -> dict[str, Any]:
    return {
        "total": 0,
        "candidate_right": 0,
        "authority_right": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "runtime_errors": {"candidate": 0, "authority": 0},
        "nonfinite_values": {"candidate": 0, "authority": 0},
        "candidate_shapes": [],
        "authority_shapes": [],
        "minimum_positive_candidate": None,
        "candidate_values_between_0_and_025": 0,
        "first_failure": None,
    }


def update_mode(row: dict[str, Any], expected: np.ndarray, base_session, candidate_session, benchmark: np.ndarray, case: int) -> None:
    row["total"] += 1
    outputs: dict[str, np.ndarray] = {}
    for name, session in (("authority", base_session), ("candidate", candidate_session)):
        try:
            value = run(session, benchmark)
        except Exception as exc:  # noqa: BLE001
            row["runtime_errors"][name] += 1
            row["first_failure"] = row["first_failure"] or {
                "case": case, "model": name, "error": f"{type(exc).__name__}: {exc}"
            }
            continue
        outputs[name] = value
        shape_key = f"{name}_shapes"
        shape = list(value.shape)
        if shape not in row[shape_key]:
            row[shape_key].append(shape)
        nonfinite = int(value.size - np.count_nonzero(np.isfinite(value)))
        row["nonfinite_values"][name] += nonfinite
        row[f"{name}_right"] += int(np.array_equal(value > 0, expected))
        if name == "candidate":
            positive = value[value > 0]
            if positive.size:
                minimum = float(positive.min())
                old = row["minimum_positive_candidate"]
                row["minimum_positive_candidate"] = minimum if old is None else min(old, minimum)
            row["candidate_values_between_0_and_025"] += int(np.count_nonzero((value > 0) & (value < 0.25)))
    if len(outputs) == 2:
        raw_equal = np.array_equal(outputs["candidate"], outputs["authority"])
        threshold_equal = np.array_equal(outputs["candidate"] > 0, outputs["authority"] > 0)
        row["raw_equal"] += int(raw_equal)
        row["threshold_equal"] += int(threshold_equal)
        if not (raw_equal and threshold_equal):
            row["first_failure"] = row["first_failure"] or {
                "case": case,
                "comparison": "candidate_vs_authority",
                "raw_equal": bool(raw_equal),
                "threshold_equal": bool(threshold_equal),
            }


def finalize_mode(row: dict[str, Any]) -> None:
    total = row["total"]
    row["candidate_rate"] = row["candidate_right"] / total if total else None
    row["authority_rate"] = row["authority_right"] / total if total else None
    row["raw_equivalence_rate"] = row["raw_equal"] / total if total else None
    row["threshold_equivalence_rate"] = row["threshold_equal"] / total if total else None
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())


def fresh_audit(base: bytes, candidate: bytes) -> dict[str, Any]:
    module = importlib.import_module("task_8731374e")
    report: dict[str, Any] = {"generator": "inputs/arc-gen-repo/tasks/task_8731374e.py", "runs": []}
    for seed in FRESH_SEEDS:
        random.seed(seed)
        sessions = {label: make_pair(base, candidate, disable) for disable, label in MODES}
        modes = {label: empty_mode() for _, label in MODES}
        generation_errors = conversion_skips = 0
        valid = 0
        for index in range(FRESH_PER_SEED):
            try:
                example = module.generate()
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                conversion_skips += 1
                continue
            expected = benchmark["output"].astype(bool)
            for _, label in MODES:
                base_session, candidate_session = sessions[label]
                update_mode(modes[label], expected, base_session, candidate_session, benchmark["input"], valid)
            valid += 1
        for row in modes.values():
            finalize_mode(row)
        report["runs"].append({
            "seed": seed,
            "requested": FRESH_PER_SEED,
            "valid": valid,
            "generation_errors": generation_errors,
            "conversion_skips": conversion_skips,
            "modes": modes,
        })
        print(f"fresh seed={seed} valid={valid}", flush=True)
    return report


def arbitrary_onehot_audit(base: bytes, candidate: bytes) -> dict[str, Any]:
    rng = np.random.default_rng(ARBITRARY_SEED)
    sessions = {label: make_pair(base, candidate, disable) for disable, label in MODES}
    modes = {
        label: {
            "total": 0,
            "raw_equal": 0,
            "threshold_equal": 0,
            "runtime_errors": {"candidate": 0, "authority": 0},
            "nonfinite_values": {"candidate": 0, "authority": 0},
            "first_failure": None,
        }
        for _, label in MODES
    }
    for index in range(ARBITRARY_CASES):
        height = int(rng.integers(1, 31))
        width = int(rng.integers(1, 31))
        colors = rng.integers(0, 10, size=(height, width))
        benchmark = np.zeros((1, 10, 30, 30), dtype=np.float32)
        rows, cols = np.indices((height, width))
        benchmark[0, colors, rows, cols] = 1.0
        for _, label in MODES:
            row = modes[label]
            row["total"] += 1
            outputs: dict[str, np.ndarray] = {}
            for name, session in (("authority", sessions[label][0]), ("candidate", sessions[label][1])):
                try:
                    value = run(session, benchmark)
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"][name] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "case": index, "model": name, "error": f"{type(exc).__name__}: {exc}"
                    }
                    continue
                outputs[name] = value
                row["nonfinite_values"][name] += int(value.size - np.count_nonzero(np.isfinite(value)))
            if len(outputs) == 2:
                raw_equal = np.array_equal(outputs["candidate"], outputs["authority"])
                threshold_equal = np.array_equal(outputs["candidate"] > 0, outputs["authority"] > 0)
                row["raw_equal"] += int(raw_equal)
                row["threshold_equal"] += int(threshold_equal)
                if not (raw_equal and threshold_equal):
                    row["first_failure"] = row["first_failure"] or {
                        "case": index, "raw_equal": bool(raw_equal), "threshold_equal": bool(threshold_equal)
                    }
    return {
        "seed": ARBITRARY_SEED,
        "cases": ARBITRARY_CASES,
        "domain": "finite nonnegative one-hot rectangles, random H/W 1..30, zero-hot outside",
        "modes": modes,
    }


def replacement_proof(base: bytes, candidate: bytes) -> dict[str, Any]:
    old = onnx.load_model_from_string(base)
    new = onnx.load_model_from_string(candidate)
    old_init = {item.name: item for item in old.graph.initializer}
    new_init = {item.name: item for item in new.graph.initializer}
    removed = sorted(set(old_init) - set(new_init))
    added = sorted(set(new_init) - set(old_init))
    common_equal = all(old_init[name].SerializeToString() == new_init[name].SerializeToString() for name in set(old_init) & set(new_init))
    diffs = []
    for index, (left, right) in enumerate(zip(old.graph.node, new.graph.node)):
        if left.SerializeToString() != right.SerializeToString():
            diffs.append({
                "index": index,
                "old": {"op": left.op_type, "inputs": list(left.input), "outputs": list(left.output)},
                "new": {
                    "op": right.op_type,
                    "inputs": list(right.input),
                    "outputs": list(right.output),
                    "attributes": {attr.name: helper.get_attribute_value(attr) for attr in right.attribute},
                },
            })
    gamma = float(np.asarray(numpy_helper.to_array(old_init["rowpow_thr"])))
    expected = {
        15: ("tall_f", "colq_scale"),
        19: ("roww_max", "roww_thr"),
    }
    replacements_ok = len(diffs) == 2
    for item in diffs:
        index = item["index"]
        replacements_ok &= index in expected
        if index in expected:
            source, output = expected[index]
            attrs = item["new"]["attributes"]
            replacements_ok &= (
                item["old"]["op"] == "Mul"
                and item["old"]["inputs"] == [source, "rowpow_thr"]
                and item["new"]["op"] == "Selu"
                and item["new"]["inputs"] == [source]
                and item["new"]["outputs"] == [output]
                and float(attrs["alpha"]) == 1.0
                and float(attrs["gamma"]) == gamma
            )
    return {
        "removed_initializers": removed,
        "added_initializers": added,
        "all_common_initializers_byte_equal": common_equal,
        "node_count_equal": len(old.graph.node) == len(new.graph.node),
        "node_differences": diffs,
        "gamma": gamma,
        "nonnegative_proof": {
            "tall_f": "ReduceSum(Cast(Greater(...))) is a sum of 0/1 values, hence >= 0.",
            "roww_max": "ReduceMax(Einsum(nonnegative one-hot input, Hardmax 0/1, nonnegative counts)) is >= 0.",
            "identity": "For x>0, Selu(alpha=1,gamma=g)=g*x; for x=0 it is g*(exp(0)-1)=0.",
            "valid_input_domain": "NeuroGolf inputs are finite 0/1 one-hot tensors, so both sources are always finite and nonnegative.",
        },
        "pass": bool(
            removed == ["rowpow_thr"]
            and not added
            and common_equal
            and len(old.graph.node) == len(new.graph.node)
            and replacements_ok
        ),
    }


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    base = BASE.read_bytes()
    candidate = CANDIDATE.read_bytes()
    proof = replacement_proof(base, candidate)
    static = SCAN.structural(onnx.load_model_from_string(candidate))
    trace = AUDIT.direct_trace(205, candidate)
    conv = CONV.check_model(onnx.load_model_from_string(candidate))
    current_profile = SCAN.official_cost(base, "high123_deep_current")
    candidate_profile = SCAN.official_cost(candidate, "high123_deep_candidate")
    screen = json.loads((HERE / "screen.json").read_text(encoding="utf-8"))
    screen_row = next(row for row in screen["entries"] if row["label"] == "rowpow_selu")
    fresh = fresh_audit(base, candidate)
    arbitrary = arbitrary_onehot_audit(base, candidate)

    known_perfect = all(row.get("perfect", False) for row in screen_row["known_four_configs"].values())
    fresh_equivalent = all(
        mode["raw_equal"] == mode["total"]
        and mode["threshold_equal"] == mode["total"]
        and mode["runtime_errors_total"] == 0
        and mode["nonfinite_values_total"] == 0
        for run_row in fresh["runs"]
        for mode in run_row["modes"].values()
    )
    arbitrary_equivalent = all(
        mode["raw_equal"] == mode["total"]
        and mode["threshold_equal"] == mode["total"]
        and sum(mode["runtime_errors"].values()) == 0
        and sum(mode["nonfinite_values"].values()) == 0
        for mode in arbitrary["modes"].values()
    )
    reasons = []
    if candidate_profile["cost"] >= current_profile["cost"]:
        reasons.append("not_strict_lower")
    if not static.get("pass", False):
        reasons.append("static_gate")
    if not trace.get("truthful", False):
        reasons.append("runtime_shape_truth")
    if conv:
        reasons.append("conv_bias_ub")
    if not known_perfect:
        reasons.append("known_four_configs")
    if not proof["pass"]:
        reasons.append("algebraic_replacement_proof")
    if not fresh_equivalent:
        reasons.append("fresh_raw_equivalence")
    if not arbitrary_equivalent:
        reasons.append("arbitrary_onehot_raw_equivalence")
    accepted = not reasons
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "task": 205,
        "current": {"path": str(BASE.relative_to(ROOT)), "sha256": digest(base), "profile": current_profile},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": digest(candidate), "profile": candidate_profile},
        "cost_reduction": current_profile["cost"] - candidate_profile["cost"],
        "score_gain": float(np.log(current_profile["cost"] / candidate_profile["cost"])),
        "static": static,
        "runtime_shape_trace": trace,
        "conv_bias_findings": [list(item) for item in conv],
        "known_four_configs": screen_row["known_four_configs"],
        "replacement_proof": proof,
        "fresh": fresh,
        "arbitrary_onehot_differential": arbitrary,
        "fresh_equivalent": fresh_equivalent,
        "arbitrary_equivalent": arbitrary_equivalent,
        "fresh_gold_is_not_admission_basis": True,
        "admission_basis": "all-valid-input algebraic equivalence to immutable authority",
        "reasons": reasons,
        "accepted": accepted,
        "verdict": "ACCEPT_EXACT_MEMSHAVE" if accepted else "REJECT",
    }
    (HERE / "winner_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": accepted,
        "profiles": {"current": current_profile, "candidate": candidate_profile},
        "proof": proof["pass"],
        "known_perfect": known_perfect,
        "fresh_equivalent": fresh_equivalent,
        "arbitrary_equivalent": arbitrary_equivalent,
        "reasons": reasons,
    }, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
