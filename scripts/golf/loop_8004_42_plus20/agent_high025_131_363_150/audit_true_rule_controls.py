#!/usr/bin/env python3
"""Audit truthful generator-rule controls used only as lower-bound evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTROLS = (
    (
        25,
        ROOT / "scripts/golf/scratch_codex/task025/candidate_v22_conventional_spec.onnx",
        "preserve guide lines; erase sparse probes; project matching colors beside their guide; transpose-invariant",
        "prior report: spec reference 3000/3000 fresh; ONNX known 266/266",
    ),
    (
        131,
        ROOT / "artifacts/optimized/task131.onnx",
        "move the green creature across the red line and place cyan immediately beyond it; flip/transpose-invariant",
        "known 266/266; independent spec reference 3000/3000 fresh",
    ),
    (
        363,
        ROOT / "scripts/golf/scratch/task363/spec_dynamic.onnx",
        "restore the red exemplar to black, detect every legal translated sprite, and paint all detections red",
        "known 263/265 because two fixed fixtures are non-identifiable; legal fresh 3000/3000",
    ),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high150_control_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high150_control_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def main() -> int:
    inventory = json.loads((HERE / "audit/inventory_exact.json").read_text())
    rows = []
    for task, path, rule, prior_fresh in CONTROLS:
        data = path.read_bytes()
        base = (HERE / f"baseline/task{task:03d}.onnx").read_bytes()
        profile = SCAN.official_cost(data, f"high150_control_{task:03d}")
        static = SCAN.structural(copy.deepcopy(onnx.load_model_from_string(data)))
        trace = AUDIT.direct_trace(task, data)
        known = AUDIT.known_config(task, base, data, True, 1)
        rows.append(
            {
                "task": task,
                "source": str(path.relative_to(ROOT)),
                "rule": rule,
                "authority_cost": inventory["tasks"][str(task)]["official_profile"]["cost"],
                "control_profile": profile,
                "strict_lower": profile["cost"] < inventory["tasks"][str(task)]["official_profile"]["cost"],
                "static": static,
                "runtime_shape_trace": trace,
                "known_disable_all_threads1": {
                    "candidate_right": known.get("candidate_right"),
                    "total": known.get("total"),
                    "candidate_runtime_errors": known.get("runtime_errors", {}).get("candidate"),
                    "candidate_nonfinite": known.get("nonfinite_values", {}).get("candidate"),
                },
                "prior_generator_evidence": prior_fresh,
                "decision": "CONTROL_ONLY_NOT_LOWER",
            }
        )
        print(
            f"task{task:03d} control={profile['cost']} known="
            f"{known.get('candidate_right')}/{known.get('total')} truthful={trace.get('truthful')}",
            flush=True,
        )
    payload = {
        "rows": rows,
        "task363_non_identifiability": {
            "proof_source": "scripts/golf/scratch_codex/task363/spec_reference.py",
            "statement": (
                "The second fixed fixture admits an extra legal translation at (1,3), producing the same input "
                "and a different legal output. No deterministic input-only SOUND model can match both relations."
            ),
        },
    }
    (HERE / "audit/true_rule_controls.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
