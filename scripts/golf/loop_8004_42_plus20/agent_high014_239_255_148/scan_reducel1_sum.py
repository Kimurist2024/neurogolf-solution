#!/usr/bin/env python3
"""Exhaust the exact nonnegative ReduceL1 -> ReduceSum rewrites for task014.

The three ReduceL1 inputs are nonnegative by construction:
  fg_h is a uint8-to-float16 cast,
  counts is a reduction of fg_h, and
  mask_h is a uint8-to-float16 CastLike.
Consequently abs(x) == x and the replacement is algebraically exact.  This
script still fail-closes on official cost and truthful runtime-shape gates.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
import tempfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "high148_reduce_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
SCAN = load_module(
    "high148_reduce_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(data: bytes) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="high148_reduce_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(onnx.load_model_from_string(data)), 14, workdir,
            label="high148_task014_reduce_sum", require_correct=False,
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def main() -> int:
    authority = ROOT / "submission.zip"
    if digest(authority.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    baseline = (HERE / "current/task014.onnx").read_bytes()
    if digest(baseline) != "15a7de7d7ad08eeb693bbe82b7527f34c15fb539b8a4b5a079daceac6c1fe7eb":
        raise RuntimeError("task014 baseline changed")
    base_model = onnx.load_model_from_string(baseline)
    reducer_indices = [i for i, node in enumerate(base_model.graph.node) if node.op_type == "ReduceL1"]
    if reducer_indices != [3, 4, 13]:
        raise RuntimeError(f"unexpected reducer layout: {reducer_indices}")

    rows = []
    for count in range(1, len(reducer_indices) + 1):
        for chosen in itertools.combinations(reducer_indices, count):
            model = copy.deepcopy(base_model)
            for index in chosen:
                model.graph.node[index].op_type = "ReduceSum"
            data = model.SerializeToString()
            profile = official(data)
            structural = SCAN.structural(copy.deepcopy(model))
            trace = AUDIT.direct_trace(14, data)
            reasons = []
            if profile["cost"] >= 360:
                reasons.append("official_cost_not_strict_lower")
            if not structural.get("pass", False):
                reasons.append("structural_gate")
            if not trace.get("truthful", False):
                reasons.append("runtime_shape_witness")
            if not reasons:
                reasons.append("known_and_fresh_required_not_run_fail_closed")
            rows.append({
                "replaced_node_indices": list(chosen),
                "sha256": digest(data),
                "official_profile": profile,
                "structural": structural,
                "runtime_shape_trace": trace,
                "algebraic_proof": "all rewritten inputs are nonnegative, so ReduceL1(x) == ReduceSum(x)",
                "accepted": False,
                "reasons": reasons,
            })

    report = {
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_sha256": digest(baseline),
        "baseline_official_cost": 360,
        "rewrite_family": "all 7 nonempty subsets of the 3 exact nonnegative ReduceL1->ReduceSum replacements",
        "rows": rows,
        "strict_lower_survivors": [],
    }
    (HERE / "reducel1_sum_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"nodes={row['replaced_node_indices']} cost={row['official_profile']['cost']} "
            f"truthful={row['runtime_shape_trace'].get('truthful')} "
            f"reasons={','.join(row['reasons'])}", flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
