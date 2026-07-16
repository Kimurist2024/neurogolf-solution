#!/usr/bin/env python3
"""Deep official/shape/known audit for optimizer preliminary-lower payloads."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "high136_optimizer_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(task: int, data: bytes) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"high136_candidate_{task:03d}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(onnx.load_model_from_string(data)), task, workdir,
            label=f"high136_task{task:03d}_optimizer", require_correct=False,
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    sweep = json.loads((HERE / "optimizer_sweep.json").read_text())
    authority = json.loads((HERE / "authority_audit.json").read_text())
    rows = []
    for source in sweep["preliminary_lower"]:
        task = int(source["task"])
        path = ROOT / source["path"]
        data = path.read_bytes()
        baseline = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        profile = official(task, data)
        trace = AUDIT.direct_trace(task, data)
        known = {
            label: AUDIT.known_config(task, baseline, data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        baseline_cost = authority["tasks"][str(task)]["official_profile"]["cost"]
        reasons = []
        if profile["cost"] >= baseline_cost:
            reasons.append("official_cost_not_strict_lower")
        if not trace.get("truthful", False):
            reasons.append("runtime_shape_witness")
        if not all(item.get("perfect", False) for item in known.values()):
            reasons.append("known_or_authority_raw_mismatch")
        if not reasons:
            reasons.append("fresh_two_seed_dual_ORT_required_not_run_fail_closed")
        rows.append({
            "task": task, "label": source["label"], "path": source["path"],
            "sha256": digest(data), "baseline_sha256": digest(baseline),
            "declared_profile": source["declared_profile"],
            "official_profile": profile,
            "official_strict_lower": profile["cost"] < baseline_cost,
            "runtime_shape_trace": trace,
            "known_four_configs": known,
            "authority_raw_equal": all(item.get("perfect", False) for item in known.values()),
            "fresh": {"status": "not_run_before_mandatory_gates"},
            "accepted": False, "reasons": reasons,
        })
    report = {"authority_sha256": AUTHORITY_SHA256, "rows": rows, "winners": []}
    (HERE / "optimizer_candidate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"task{row['task']:03d} declared={row['declared_profile']['cost']} "
            f"official={row['official_profile']['cost']} truthful={row['runtime_shape_trace'].get('truthful')} "
            f"known4={row['authority_raw_equal']} reasons={','.join(row['reasons'])}", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
