#!/usr/bin/env python3
"""Re-audit the only historical frontiers below the 8009.46 task costs."""

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
FRONTIER = {
    12: ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task012_r01_static500.onnx",
    107: ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task107_r01_static638.onnx",
}
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


SCAN = load_module(
    "high160_frontier_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high160_frontier_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(task: int, model: onnx.ModelProto) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"high160_history_{task:03d}_", dir="/tmp") as wd:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, wd,
            label=f"high160_history_task{task:03d}", require_correct=False,
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    authority = json.loads((HERE / "authority_audit.json").read_text(encoding="utf-8"))
    rows = []
    for task, path in FRONTIER.items():
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        base = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        profile = official(task, model)
        structural = SCAN.structural(copy.deepcopy(model))
        trace = AUDIT.direct_trace(task, data)
        known = {
            label: AUDIT.known_config(task, base, data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        einsum_arities = [
            len([name for name in node.input if name])
            for node in model.graph.node if node.op_type == "Einsum"
        ]
        max_einsum = max(einsum_arities, default=0)
        base_cost = authority["tasks"][str(task)]["official_profile"]["cost"]
        reasons = []
        if profile["cost"] >= base_cost:
            reasons.append("not_strict_lower")
        if not profile["correct"]:
            reasons.append("official_known_incorrect")
        if not structural.get("pass", False):
            reasons.append("checker_or_strict_structure")
        if not trace.get("truthful", False):
            reasons.append("runtime_shape_not_truthful")
        if max_einsum > 8:
            reasons.append(f"giant_einsum:{max_einsum}")
        if not all(item.get("perfect", False) for item in known.values()):
            reasons.append("known_four_config_not_perfect")
        rows.append({
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "authority_member_sha256": authority["tasks"][str(task)]["sha256"],
            "authority_cost": base_cost,
            "official_profile": profile,
            "strict_lower": profile["cost"] < base_cost,
            "structural": structural,
            "runtime_shape_trace": trace,
            "known_four_configs_vs_authority": known,
            "max_einsum_inputs": max_einsum,
            "accepted": not reasons,
            "reasons": reasons,
        })
        print(
            f"task{task:03d} cost={profile['cost']} correct={profile['correct']} "
            f"truthful={trace.get('truthful')} max_einsum={max_einsum} "
            f"known4={all(item.get('perfect', False) for item in known.values())}",
            flush=True,
        )
    report = {
        "authority_sha256": AUTHORITY_SHA256,
        "rows": rows,
        "survivors": [row for row in rows if row["accepted"]],
    }
    (HERE / "history_frontier_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
