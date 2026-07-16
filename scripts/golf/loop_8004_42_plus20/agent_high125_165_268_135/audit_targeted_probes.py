#!/usr/bin/env python3
"""Build and fail-closed audit the two exact strict-lower targeted probes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import helper


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


SCAN = load_module(
    "high135_target_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high135_target_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official_profile(task: int, data: bytes, label: str) -> dict[str, object]:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"high135_probe_{task:03d}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError(f"task{task:03d}: score_and_verify returned None")
    return result


def build_task268_cast() -> tuple[bytes, dict[str, object]]:
    model = onnx.load(HERE / "current/task268.onnx")
    matches = [node for node in model.graph.node if node.name == "CastLike_final_bool"]
    if len(matches) != 1:
        raise RuntimeError(f"task268 final CastLike count={len(matches)}")
    node = matches[0]
    if node.op_type != "CastLike" or list(node.input) != ["_csp_9_30", "_bool_like"]:
        raise RuntimeError("task268 final CastLike signature changed")
    node.op_type = "Cast"
    del node.input[:]
    node.input.append("_csp_9_30")
    del node.attribute[:]
    node.attribute.append(helper.make_attribute("to", onnx.TensorProto.BOOL))
    before = len(model.graph.initializer)
    kept = [value for value in model.graph.initializer if value.name != "_bool_like"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if len(kept) != before - 1:
        raise RuntimeError("task268 _bool_like initializer not uniquely removed")
    proof = {
        "kind": "castlike_to_cast_and_remove_type_initializer",
        "all_valid_inputs": "ONNX CastLike(x, bool_scalar) delegates to Cast(x,to=BOOL); _bool_like has no other consumer",
        "floating_associativity": "none",
        "rounding": "identical Cast conversion",
        "overflow": "not applicable for BOOL cast",
    }
    return model.SerializeToString(), proof


def audit_one(task: int, label: str, candidate: bytes, proof: dict[str, object]) -> dict[str, object]:
    baseline = (HERE / f"current/task{task:03d}.onnx").read_bytes()
    outdir = HERE / "rejected_probes"
    outdir.mkdir(parents=True, exist_ok=True)
    sha = digest(candidate)
    path = outdir / f"task{task:03d}_{label}_{sha[:12]}.onnx"
    path.write_bytes(candidate)
    baseline_profile = json.loads((HERE / "authority_audit.json").read_text())["tasks"][str(task)]["official_profile"]
    profile = official_profile(task, candidate, f"high135_task{task:03d}_{label}")
    structural = SCAN.structural(copy.deepcopy(onnx.load_model_from_string(candidate)))
    trace = AUDIT.direct_trace(task, candidate)
    known = {
        config_label: AUDIT.known_config(task, baseline, candidate, disable, threads)
        for disable, threads, config_label in CONFIGS
    }
    reasons = []
    if profile["cost"] >= baseline_profile["cost"]:
        reasons.append("not_strict_lower")
    if not structural.get("pass", False):
        reasons.append("full_or_strict_structural_gate_failed")
    if not trace.get("truthful", False):
        reasons.append("runtime_shape_witness")
    if not all(item.get("perfect", False) for item in known.values()):
        reasons.append("known4_or_runtime_error")
    if not reasons:
        reasons.append("fresh_two_seed_dual_ORT_required_not_run_fail_closed")
    return {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha,
        "baseline_sha256": digest(baseline),
        "proof": proof,
        "baseline_official_profile": baseline_profile,
        "official_profile": profile,
        "strict_lower": profile["cost"] < baseline_profile["cost"],
        "structural": structural,
        "runtime_shape_trace": trace,
        "known_four_configs": known,
        "fresh": {"status": "not_run_before_mandatory_gates"},
        "accepted": False,
        "reasons": reasons,
    }


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    exact = json.loads((HERE / "exact_scan.json").read_text(encoding="utf-8"))
    task165_rows = [row for row in exact["preliminary_lower"] if row["task"] == 165 and row["kind"] == "cse"]
    if len(task165_rows) != 1:
        raise RuntimeError(f"task165 CSE row count={len(task165_rows)}")
    cse_path = ROOT / task165_rows[0]["path"]
    task165_data = cse_path.read_bytes()
    task165_proof = {
        "kind": "identical_pure_node_CSE",
        "all_valid_inputs": "two CastLike nodes have byte-identical op/domain/inputs/attributes; output replacement is algebraically exact",
        "runtime_caveat": "allocator behavior can differ after removing the duplicate node and is independently gated",
    }
    task268_data, task268_proof = build_task268_cast()
    rows = [
        audit_one(165, "duplicate_castlike_cse", task165_data, task165_proof),
        audit_one(268, "final_bool_cast_attribute", task268_data, task268_proof),
    ]
    report = {
        "authority_sha256": AUTHORITY_SHA256,
        "rows": rows,
        "winners": [],
        "verdict": "NO_SAFE_TARGETED_CANDIDATE",
    }
    (HERE / "targeted_probe_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"task{row['task']:03d} {row['label']} cost={row['official_profile']['cost']} "
            f"truthful={row['runtime_shape_trace'].get('truthful')} "
            f"known4={all(v.get('perfect', False) for v in row['known_four_configs'].values())} "
            f"reasons={','.join(row['reasons'])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
