#!/usr/bin/env python3
"""Audit exact CastLike type-reference removal shaves for high154.

Only initializers used exclusively as CastLike's second input are eligible.
Replacing CastLike(x, ref) by Cast(x, to=type(ref)) is an exact ONNX semantic
rewrite; official runtime cost and truthful shape remain mandatory gates.
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
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
BASE_COST = {367: 2179, 270: 587}
ELIGIBLE = {367: ("one8", "bfalse"), 270: ("i32like",)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "high154_castlike_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
SCAN = load_module(
    "high154_castlike_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(task: int, data: bytes) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"high154_castlike_{task:03d}_", dir="/tmp") as wd:
        result = scoring.score_and_verify(
            copy.deepcopy(onnx.load_model_from_string(data)), task, wd,
            label=f"high154_task{task:03d}_castlike", require_correct=False,
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def replace(model: onnx.ModelProto, names: tuple[str, ...]) -> dict[str, int]:
    init_map = {item.name: item for item in model.graph.initializer}
    counts = {name: 0 for name in names}
    for name in names:
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(numpy_helper.to_array(init_map[name]).dtype)
        for node in model.graph.node:
            if node.op_type == "CastLike" and len(node.input) >= 2 and node.input[1] == name:
                del node.input[1:]
                node.op_type = "Cast"
                del node.attribute[:]
                node.attribute.extend([onnx.helper.make_attribute("to", elem_type)])
                counts[name] += 1
    kept = [item for item in model.graph.initializer if item.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return counts


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    rows = []
    outdir = HERE / "castlike_probes"
    outdir.mkdir(parents=True, exist_ok=True)
    for task, eligible in ELIGIBLE.items():
        baseline = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        base_model = onnx.load_model_from_string(baseline)
        for width in range(1, len(eligible) + 1):
            for names in itertools.combinations(eligible, width):
                model = copy.deepcopy(base_model)
                counts = replace(model, names)
                data = model.SerializeToString()
                path = outdir / f"task{task:03d}_{'_'.join(names)}_{digest(data)[:12]}.onnx"
                path.write_bytes(data)
                try:
                    profile = official(task, data)
                except Exception as exc:  # noqa: BLE001
                    profile = {"error": f"{type(exc).__name__}: {exc}"}
                structural = SCAN.structural(copy.deepcopy(model))
                try:
                    trace = AUDIT.direct_trace(task, data)
                except Exception as exc:  # noqa: BLE001
                    trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
                lower = isinstance(profile.get("cost"), int) and profile["cost"] < BASE_COST[task]
                reasons = []
                if not lower:
                    reasons.append("official_cost_not_strict_lower_or_unscorable")
                if not structural.get("pass", False):
                    reasons.append("structural_gate")
                if not trace.get("truthful", False):
                    reasons.append("runtime_shape_witness")
                if not reasons:
                    reasons.append("known_dual_ORT_required_not_run_fail_closed")
                rows.append({
                    "task": task,
                    "removed_type_references": list(names),
                    "rewritten_castlike_nodes": counts,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(data),
                    "official_profile": profile,
                    "official_strict_lower": lower,
                    "structural": structural,
                    "runtime_shape_trace": trace,
                    "accepted": False,
                    "reasons": reasons,
                })
    report = {
        "authority_sha256": AUTHORITY_SHA256,
        "rewrite_proof": "CastLike(x, ref) == Cast(x, to=dtype(ref))",
        "rows": rows,
        "strict_lower_truthful_survivors": [],
    }
    (HERE / "castlike_shave_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"task{row['task']:03d} refs={row['removed_type_references']} "
            f"official={row['official_profile'].get('cost')} "
            f"truthful={row['runtime_shape_trace'].get('truthful')} "
            f"reasons={','.join(row['reasons'])}", flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
