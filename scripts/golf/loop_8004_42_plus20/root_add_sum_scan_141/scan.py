#!/usr/bin/env python3
"""Profile exact binary Add <-> Sum carrier substitutions on all400 tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank")
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int] | dict[str, str]:
    try:
        memory, params, cost = cost_of(str(path))
        return {"memory": int(memory), "params": int(params), "cost": int(cost)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def structure(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    try:
        findings = check_conv_bias(model)
        row["conv_bias_findings"] = findings
        row["conv_bias_ub0"] = not findings
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["pass"] = bool(row.get("full_check") and row.get("strict_data_prop") and row.get("conv_bias_ub0"))
    return row


def rewrite(model: onnx.ModelProto, source: str, target: str) -> tuple[onnx.ModelProto, int]:
    result = copy.deepcopy(model)
    count = 0
    for node in result.graph.node:
        if node.op_type == source and len(node.input) == 2 and not node.attribute:
            node.op_type = target
            count += 1
    return result, count


def main() -> int:
    if sha(AUTHORITY_ZIP) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    base_profiles: dict[int, dict[str, int] | dict[str, str]] = {}
    for base_path in sorted(AUTHORITY.glob("task*.onnx")):
        task = int(base_path.stem[4:])
        base = onnx.load(base_path)
        for source, target, label in (("Add", "Sum", "add_to_sum"), ("Sum", "Add", "sum_to_add")):
            candidate, count = rewrite(base, source, target)
            if not count:
                continue
            path = CANDIDATES / f"task{task:03d}_{label}.onnx"
            onnx.save(candidate, path)
            if task not in base_profiles:
                base_profiles[task] = profile(base_path)
            base_profile = base_profiles[task]
            candidate_profile = profile(path)
            candidate_structure = structure(candidate)
            strict_lower = bool(
                "cost" in base_profile
                and "cost" in candidate_profile
                and candidate_profile["cost"] < base_profile["cost"]
            )
            row = {
                "task": task,
                "label": label,
                "replaced_nodes": count,
                "authority_sha256": sha(base_path),
                "candidate_sha256": sha(path),
                "authority_profile": base_profile,
                "candidate_profile": candidate_profile,
                "candidate_structure": candidate_structure,
                "strict_lower": strict_lower,
                "runtime_audit": bool(strict_lower and candidate_structure["pass"]),
            }
            rows.append(row)
            if strict_lower:
                print(task, label, base_profile, "->", candidate_profile, "structure", candidate_structure["pass"])
    result = {
        "authority_sha256": AUTHORITY_SHA256,
        "identity": "binary Add and binary Sum have identical multidirectional broadcast addition semantics",
        "rows": rows,
        "runtime_audit": [
            {"task": row["task"], "label": row["label"]}
            for row in rows if row["runtime_audit"]
        ],
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"runtime audit shortlist={len(result['runtime_audit'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
