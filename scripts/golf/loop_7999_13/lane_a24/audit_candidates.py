#!/usr/bin/env python3
"""Strict A24 audit for private-black/ambiguous task198 and task277."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


BASE_COST = {198: 661, 277: 731}
shared.BASE_COST = BASE_COST
UNSAFE_SOURCE = re.compile(r"quarantine|private0|processing.error|risky", re.I)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def forbidden_history_audit(
    task: int, label: str, path: Path, entry: dict[str, object]
) -> dict[str, object]:
    """Archive models cannot be promoted for these tasks, even after empirical passes."""
    model = onnx.load(path, load_external_data=False)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    giant_initializers = [
        item.name
        for item in model.graph.initializer
        if int(numpy_helper.to_array(item).size) >= 10_000
    ]
    sources = list(entry["sources"])
    reasons = ["archive_provenance_not_sound_or_exact_equivalent"]
    if max_einsum >= 15:
        reasons.append("giant_einsum")
    if ops.get("TfIdfVectorizer"):
        reasons.append("lookup_tfidf")
    if giant_initializers:
        reasons.append("giant_initializer")
    if any(UNSAFE_SOURCE.search(source) for source in sources):
        reasons.append("unsafe_source_lineage")
    try:
        onnx.checker.check_model(model, full_check=True)
        full_check: bool | str = True
    except Exception as exc:  # noqa: BLE001
        full_check = f"{type(exc).__name__}: {exc}"
        reasons.append("full_check")
    return {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "baseline_actual_cost": BASE_COST[task],
        "inventory_static_cost": int(entry["static_cost"]),
        "sources": sources,
        "nodes": len(model.graph.node),
        "params": shared.scoring.calculate_params(model),
        "value_info": len(model.graph.value_info),
        "declared_output_shapes": [dims(item) for item in model.graph.output],
        "ops": dict(ops),
        "max_einsum_inputs": max_einsum,
        "giant_initializers": giant_initializers,
        "full_check": full_check,
        "empirical_validation_skipped": (
            "terminal policy rejection: private-black/ambiguous archive provenance cannot "
            "establish generator-rule soundness or exact bitwise equivalence"
        ),
        "pre_fresh_pass": False,
        "pre_fresh_reasons": sorted(set(reasons)),
    }


def rule_reference_audit(
    task: int, label: str, path: Path, source: str
) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    if max_einsum >= 15:
        return {
            "task": task,
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "source": source,
            "nodes": len(model.graph.node),
            "params": shared.scoring.calculate_params(model),
            "max_einsum_inputs": max_einsum,
            "empirical_validation_skipped": "terminal forbidden giant Einsum",
            "pre_fresh_pass": False,
            "pre_fresh_reasons": ["giant_einsum", "known_skipped_after_terminal_screen"],
        }
    row = shared.audit(task, label, path, None, [source], baseline=False)
    # The behavioral 1256 reference is generator-valid empirically but retains
    # false intermediate declarations; generator soundness does not waive the
    # truthful-shape gate.
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows: list[dict[str, object]] = []

    for task in (198, 277):
        label = f"task{task:03d}_base"
        row = shared.audit(
            task,
            label,
            HERE / "baseline" / f"task{task:03d}.onnx",
            BASE_COST[task],
            ["submission_base_7999.13.zip"],
            baseline=True,
        )
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)

    for label, entry in manifest["history_entries"].items():
        task = int(entry["task"])
        row = forbidden_history_audit(task, label, HERE / "history" / f"{label}.onnx", entry)
        rows.append(row)
        print(label, row["pre_fresh_reasons"], flush=True)

    for label, entry in manifest["rule_references"].items():
        task = int(entry["task"])
        row = rule_reference_audit(
            task,
            label,
            HERE / "rule_references" / f"{label}.onnx",
            str(entry["source"]),
        )
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "audit_rows.json").write_text(
            json.dumps({"rows": rows, "complete": False}, indent=2) + "\n"
        )

    pending = [
        row
        for row in rows
        if row["pre_fresh_pass"] and not str(row["label"]).endswith("_base")
    ]
    (HERE / "audit_rows.json").write_text(
        json.dumps({"rows": rows, "pending": pending, "complete": True}, indent=2) + "\n"
    )
    (HERE / "fresh_dual_5000.json").write_text(
        json.dumps(
            {
                "required_for": "strictly-cheaper admissible pre-fresh finalists only",
                "pending_count": len(pending),
                "runs": [],
                "complete": True,
                "reason": "no candidate survived provenance, structure, truthful-shape, and cost gates",
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "external_validation_summary.json").write_text(
        json.dumps(
            {
                "required_for": "strictly-cheaper candidates passing dual fresh5000",
                "pending_count": 0,
                "runs": [],
                "errors": 0,
                "complete": True,
                "reason": "no dual-fresh finalist",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE rows={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
