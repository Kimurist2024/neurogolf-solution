#!/usr/bin/env python3
"""Run the A21 strict audit by extending the reusable A20 auditor."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


BASE_COST = {285: 8717, 286: 7561}
shared.BASE_COST = BASE_COST
LOOKUP_PATTERN = re.compile(
    r"corr|fixture|signature|prototype|lookup|target_rows|target_keep_bits|badrow",
    re.I,
)


def augmented_audit(task: int, label: str, path: Path, static: int | None, sources: list[str], baseline: bool = False) -> dict[str, object]:
    row = shared.audit(task, label, path, static, sources, baseline=baseline)
    model = onnx.load(path)
    hits = [x.name for x in model.graph.initializer if LOOKUP_PATTERN.search(x.name)]
    hits += [x.name for x in model.graph.node if LOOKUP_PATTERN.search(x.name)]
    row["fixture_signature_lookup_names"] = hits
    reasons = set(row["pre_fresh_reasons"])
    if hits:
        reasons.add("fixture_signature_lookup")
    row["pre_fresh_reasons"] = sorted(reasons)
    row["pre_fresh_pass"] = not reasons
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows = []
    for task in (285, 286):
        label = f"task{task:03d}_base"
        row = augmented_audit(task, label, HERE / "baseline" / f"task{task:03d}.onnx", BASE_COST[task], ["submission_base_7999.13.zip"], baseline=True)
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
    for label, item in manifest["history_entries"].items():
        task = int(item["task"])
        row = augmented_audit(task, label, HERE / "candidates" / f"{label}.onnx", int(item["static_cost"]), list(item["sources"]))
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    for label, item in manifest["sound_entries"].items():
        task = int(item["task"])
        row = augmented_audit(task, label, HERE / "sound" / f"{label}.onnx", None, [item["source"]])
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    pending = [x for x in rows if x["pre_fresh_pass"] and not x["label"].endswith("_base")]
    (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "pending": pending, "complete": True}, indent=2) + "\n")
    print(f"DONE rows={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
