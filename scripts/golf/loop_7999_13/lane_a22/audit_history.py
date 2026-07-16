#!/usr/bin/env python3
"""Run the A22 strict audit by extending the reusable A20 auditor."""

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


BASE_COST = {101: 5712, 158: 7815}
shared.BASE_COST = BASE_COST
# These names are specific to known task101 public/private fixture branches.  Generic
# tensor names containing "patch" are deliberately not rejected.
TASK101_LOOKUP = re.compile(
    r"(?:^|_)(?:t101_\d+_patch|probe_idx|wide_probe|wide_origin|wide_safe)(?:_|$)",
    re.I,
)
UNSAFE_SOURCE = re.compile(r"quarantine|private0|processing.?error", re.I)


def augmented_audit(task: int, label: str, path: Path, static: int | None, sources: list[str], baseline: bool = False) -> dict[str, object]:
    row = shared.audit(task, label, path, static, sources, baseline=baseline)
    model = onnx.load(path)
    names = [x.name for x in model.graph.initializer] + [x.name for x in model.graph.node]
    hits = [name for name in names if task == 101 and TASK101_LOOKUP.search(name)]
    unsafe_sources = [source for source in sources if UNSAFE_SOURCE.search(source)]
    row["fixture_signature_lookup_names"] = hits
    row["unsafe_source_lineage"] = unsafe_sources
    reasons = set(row["pre_fresh_reasons"])
    if hits:
        reasons.add("fixture_signature_lookup")
    if unsafe_sources:
        reasons.add("unsafe_source_lineage")
    row["pre_fresh_reasons"] = sorted(reasons)
    row["pre_fresh_pass"] = not reasons
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows = []
    for task in (101, 158):
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
