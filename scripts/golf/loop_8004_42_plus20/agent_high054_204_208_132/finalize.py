#!/usr/bin/env python3
"""Classify lane 132 probes and write immutable empty-winner manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
FALSE_LOWERS = {
    (54, "generic_noops"): "REJECT_RUNTIME_KNOWN4",
    (204, "fold_shape_batch1"): "REJECT_RUNTIME_KNOWN4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    probe = json.loads((HERE / "audit/probe_results.json").read_text())
    rows = []
    for original in probe["rows"]:
        row = dict(original)
        key = (int(row["task"]), str(row["kind"]))
        structure = row.get("structure") or {}
        measured = row.get("cost") or {}
        cost = measured.get("cost")
        if key in FALSE_LOWERS:
            stage = FALSE_LOWERS[key]
        elif structure.get("checker_full") is not True or structure.get(
            "strict_shape_data_prop"
        ) is not True:
            stage = "REJECT_CHECKER_OR_STRICT_SHAPE"
        elif row.get("cost_error") or cost is None or int(cost) < 0:
            stage = "REJECT_UNSCORABLE_OR_UNSUPPORTED_ORT_KERNEL"
        elif int(cost) >= int(row["authority_cost"]):
            stage = "REJECT_NOT_STRICTLY_LOWER"
        else:
            stage = "REJECT_UNAUDITED_LOWER_FAIL_CLOSED"
        row["stage"] = stage
        rows.append(row)
    stage_counts = dict(Counter(row["stage"] for row in rows))
    authority_after = sha256(ROOT / "submission.zip")
    if authority_after != AUTHORITY_SHA256:
        raise RuntimeError(f"authority ZIP drift: {authority_after}")
    manifest = {
        "lane": "agent_high054_204_208_132",
        "authority_score": 8009.46,
        "authority_zip": "submission.zip",
        "authority_zip_sha256_before": probe["authority_zip_sha256_before"],
        "authority_zip_sha256_after": authority_after,
        "targets": [54, 204, 208],
        "authority_costs": probe["base_costs"],
        "candidate_count": len(rows),
        "stage_counts": stage_counts,
        "winner_count": 0,
        "projected_gain": 0.0,
        "rows": rows,
    }
    (HERE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    winners = {
        "lane": manifest["lane"],
        "authority_score": 8009.46,
        "authority_zip_sha256": authority_after,
        "winners": [],
        "winner_count": 0,
        "projected_gain": 0.0,
        "decision": "NO_PROMOTION",
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winners, indent=2) + "\n")
    evidence_paths = [
        HERE / "REPORT.md",
        HERE / "audit/probe_results.json",
        HERE / "audit/memory_anatomy.json",
        HERE / "audit/domain_proofs.json",
        HERE / "audit/known4.json",
        HERE / "audit/fresh_two_seed.json",
        HERE / "audit/runtime_shape.json",
        HERE / "audit/structural.json",
        HERE / "candidate_manifest.json",
        HERE / "winner_manifest.json",
    ]
    evidence = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in evidence_paths
        if path.exists()
    }
    (HERE / "evidence_sha256.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": len(rows),
        "stage_counts": stage_counts,
        "winner_count": 0,
        "authority_zip_sha256": authority_after,
    }, indent=2))


if __name__ == "__main__":
    main()
