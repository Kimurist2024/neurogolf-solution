#!/usr/bin/env python3
"""Apply SHA-specific LB history to the six task185 probe leads."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
KNOWN_BLACK = {
    "819bc2f00d96": "70207 third task185 cost185 was directly LB-black",
    "d3da20db8d73": "70203 task185_cost186_verified_lowcost lineage was the 70203 culprit",
    "d21f1db4d69b": "cost273 task185 network was LB-black in the later partner probe",
}
LOCAL_FALSE_ACCEPT = {
    "ce35307db278": "same lookup family; independent fresh seeds both 1/500 (0.2%)",
    "d675c9b4b81c": "same lookup family; independent fresh seeds both 1/500 (0.2%)",
    "e086e1c96c9f": "same lookup family; independent fresh seeds both 1/500 (0.2%)",
}


def main() -> int:
    preliminary = json.loads((HERE / "probe_manifest.json").read_text())
    classified = []
    for row in preliminary["candidates"]:
        prefix = row["sha256"][:12]
        if prefix in KNOWN_BLACK:
            classification = "KNOWN_LB_BLACK"
            reason = KNOWN_BLACK[prefix]
        elif prefix in LOCAL_FALSE_ACCEPT:
            classification = "REJECT_LOCAL_FALSE_ACCEPT"
            reason = LOCAL_FALSE_ACCEPT[prefix]
        else:
            classification = "LB_PROBE_REQUIRED"
            reason = "no SHA-specific LB disposition"
        classified.append(
            {
                **row,
                "classification": classification,
                "history_reason": reason,
                "probe_eligible": classification == "LB_PROBE_REQUIRED",
                "fixed_safe": False,
            }
        )
    audit = {
        "baseline_zip": preliminary["baseline_zip"],
        "baseline_zip_sha256": preliminary["baseline_zip_sha256"],
        "task": 185,
        "candidate_count": len(classified),
        "known_lb_black_count": sum(row["classification"] == "KNOWN_LB_BLACK" for row in classified),
        "local_false_accept_count": sum(
            row["classification"] == "REJECT_LOCAL_FALSE_ACCEPT" for row in classified
        ),
        "lb_probe_required_count": sum(row["classification"] == "LB_PROBE_REQUIRED" for row in classified),
        "rows": classified,
    }
    (HERE / "audit" / "lb_history_classification.json").write_text(json.dumps(audit, indent=2) + "\n")
    final_manifest = {
        "baseline_zip": preliminary["baseline_zip"],
        "baseline_zip_sha256": preliminary["baseline_zip_sha256"],
        "classification": "LB_PROBE_REQUIRED",
        "candidate_count": audit["lb_probe_required_count"],
        "candidates": [row for row in classified if row["probe_eligible"]],
        "excluded_evidence": "audit/lb_history_classification.json",
    }
    (HERE / "probe_manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
