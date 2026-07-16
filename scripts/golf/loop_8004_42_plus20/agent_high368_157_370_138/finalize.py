#!/usr/bin/env python3
"""Freeze lane 138 no-promotion manifests and evidence hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    screen = json.loads((HERE / "audit/screen_results.json").read_text())
    authority = sha256(ROOT / "submission.zip")
    if authority != AUTHORITY_SHA256:
        raise RuntimeError(f"authority ZIP drift: {authority}")
    if screen["fresh_required_count"] != 0:
        raise RuntimeError("unresolved fresh-required candidates remain")
    manifest = {
        "lane": screen["lane"],
        "authority": screen["authority"],
        "targets": [157, 368, 370],
        "candidate_count": screen["candidate_count"],
        "stage_counts": screen["stage_counts"],
        "winner_count": 0,
        "projected_gain": 0.0,
        "rows": screen["rows"],
    }
    (HERE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    winners = {
        "lane": screen["lane"],
        "authority_score": 8009.46,
        "authority_zip_sha256": authority,
        "winners": [],
        "winner_count": 0,
        "projected_gain": 0.0,
        "decision": "NO_PROMOTION",
        "fresh_not_run_reason": "zero candidate passed strict-lower + structure + known4 + authority raw equivalence + truthful shape gates",
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winners, indent=2) + "\n")
    evidence_paths = [
        HERE / "REPORT.md",
        HERE / "audit/build_manifest.json",
        HERE / "audit/screen_results.json",
        HERE / "audit/model_anatomy.json",
        HERE / "candidate_manifest.json",
        HERE / "winner_manifest.json",
    ]
    evidence = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in evidence_paths
        if path.exists()
    }
    (HERE / "evidence_sha256.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": manifest["candidate_count"],
                "stage_counts": manifest["stage_counts"],
                "winner_count": 0,
                "authority_zip_sha256": authority,
                "evidence_entries": len(evidence),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
