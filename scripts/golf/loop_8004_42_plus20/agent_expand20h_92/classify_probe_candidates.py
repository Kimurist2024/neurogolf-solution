#!/usr/bin/env python3
"""Apply exact-SHA LB history and two-seed fresh classifications to probes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


# These are exact historical nets, not task-level bans.  The paths below are
# retained as human-auditable provenance for each SHA decision.
KNOWN_LB_BLACK = {
    "6fffa0b87f151603b0def28aa6ef29a4b46c1334a8a21a54d27eea16741ff186": {
        "task": 219,
        "cost": 1081,
        "evidence": "scripts/golf/loop_7999_13/lane_b32/winner_manifest.json",
        "reason": "exact SHA is a historical private-zero/TfIdf LB-black net (also fresh 32/500 in history)",
    },
    "bbfa8f5b79d2e8345a39a41f327ac1c2c851f3c7f388dd595c72ef951e1b3050": {
        "task": 205,
        "cost": 937,
        "evidence": "others/2/7805/task205_rebuilt_top2_cost937.onnx + docs/golf/private_zero_tasks.md (7805 black set)",
        "reason": "exact SHA is task205 from the historical 7805 LB-black set",
    },
    "a750a7b84c4b502eaf2de33597c0f1e647deb4e6b6634bc515b2796d338645e7": {
        "task": 205,
        "cost": 1010,
        "evidence": "artifacts/quarantine/task205_7614rej_cost1010_private0_decoded.onnx",
        "reason": "exact SHA is retained in the canonical private-zero quarantine",
    },
    "14de50b94761f76c45bde1c1d0acafda18139836922acd7b9cf7f09fc2bae18c": {
        "task": 205,
        "cost": 1015,
        "evidence": "artifacts/quarantine/task205_70204_private0.onnx",
        "reason": "exact SHA is retained in the canonical private-zero quarantine",
    },
    "887d5695902ffc43583cda9340c564e51972f2850df3e69c5495093b5e06a576": {
        "task": 365,
        "cost": 1337,
        "evidence": "direct 705-pool LB report (black12 includes task365), exact SHA confirmed by parent coordinator",
        "reason": "exact task365@1337 SHA is a reported 705-pool LB-black net",
    },
}


# The 7804/7802 task396 results demonstrate why a task-level blacklist would
# be invalid.  Neither SHA appears among this lane's task396 candidates.
OTHER_EXACT_LB_HISTORY = [
    {
        "task": 396,
        "cost": 982,
        "sha256": "31d8fb23ed73f7dd244cb3fc02724d4568e04afb25ea2a4263abc274bff8db3a",
        "lb": "BLACK",
        "evidence": "others/2/7804/task396_improved.onnx + docs/golf/private_zero_tasks.md",
    },
    {
        "task": 396,
        "cost": 1026,
        "sha256": "7543068bda0d7750545d5c17b1cc12de66fc220b158de9e7252ed806d06f3b8b",
        "lb": "WHITE",
        "evidence": "others/2/7802/task396_improved.onnx + docs/golf/private_zero_tasks.md",
    },
]


# Only overwhelming fresh failures are false accepts.  The small 219 shaves
# are deliberately preserved for SHA-specific LB probes even though their
# generated distribution is imperfect.
SEVERE_FALSE_ACCEPT = {
    "8a5431580aa6e50b8b719294b67d5c508f9fea2e7b02a740731cd3fa5856a7ea",
    "192ab4e910ce562d387ec5ff1d11c35d5fa9d965d176e20a264d1dd712edc303",
}


def main() -> int:
    manifest_path = HERE / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fresh = json.loads((HERE / "audit/fresh_two_seed.json").read_text())
    fresh_by_sha = {row["sha256"]: row for row in fresh["per_sha"]}
    initial = list(manifest["candidates"])
    retained = []
    classifications = []
    for row in initial:
        digest = row["sha256"]
        fresh_row = fresh_by_sha[digest]
        item = {
            "task": int(row["task"]),
            "sha256": digest,
            "candidate_cost": int(row["candidate_cost"]),
            "fresh_two_seed": fresh_row,
        }
        if digest in KNOWN_LB_BLACK:
            item["classification"] = "KNOWN_LB_BLACK"
            item["lb_history"] = KNOWN_LB_BLACK[digest]
        elif digest in SEVERE_FALSE_ACCEPT:
            item["classification"] = "FALSE_ACCEPT"
            item["reason"] = "two independent fresh seeds fail overwhelmingly; not a credible rule net"
        else:
            item["classification"] = "LB_PROBE_REQUIRED"
            kept = dict(row)
            kept["fresh_two_seed"] = fresh_row
            if fresh_row["minimum_mode_rate"] < 0.90:
                kept["probe_priority"] = "LOW"
                kept["risk_flags"] = [
                    "fresh_below_90_but_retained_for_SHA_specific_LB_probe",
                    "not_an_adoption_without_LB_confirmation",
                ]
            retained.append(kept)
        classifications.append(item)

    manifest["pre_history_count"] = len(initial)
    manifest["classification_counts"] = dict(Counter(row["classification"] for row in classifications))
    manifest["known_lb_black_excluded"] = sum(row["classification"] == "KNOWN_LB_BLACK" for row in classifications)
    manifest["false_accept_excluded"] = sum(row["classification"] == "FALSE_ACCEPT" for row in classifications)
    manifest["count"] = len(retained)
    manifest["candidates"] = retained
    manifest["fresh_two_seed_evidence"] = str((HERE / "audit/fresh_two_seed.json").relative_to(ROOT))
    manifest["exact_sha_history_evidence"] = str((HERE / "audit/lb_history_exact_sha.json").relative_to(ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    history_report = {
        "rule": "exact SHA match only; no permanent task blacklist",
        "candidate_exact_lb_black": [
            {"sha256": digest, **metadata} for digest, metadata in KNOWN_LB_BLACK.items()
        ],
        "other_exact_task396_history_not_matching_current_candidates": OTHER_EXACT_LB_HISTORY,
    }
    (HERE / "audit/lb_history_exact_sha.json").write_text(json.dumps(history_report, indent=2) + "\n")
    (HERE / "audit/probe_classification.json").write_text(json.dumps({
        "initial_count": len(initial),
        "retained_probe_count": len(retained),
        "classification_counts": manifest["classification_counts"],
        "rows": classifications,
    }, indent=2) + "\n")
    print(json.dumps({
        "initial_count": len(initial),
        "retained_probe_count": len(retained),
        "classification_counts": manifest["classification_counts"],
        "retained_tasks": dict(Counter(row["task"] for row in retained)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
