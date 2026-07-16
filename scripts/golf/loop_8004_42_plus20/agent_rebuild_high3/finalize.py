#!/usr/bin/env python3
"""Emit the bounded Wave4 result and evidence-oriented report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = {18: 4754, 233: 7432, 286: 7481, 366: 7987}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gain(task: int, cost: int) -> float:
    return math.log(BASE[task] / cost) if cost < BASE[task] else 0.0


def main() -> None:
    refs = json.loads((HERE / "reference_evidence.json").read_text())
    audit = json.loads((HERE / "attempt_audit.json").read_text())
    fresh18 = json.loads((HERE / "audits/task018_a03_privatezero_fresh32.json").read_text())
    fresh233 = json.loads((HERE / "audits/task233_a02_privatezero_fresh32.json").read_text())
    fresh286 = json.loads((HERE / "audits/task286_a01_fresh2000.json").read_text())

    rejected = [
        {
            "task": 18,
            "label": "task018_a03",
            "sha256": sha(HERE / "attempts/task018_a03.onnx"),
            "baseline_cost": BASE[18],
            "candidate_cost": 4682,
            "possible_gain": gain(18, 4682),
            "fresh": fresh18["aggregate"],
            "decision": "REJECT_PRIVATE_ZERO_TRUE_RULE_FAILURE",
            "reasons": ["24 TfIdfVectorizer lookup nodes", "fresh 0/32 in both ORT modes"],
        },
        {
            "task": 233,
            "label": "task233_a02",
            "sha256": sha(HERE / "attempts/task233_a02.onnx"),
            "baseline_cost": BASE[233],
            "candidate_cost": 4936,
            "possible_gain": gain(233, 4936),
            "fresh": fresh233["aggregate"],
            "decision": "REJECT_PRIVATE_ZERO_TRUE_RULE_FAILURE",
            "reasons": [
                "private-zero lookup lineage",
                "fresh 0/32 in both ORT modes (independent retained audit also 0/100)",
            ],
        },
        {
            "task": 286,
            "label": "task286_a01",
            "sha256": sha(HERE / "attempts/task286_a01.onnx"),
            "baseline_cost": BASE[286],
            "candidate_cost": 7122,
            "possible_gain": gain(286, 7122),
            "fresh": fresh286["aggregate"],
            "decision": "REJECT_POLICY90_AND_LOOKUP",
            "reasons": [
                "fresh 1727/2000 = 86.35% in both ORT modes",
                "public-fixture rcorr correction-table lineage",
            ],
        },
        {
            "task": 286,
            "label": "task286_a02",
            "sha256": sha(HERE / "attempts/task286_a02.onnx"),
            "baseline_cost": BASE[286],
            "candidate_cost": 7263,
            "possible_gain": gain(286, 7263),
            "fresh_retained": "4612/5000 = 92.24%",
            "decision": "REJECT_LOOKUP_LINEAGE",
            "reasons": ["public-fixture rcorr correction-table lineage; lookup remains prohibited"],
        },
        {
            "task": 366,
            "label": "task366_a01",
            "sha256": sha(HERE / "attempts/task366_a01.onnx"),
            "baseline_cost": BASE[366],
            "candidate_cost": 7646,
            "possible_gain": gain(366, 7646),
            "fresh_retained": "4685/4757 = 98.4864%",
            "decision": "REJECT_SHAPE_CLOAK",
            "reasons": [
                "107 declared/runtime shape mismatches and 8 undeclared intermediates",
                "metadata-only truthful repair costs 9465, above baseline 7987",
            ],
        },
        {
            "task": 366,
            "label": "task366_a04",
            "sha256": sha(HERE / "attempts/task366_a04.onnx"),
            "baseline_cost": BASE[366],
            "candidate_cost": 7839,
            "possible_gain": gain(366, 7839),
            "decision": "REJECT_SHAPE_CLOAK",
            "reasons": ["100 declared/runtime shape mismatches"],
        },
    ]
    winners: list[dict[str, object]] = []
    result = {
        "lane": "agent_rebuild_high3",
        "baseline_zip": "submission_base_8004.50.zip",
        "baseline_zip_sha256": sha(ROOT / "submission_base_8004.50.zip"),
        "tasks": [18, 233, 286, 366],
        "attempt_count": audit["attempt_count"],
        "policy_fresh_threshold": 0.90,
        "private_zero_threshold": 1.0,
        "winner_count": 0,
        "projected_gain": 0.0,
        "winners": winners,
        "rejected_shortlist": rejected,
        "reference_summary": {
            task: {
                "classification": refs["references"][task]["classification"],
                "stored": refs["references"][task]["stored"]["total"],
                "fresh": refs["references"][task]["fresh"],
            }
            for task in ("18", "233", "286", "366")
        },
        "root_or_zip_modified": False,
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winners, indent=2) + "\n")
    (HERE / "rejected_manifest.json").write_text(json.dumps(rejected, indent=2) + "\n")
    (HERE / "result_manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    report = f"""# Wave4 true-rule rebuild — tasks 018/233/286/366

## Outcome

No model is eligible for promotion. The exact `submission_base_8004.50.zip`
remains unchanged. **Projected gain: +0.000000**. This lane audited
{audit['attempt_count']} bounded attempts and did not build a ZIP.

## True-rule references

The generator sources and `common.py` were read before model work. Independent
multi-seed fresh checks produced:

| task | stored | fresh | classification |
|---:|---:|---:|---|
| 018 | 266/266 | 1976/2000 (98.8%) | canonical deterministic policy; generator is non-injective |
| 233 | 266/266 | 2000/2000 | exact NumPy reference |
| 286 | 265/265 | 2000/2000 | exact NumPy reference |
| 366 | 266/266 | 2000/2000 | exact NumPy reference |

Task018 cannot have a universally exact deterministic input-only solver: retained
legal generator calls have byte-identical inputs and different outputs. The
canonical spec-derived policy still clears the user's 90% threshold, but every
cheaper ONNX must independently clear the model gate.

## Decisive candidate results

| task | baseline -> candidate | dual-ORT fresh | decision |
|---:|---:|---:|---|
| 018 | 4754 -> 4682 | 0/32, 0 errors | reject: 24-node TfIdf/private-zero lookup, not true rule |
| 233 | 7432 -> 4936 | 0/32, 0 errors | reject: private-zero candidate fails the required fresh100 guarantee |
| 286 | 7481 -> 7122 | 1727/2000 = 86.35%, 0 errors | reject: below policy90 and rcorr lookup lineage |
| 286 | 7481 -> 7263 | retained 4612/5000 = 92.24% | reject: public-fixture correction lookup is prohibited |
| 366 | 7987 -> 7646 | retained 4685/4757 = 98.4864% | reject: 107 shape mismatches; truthful cost is 9465 |
| 366 | 7987 -> 7839 | known dual100 | reject: 100 shape mismatches |

The task018 full 2000 run was not continued after independent 0/32 in both ORT
modes, because private-zero admission requires 100%; one mismatch is already a
terminal rejection. The result is also consistent with the retained 0/5 audit.

## Sound floors reached

- task018's shape-clean, lookup-free known-dual rebuild costs **10857**, above 4754.
- task233's closest spec diagnostic costs **17007**, above 7432 (and still has nine
  nonfatal declaration mismatches); the cheap 4936 graph is fresh-zero.
- task286's correction-free full-row implementation costs **54552**, above 7481.
- task366's metadata-only truthful repair costs **9465**, above 7987. Every
  known-perfect sub-7987 graph in this pool relies on false declared shapes.

These establish the stop condition: additional local shaving cannot cross the
incumbents without reintroducing lookup tables, finite-depth policy failures, or
shape cloaking.

## Evidence

- `reference_evidence.json`: stored and 8-seed fresh NumPy reference results.
- `attempt_audit.json`: actual scorer costs, dual-ORT known results, checker,
  strict data propagation, banned/nested/domain checks, runtime shape traces,
  and Conv-family bias findings for all attempts.
- `audits/`: independent dual-ORT fresh results for tasks 018, 233, and 286.
- `winner_manifest.json`: empty; `rejected_manifest.json`: decisive rejections.

No root ZIP, CSV, score ledger, or baseline model was modified.
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"winner_count": 0, "projected_gain": 0.0, "attempt_count": audit["attempt_count"]}, indent=2))


if __name__ == "__main__":
    main()
