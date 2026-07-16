#!/usr/bin/env python3
"""Non-promoting absolute audit for the task275 cost-419 candidate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = 275
AUTHORITY_COST = 428
CANDIDATE = HERE / "task275_diagonal_reuse_cost419_c7ddaab77f6d.onnx"
EXPECTED_SHA256 = "c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2"
SEEDS = (275_419_001, 275_419_002)


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LANE = import_path(
    "task275_gold_audit_support",
    ROOT / "scripts/golf/cost351_500_gold_loop/worker.py",
)


def main() -> int:
    data = CANDIDATE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError((digest, EXPECTED_SHA256))

    official = LANE.official_gate(CANDIDATE, TASK, AUTHORITY_COST)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    fresh_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        cases, generation = LANE.BASE.SUPPORT.fresh_cases(TASK, seed, task_map)
        runtime = LANE.BASE.failfast_known(data, cases)
        passed = bool(
            runtime.get("early_reject_reason") is None
            and LANE.BASE.runtime_pass(runtime)
        )
        fresh_rows.append(
            {
                "seed": seed,
                "case_count": len(cases),
                "generation": generation,
                "runtime": runtime,
                "pass": passed,
            }
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "case_count": len(cases),
                    "pass": passed,
                    "runtime": runtime,
                }
            ),
            flush=True,
        )

    passed = bool(official["pass"] and all(row["pass"] for row in fresh_rows))
    payload = {
        "task": TASK,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest,
        "authority_cost": AUTHORITY_COST,
        "official_gate": official,
        "fresh": fresh_rows,
        "absolute_gate": (
            "official gold exact + strict checker/static shape + margin + score; "
            "two independent fresh-2000 streams each 100%"
        ),
        "root_authority_modified": False,
        "pass": passed,
    }
    (HERE / "final_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("absolute audit rejected candidate")

    gain = float(__import__("math").log(AUTHORITY_COST / official["candidate_cost"]))
    report = f"""# task275 gold-exact diagonal-reuse result

- Authority: cost {AUTHORITY_COST}
- Candidate: cost {official['candidate_cost']}
- Projected gain: +{gain:.12f}
- Candidate SHA-256: `{digest}`
- Official gold: exact pass
- Strict checker/static shape: pass
- Minimum positive raw margin: {official['minimum_positive']}
- Fresh: {fresh_rows[0]['case_count']}/{fresh_rows[0]['case_count']} seed {SEEDS[0]}
- Fresh: {fresh_rows[1]['case_count']}/{fresh_rows[1]['case_count']} seed {SEEDS[1]}
- Root submission/CSV/score pointers modified: no

The final Einsum reuses the same learned 3x3 color map in both former T/W
roles and reads its diagonal with repeated subscript `aa` as the W-row scale.
This preserves the required rank three while removing W's nine parameters.
The spatial router is byte-for-byte unchanged.

The earlier `task275_diag_color_cost413_351d0b2a8557.onnx` experiment is
rejected: it incorrectly collapsed the quotient and remainder distractor
colors and failed official gold at train[1].  It is not an admission.
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pass": True, "cost": official["candidate_cost"], "gain": gain}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
