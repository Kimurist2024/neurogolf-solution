#!/usr/bin/env python3
"""Known-set gate for the changed-task broadcast candidates.

This script is deliberately read-only outside ``agent_changed_resume``.  It
uses the independent team validator and records every rejected candidate so a
known-set failure can never reach the more permissive fresh >=95% gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parent / "agent_changed_tasks" / "broadcast_prunes"
VALIDATOR = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"
BASELINE = ROOT / "submission_base_8003.40.zip"
DATA_DIR = ROOT / "inputs/neurogolf-2026"
DATA_ZIP = ROOT / "inputs/neurogolf-2026.zip"
BIAS_CHECKER = ROOT / "scripts/golf/check_conv_bias.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conv_bias_issues(path: Path) -> list[list[object]]:
    namespace: dict[str, object] = {
        "__name__": "check_conv_bias_import",
        "__file__": str(BIAS_CHECKER),
    }
    exec(BIAS_CHECKER.read_text(), namespace)  # noqa: S102 - trusted local checker
    issues = namespace["check_model"](onnx.load(path))
    return [list(item) for item in issues]


def validate(path: Path) -> dict[str, object]:
    task = int(path.name[4:7])
    out = HERE / "known" / f"{path.stem}.json"
    log = HERE / "known" / f"{path.stem}.log"
    command = [
        sys.executable,
        str(VALIDATOR),
        "validate-task",
        "--task",
        str(task),
        "--candidate-model",
        str(path),
        "--baseline-zip",
        str(BASELINE),
        "--data-dir",
        str(DATA_DIR),
        "--data-zip",
        str(DATA_ZIP),
        "--random-cases",
        "0",
        "--seed",
        str(8_003_400 + task),
        "--out-json",
        str(out),
    ]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
    log.write_text(proc.stdout + proc.stderr)
    payload = json.loads(out.read_text()) if out.exists() else {}
    candidate = payload.get("candidate", {})
    decision = payload.get("decision", {})
    issues = conv_bias_issues(path)
    known = candidate.get("known", {})
    known_complete = bool(
        known.get("total_seen", 0) > 0
        and known.get("wrong") == 0
        and known.get("errors") == 0
        and known.get("right") == known.get("total_seen")
    )
    return {
        "task": task,
        "candidate": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "returncode": proc.returncode,
        "checker_and_shape_preflight": candidate.get("preflight_ok"),
        "valid": candidate.get("valid"),
        "known": known,
        "cost": candidate.get("cost"),
        "baseline_cost": payload.get("baseline", {}).get("cost"),
        "cost_reduction": decision.get("cost_reduction"),
        "projected_gain": decision.get("projected_gain"),
        "validator_verdict": decision.get("verdict"),
        "conv_bias_issues": issues,
        "known_complete": known_complete,
        "eligible_for_fresh": bool(
            proc.returncode == 0
            and known_complete
            and candidate.get("preflight_ok") is True
            and not issues
            and (decision.get("cost_reduction") or 0) > 0
        ),
    }


def main() -> None:
    (HERE / "known").mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob("task260_*.onnx")) + sorted(SOURCE.glob("task359_*.onnx"))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(validate, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "task": int(path.name[4:7]),
                    "candidate": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "eligible_for_fresh": False,
                    "audit_error": repr(exc),
                }
            rows.append(row)
            print(path.name, row.get("known"), row.get("eligible_for_fresh"), flush=True)
    rows.sort(key=lambda row: str(row["candidate"]))
    summary = {
        "baseline": "submission_base_8003.40.zip",
        "candidate_count": len(rows),
        "known_complete_count": sum(bool(row.get("known_complete")) for row in rows),
        "eligible_for_fresh_count": sum(bool(row.get("eligible_for_fresh")) for row in rows),
        "candidates": rows,
    }
    (HERE / "existing_candidate_known_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}))


if __name__ == "__main__":
    main()
