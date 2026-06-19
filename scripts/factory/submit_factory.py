#!/usr/bin/env python3
"""Submit the factory merge zip when merge-001 expected score improves."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_JSON = REPO_ROOT / "artifacts" / "reports" / "merge-001.json"
SUBMISSION_ZIP = REPO_ROOT / "artifacts" / "submission.zip"
BEST_SCORE_FILE = REPO_ROOT / "artifacts" / "best_score.json"
COMPETITION = "neurogolf-2026"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_best(score: float, message: str) -> None:
    run_id = "factory-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    BEST_SCORE_FILE.write_text(
        json.dumps(
            {
                "score": score,
                "run": run_id,
                "message": message,
                "source_report": str(MERGE_JSON.relative_to(REPO_ROOT)),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _current_score() -> float:
    report = _read_json(MERGE_JSON, None)
    if report is None:
        raise FileNotFoundError(f"{MERGE_JSON} not found")
    return float(report["totals"]["score_after"])


def _submit(message: str) -> bool:
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(SUBMISSION_ZIP),
        "-m",
        message,
    ]
    print("Submitting:", " ".join(cmd))
    return subprocess.run(cmd, capture_output=False).returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-m", "--message", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SUBMISSION_ZIP.is_file():
        print(f"ERROR: {SUBMISSION_ZIP} not found. Run merge --zip first.")
        return 1

    current = _current_score()
    best = _read_json(BEST_SCORE_FILE, {"score": None, "run": None})
    best_score = best.get("score")
    print(f"Factory merge expected score = {current:.4f}")
    if best_score is None:
        print("Best so far: none")
    else:
        print(f"Best so far: {float(best_score):.4f} ({best.get('run')})")

    improved = best_score is None or current > float(best_score)
    if not improved and not args.force:
        print("No improvement - not submitting.")
        return 0

    message = args.message or (
        f"factory expected {current:.4f}"
        if best_score is None
        else f"factory expected {current:.4f} (prev {float(best_score):.4f})"
    )
    if args.dry_run:
        print(f"DRY RUN - would submit with message: {message!r}")
        return 0

    if not _submit(message):
        print("ERROR: kaggle submit failed; best_score.json NOT updated.")
        return 1
    _write_best(current, message)
    print(f"Submitted and recorded new best: {current:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
