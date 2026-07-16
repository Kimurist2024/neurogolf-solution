"""Submit artifacts/submission.zip to Kaggle when the expected score improves.

Policy (user directive): submit on EVERY improvement, even if only a single
task improved. The comparison metric is the sum of expected scores from the
latest run report (``artifacts/reports/run-NNN.json``), tracked against the
best previously-submitted value in ``artifacts/best_score.json``.

Usage:
    .venv/bin/python scripts/submit_if_improved.py            # compare + submit
    .venv/bin/python scripts/submit_if_improved.py --dry-run  # compare only
    .venv/bin/python scripts/submit_if_improved.py --force -m "msg"  # always submit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
SUBMISSION_ZIP = REPO_ROOT / "artifacts" / "submission.zip"
BEST_SCORE_FILE = REPO_ROOT / "artifacts" / "best_score.json"
COMPETITION = "neurogolf-2026"


NUM_TASKS = 400


def latest_run_report() -> tuple[int, dict] | None:
    """Latest run report that covers ALL tasks (partial reruns are skipped)."""
    runs = []
    for path in REPORTS_DIR.glob("run-*.json"):
        stem = path.stem.replace("run-", "")
        if stem.isdigit():
            runs.append((int(stem), path))
    for run_number, path in sorted(runs, reverse=True):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if len(payload.get("tasks", [])) == NUM_TASKS:
            return run_number, payload
    return None


def load_best() -> dict:
    if BEST_SCORE_FILE.is_file():
        with open(BEST_SCORE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"score": None, "run": None}


def save_best(score: float, run_number: int, message: str) -> None:
    BEST_SCORE_FILE.write_text(
        json.dumps(
            {"score": score, "run": run_number, "message": message}, indent=2
        ),
        encoding="utf-8",
    )


def submit(message: str) -> bool:
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
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Compare only.")
    parser.add_argument(
        "--force", action="store_true", help="Submit regardless of comparison."
    )
    parser.add_argument("-m", "--message", default="", help="Submission message.")
    args = parser.parse_args(argv)

    if not SUBMISSION_ZIP.is_file():
        print(f"ERROR: {SUBMISSION_ZIP} not found. Build it first (--zip).")
        return 1

    report = latest_run_report()
    if report is None:
        print("ERROR: no run reports found in artifacts/reports/.")
        return 1
    run_number, payload = report
    current = float(payload["totals"]["score_after"])

    best = load_best()
    best_score = best.get("score")
    print(f"Latest run:  run-{run_number:03d}  expected score = {current:.4f}")
    print(
        f"Best so far: {best_score:.4f} (run-{best['run']:03d})"
        if best_score is not None
        else "Best so far: none (never submitted)"
    )

    improved = best_score is None or current > best_score
    if not improved and not args.force:
        print("No improvement — not submitting.")
        return 0

    message = args.message or (
        f"run-{run_number:03d} expected {current:.4f} "
        f"(prev best {best_score:.4f})"
        if best_score is not None
        else f"run-{run_number:03d} expected {current:.4f} (first submission)"
    )

    if args.dry_run:
        print(f"DRY RUN — would submit with message: {message!r}")
        return 0

    if not submit(message):
        print("ERROR: kaggle submit failed; best_score.json NOT updated.")
        return 1
    save_best(current, run_number, message)
    print(f"Submitted and recorded new best: {current:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
