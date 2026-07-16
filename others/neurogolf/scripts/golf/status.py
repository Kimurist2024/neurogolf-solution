"""Show the per-task handcrafted golf queue status."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib import scoring  # noqa: E402

REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "run-012.json"
HANDCRAFTED_DIR = REPO_ROOT / "artifacts" / "handcrafted"
TARGET_COST = 900
NUM_TASKS = 400


@dataclass(frozen=True)
class Row:
    task: int
    optimized_cost: int
    handcrafted_cost: int | None
    incumbent_cost: int
    remaining_gain: float


def _score_for_cost(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _load_optimized_costs() -> dict[int, int]:
    with open(REPORT_PATH) as f:
        report = json.load(f)
    out: dict[int, int] = {}
    for item in report.get("tasks", []):
        out[int(item["task"])] = int(item["cost_post_002"])
    return out


def _score_handcrafted(task_num: int, workdir: str) -> int | None:
    path = HANDCRAFTED_DIR / f"task{task_num:03d}.onnx"
    if not path.is_file():
        return None
    model = onnx.load(str(path))
    scored = scoring.score_and_verify(
        model,
        task_num,
        workdir,
        label=f"status{task_num:03d}",
        require_correct=True,
    )
    if scored is None:
        return None
    return int(scored["cost"])


def _rows() -> list[Row]:
    optimized_costs = _load_optimized_costs()
    target_score = _score_for_cost(TARGET_COST)
    rows: list[Row] = []

    with tempfile.TemporaryDirectory(prefix="neurogolf_status_") as workdir:
        for task_num in range(1, NUM_TASKS + 1):
            optimized_cost = optimized_costs[task_num]
            handcrafted_cost = _score_handcrafted(task_num, workdir)
            if handcrafted_cost is not None and handcrafted_cost < optimized_cost:
                incumbent_cost = handcrafted_cost
            else:
                incumbent_cost = optimized_cost
            remaining_gain = max(0.0, target_score - _score_for_cost(incumbent_cost))
            rows.append(
                Row(
                    task=task_num,
                    optimized_cost=optimized_cost,
                    handcrafted_cost=handcrafted_cost,
                    incumbent_cost=incumbent_cost,
                    remaining_gain=remaining_gain,
                )
            )

    rows.sort(key=lambda row: (row.remaining_gain, row.incumbent_cost), reverse=True)
    return rows


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _print_table(rows: list[Row]) -> None:
    headers = ["task", "optimized", "handcrafted", "incumbent", "gain_to_900"]
    print(
        f"{headers[0]:>4}  {headers[1]:>10}  {headers[2]:>11}  "
        f"{headers[3]:>10}  {headers[4]:>11}"
    )
    print(f"{'-' * 4}  {'-' * 10}  {'-' * 11}  {'-' * 10}  {'-' * 11}")
    for row in rows:
        print(
            f"{row.task:>4}  {row.optimized_cost:>10}  "
            f"{_fmt_int(row.handcrafted_cost):>11}  "
            f"{row.incumbent_cost:>10}  {row.remaining_gain:>11.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, help="Only show the top N rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _rows()
    if args.top is not None:
        if args.top < 0:
            raise SystemExit("--top must be non-negative")
        rows = rows[: args.top]
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
