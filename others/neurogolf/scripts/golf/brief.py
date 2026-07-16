"""Generate per-task golf briefs for handcrafted NeuroGolf workers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lib import scoring  # noqa: E402

OPTIMIZED_DIR = REPO_ROOT / "artifacts" / "optimized"
BRIEFS_DIR = REPO_ROOT / "docs" / "golf" / "briefs"
COST_GAP_PATH = REPO_ROOT / "docs" / "research" / "cost-gap-analysis.json"
TARGET_COSTS = (900, 314)


def _score_for_cost(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _task_path(task_num: int) -> Path:
    return OPTIMIZED_DIR / f"task{task_num:03d}.onnx"


def _grid_shape(grid: list[list[int]]) -> str:
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def _render_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _render_pair(name: str, example: dict[str, Any]) -> list[str]:
    input_grid = example["input"]
    output_grid = example["output"]
    return [
        f"### {name}",
        f"input {_grid_shape(input_grid)} -> output {_grid_shape(output_grid)}",
        "",
        "input:",
        "```text",
        _render_grid(input_grid),
        "```",
        "",
        "output:",
        "```text",
        _render_grid(output_grid),
        "```",
        "",
    ]


def _load_cost_gap_entry(task_num: int) -> dict[str, Any] | None:
    if not COST_GAP_PATH.is_file():
        return None
    with open(COST_GAP_PATH) as f:
        data = json.load(f)
    for rank, entry in enumerate(data.get("priority_queue_top50", []), start=1):
        if entry.get("task") == task_num:
            out = dict(entry)
            out["priority_rank"] = rank
            return out
    return None


def _model_anatomy(task_num: int, model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    op_hist = Counter(node.op_type for node in inferred.graph.node)

    with tempfile.TemporaryDirectory(prefix="neurogolf_brief_") as workdir:
        scored = scoring.score_and_verify(
            model,
            task_num,
            workdir,
            label="brief",
            require_correct=False,
        )

    if scored is None:
        cost = memory = params = score = None
        correct = None
    else:
        cost = scored["cost"]
        memory = scored["memory"]
        params = scored["params"]
        score = scored["score"]
        correct = scored["correct"]

    return {
        "cost": cost,
        "memory": memory,
        "params": params,
        "score": score,
        "correct": correct,
        "n_nodes": len(inferred.graph.node),
        "n_value_info": len(inferred.graph.value_info),
        "op_hist": op_hist,
        "file_size": model_path.stat().st_size,
    }


def _format_optional_int(value: Any) -> str:
    return "n/a" if value is None else f"{int(value)}"


def _format_optional_float(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _build_brief(task_num: int) -> str:
    examples = scoring.load_examples(task_num)
    model_path = _task_path(task_num)
    if not model_path.is_file():
        raise FileNotFoundError(f"optimized model missing: {model_path}")

    anatomy = _model_anatomy(task_num, model_path)
    cost_gap = _load_cost_gap_entry(task_num)

    lines: list[str] = [
        f"# Task {task_num:03d} Golf Brief",
        "",
        "## Current Net",
        f"- path: `{model_path.relative_to(REPO_ROOT)}`",
        f"- file size: {anatomy['file_size']} bytes",
        f"- cost: {_format_optional_int(anatomy['cost'])}",
        f"- score: {_format_optional_float(anatomy['score'])}",
        f"- memory: {_format_optional_int(anatomy['memory'])}",
        f"- params: {_format_optional_int(anatomy['params'])}",
        f"- nodes: {anatomy['n_nodes']}",
        f"- value_info tensors after shape inference: {anatomy['n_value_info']}",
        f"- local gold-correct: {anatomy['correct']}",
        "",
    ]

    if cost_gap:
        lines.extend(
            [
                "## Research Queue",
                f"- priority rank: {cost_gap['priority_rank']}",
                f"- recorded cost: {cost_gap.get('cost', 'n/a')}",
                f"- recorded memory: {cost_gap.get('memory', 'n/a')}",
                f"- recorded params: {cost_gap.get('params', 'n/a')}",
                f"- recorded nodes: {cost_gap.get('n_nodes', 'n/a')}",
                "",
            ]
        )

    lines.extend(["## Op Histogram", ""])
    for op_type, count in anatomy["op_hist"].most_common():
        lines.append(f"- {op_type}: {count}")
    lines.append("")

    current_score = anatomy["score"]
    lines.extend(["## Targets", ""])
    for target_cost in TARGET_COSTS:
        target_score = _score_for_cost(target_cost)
        if current_score is None:
            delta = "n/a"
        else:
            delta = f"{target_score - current_score:+.6f}"
        lines.append(
            f"- cost {target_cost}: score {target_score:.6f}, delta {delta}"
        )
    lines.append("")

    train = list(examples.get("train", []))
    test = list(examples.get("test", []))
    arc_gen = list(examples.get("arc-gen", []))
    shown_arc_gen = arc_gen[:3]
    lines.extend(
        [
            "## Examples",
            f"- train: {len(train)} shown",
            f"- test: {len(test)} shown",
            f"- arc-gen: {len(shown_arc_gen)} shown, "
            f"{max(0, len(arc_gen) - len(shown_arc_gen))} remaining",
            "",
        ]
    )

    for idx, example in enumerate(train, start=1):
        lines.extend(_render_pair(f"train[{idx}]", example))
    for idx, example in enumerate(test, start=1):
        lines.extend(_render_pair(f"test[{idx}]", example))
    for idx, example in enumerate(shown_arc_gen, start=1):
        lines.extend(_render_pair(f"arc-gen[{idx}]", example))

    lines.extend(
        [
            "## Verification",
            "```bash",
            ".venv/bin/python scripts/golf/try_candidate.py "
            f"--task {task_num} --onnx path/to/candidate.onnx",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True, help="Task number 1..400")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_num = args.task
    if not 1 <= task_num <= 400:
        raise SystemExit("--task must be in 1..400")

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRIEFS_DIR / f"task{task_num:03d}.md"
    out_path.write_text(_build_brief(task_num), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
