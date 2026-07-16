#!/usr/bin/env python3
"""Render ARC-AGI task grids as PNG contact sheets."""

from __future__ import annotations

import argparse
import html
import json
import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "neurogolf_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


INPUT_DIR = Path("inputs/neurogolf-2026")
OUTPUT_DIR = Path("artifacts/task_viz")
TASK_MIN = 1
TASK_MAX = 400
ARC_GEN_LIMIT = 2
PAIRS_PER_ROW = 2

ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)

Grid = list[list[int]]
TaskPair = dict[str, Grid]


@dataclass(frozen=True)
class RenderJob:
    task_id: int
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DisplayPair:
    split: str
    index: int
    pair: TaskPair


def parse_task_selection(raw: str | None) -> list[int]:
    if raw is None:
        return list(range(TASK_MIN, TASK_MAX + 1))

    selected: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"Invalid descending task range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    if not selected:
        raise ValueError("--tasks did not contain any task IDs")

    out_of_range = [task_id for task_id in selected if not TASK_MIN <= task_id <= TASK_MAX]
    if out_of_range:
        raise ValueError(
            f"Task IDs must be between {TASK_MIN} and {TASK_MAX}: {out_of_range}"
        )

    return sorted(selected)


def task_name(task_id: int) -> str:
    return f"task{task_id:03d}"


def grid_shape(grid: Grid) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


def collect_display_pairs(task_data: dict[str, list[TaskPair]]) -> list[DisplayPair]:
    pairs: list[DisplayPair] = []

    for split in ("train", "test"):
        for index, pair in enumerate(task_data.get(split, [])):
            pairs.append(DisplayPair(split=split, index=index, pair=pair))

    for index, pair in enumerate(task_data.get("arc-gen", [])[:ARC_GEN_LIMIT]):
        pairs.append(DisplayPair(split="arc-gen", index=index, pair=pair))

    return pairs


def draw_grid(ax: plt.Axes, grid: Grid, label: str) -> None:
    array = np.asarray(grid, dtype=int)
    rows, cols = array.shape

    ax.imshow(array, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(label, fontsize=8, pad=5)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)

    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.5)


def add_arrow_axis(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "→",
        ha="center",
        va="center",
        fontsize=16,
        color="#333333",
        transform=ax.transAxes,
    )


def pair_grid_labels(display_pair: DisplayPair) -> tuple[str, str]:
    in_rows, in_cols = grid_shape(display_pair.pair["input"])
    out_rows, out_cols = grid_shape(display_pair.pair["output"])
    prefix = f"{display_pair.split}[{display_pair.index}]"
    return (
        f"{prefix} input ({in_rows}x{in_cols})",
        f"{prefix} output ({out_rows}x{out_cols})",
    )


def render_task(job: RenderJob) -> str:
    task_id = job.task_id
    name = task_name(task_id)
    input_path = job.input_dir / f"{name}.json"
    output_path = job.output_dir / f"{name}.png"
    job.output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        task_data: dict[str, list[TaskPair]] = json.load(handle)

    display_pairs = collect_display_pairs(task_data)
    row_count = max(1, math.ceil(len(display_pairs) / PAIRS_PER_ROW))
    arc_total = len(task_data.get("arc-gen", []))
    arc_shown = min(ARC_GEN_LIMIT, arc_total)

    fig_width = 16.0
    fig_height = max(3.4, 3.7 * row_count + 0.7)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=140)
    grid_spec = fig.add_gridspec(
        nrows=row_count,
        ncols=7,
        width_ratios=[1.0, 0.08, 1.0, 0.22, 1.0, 0.08, 1.0],
        left=0.025,
        right=0.985,
        bottom=0.03,
        top=0.91,
        wspace=0.12,
        hspace=0.62,
    )

    fig.suptitle(
        (
            f"{name} | train: {len(task_data.get('train', []))} | "
            f"test: {len(task_data.get('test', []))} | "
            f"arc-gen: {arc_shown}/{arc_total} shown"
        ),
        fontsize=13,
        y=0.975,
    )

    for pair_number, display_pair in enumerate(display_pairs):
        row = pair_number // PAIRS_PER_ROW
        pair_column = pair_number % PAIRS_PER_ROW
        base_column = 0 if pair_column == 0 else 4

        input_ax = fig.add_subplot(grid_spec[row, base_column])
        arrow_ax = fig.add_subplot(grid_spec[row, base_column + 1])
        output_ax = fig.add_subplot(grid_spec[row, base_column + 2])

        input_label, output_label = pair_grid_labels(display_pair)
        draw_grid(input_ax, display_pair.pair["input"], input_label)
        add_arrow_axis(arrow_ax)
        draw_grid(output_ax, display_pair.pair["output"], output_label)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def write_index(task_ids: list[int], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"

    cards = []
    for task_id in task_ids:
        name = html.escape(task_name(task_id))
        image_name = f"{name}.png"
        cards.append(
            "\n".join(
                [
                    '<article class="task-card">',
                    f"  <h2>{name}</h2>",
                    f'  <a href="{image_name}">',
                    (
                        f'    <img src="{image_name}" alt="{name} visualization" '
                        'loading="lazy">'
                    ),
                    "  </a>",
                    "</article>",
                ]
            )
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARC-AGI Task Visualizations</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 24px;
      font-weight: 700;
    }}
    .meta {{
      margin: 0 0 20px;
      color: #52606d;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .task-card {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      padding: 10px;
    }}
    .task-card h2 {{
      margin: 0 0 8px;
      font-size: 15px;
      line-height: 1.2;
    }}
    .task-card img {{
      display: block;
      width: 100%;
      max-width: 420px;
      height: auto;
      border: 1px solid #bcccdc;
    }}
  </style>
</head>
<body>
  <h1>ARC-AGI Task Visualizations</h1>
  <p class="meta">{len(task_ids)} task(s)</p>
  <main class="grid">
{chr(10).join(cards)}
  </main>
</body>
</html>
"""
    index_path.write_text(document, encoding="utf-8")
    return index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize ARC-AGI input/output grids for Neurogolf tasks."
    )
    parser.add_argument(
        "--tasks",
        help="Task IDs to render, e.g. 1,2,3 or 1-10. Defaults to all 400 tasks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to min(CPU count, tasks, 8).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        task_ids = parse_task_selection(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))

    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    worker_count = args.workers or min(os.cpu_count() or 1, len(task_ids), 8)
    jobs = [RenderJob(task_id=task_id, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR) for task_id in task_ids]

    with mp.Pool(processes=worker_count) as pool:
        rendered = list(pool.imap_unordered(render_task, jobs))

    index_path = write_index(task_ids, OUTPUT_DIR)
    print(
        f"Rendered {len(rendered)} task(s) to {OUTPUT_DIR} "
        f"with index {index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
