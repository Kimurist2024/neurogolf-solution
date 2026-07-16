#!/usr/bin/env python3
"""Dual-ORT incumbent check and readable Sakana-rule audit for low45."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (24, 113, 385, 389, 296, 399, 359, 110)
RULES = {
    24: "Fill every row containing color 1 or 3 with that marker color; in all remaining rows, fill columns containing color 2 with color 2.",
    113: "Keep the first five rows and append those rows in reverse order, yielding ten rows.",
    385: "Reverse the bottom five rows, then append the original bottom five rows, yielding ten rows.",
    389: "The grid contains color 5 and one other nonzero color; erase the other color's cells and recolor every 5 cell with that other color.",
    296: "Fold the fixed 5x7 grid across both center axes by bitwise OR, yielding a 3x3 grid.",
    399: "Count the disjoint 2x2 color-2 blocks and encode thresholds 1..5 in a fixed five-cell 3x3 pattern.",
    359: "For every cell choose the mode of its row concatenated with its column, with first-occurrence tie breaking (known-corpus exact but not generator-exact).",
    110: "Infer the vertical period of the 29x29 repeated pattern and restore each column from the nonzero periodic samples, using the shifted source cell as fallback.",
}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_known(model: onnx.ModelProto, task: int, disable: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    examples = scoring.load_examples(task)
    total = sum(len(examples[name]) for name in ("train", "test", "arc-gen"))
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {
            "right": 0,
            "wrong": 0,
            "errors": total,
            "total": total,
            "session_error": f"{type(exc).__name__}: {exc}",
        }
    right = wrong = errors = skipped = near_margin = 0
    first_failure = None
    shapes: set[tuple[int, ...]] = set()
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                skipped += 1
                continue
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                shapes.add(tuple(int(item) for item in raw.shape))
                near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {
                        "split": split,
                        "index": index,
                        "kind": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped": skipped,
        "total": right + wrong + errors,
        "near_margin_count": near_margin,
        "output_shapes": [list(item) for item in sorted(shapes)],
        "first_failure": first_failure,
    }


def audit_rule(task: int) -> dict[str, object]:
    raw_path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low45_task{task:03d}", raw_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(raw_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    right = wrong = errors = 0
    split_counts: dict[str, int] = {}
    first_failure = None
    input_shapes: set[tuple[int, int]] = set()
    output_shapes: set[tuple[int, int]] = set()
    for split, examples in payload.items():
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or "input" not in example or "output" not in example:
                continue
            split_counts[split] = split_counts.get(split, 0) + 1
            input_shapes.add((len(example["input"]), len(example["input"][0])))
            output_shapes.add((len(example["output"]), len(example["output"][0])))
            try:
                actual = normalize(module.p(copy.deepcopy(example["input"])))
                if actual == example["output"]:
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong"}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {
                        "split": split,
                        "index": index,
                        "kind": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    total = right + wrong + errors
    return {
        "task": task,
        "rule_summary": RULES[task],
        "known": {
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "total": total,
            "perfect": right == total and total > 0,
            "first_failure": first_failure,
        },
        "split_counts": split_counts,
        "input_shapes": [list(item) for item in sorted(input_shapes)],
        "output_shapes": [list(item) for item in sorted(output_shapes)],
    }


def main() -> None:
    known_rows = []
    rule_rows = []
    for task in TARGETS:
        model = onnx.load(HERE / "baselines" / f"task{task:03d}.onnx")
        row = {
            "task": task,
            "disable_all": run_known(model, task, True),
            "default": run_known(model, task, False),
        }
        known_rows.append(row)
        rule = audit_rule(task)
        rule_rows.append(rule)
        print(
            f"task{task:03d}: dual={row['disable_all']['right']}/{row['disable_all']['total']}+"
            f"{row['default']['right']}/{row['default']['total']} "
            f"rule={rule['known']['right']}/{rule['known']['total']}",
            flush=True,
        )
    (HERE / "known_baseline_dual.json").write_text(
        json.dumps({"targets_completed": len(known_rows), "rows": known_rows}, indent=2) + "\n"
    )
    (HERE / "true_rule_audit.json").write_text(
        json.dumps(
            {
                "source": "inputs/sakana-gcg-2025/raw/taskNNN.py",
                "dataset": "inputs/neurogolf-2026/taskNNN.json",
                "targets_completed": len(rule_rows),
                "all_perfect": all(row["known"]["perfect"] for row in rule_rows),
                "rows": rule_rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
