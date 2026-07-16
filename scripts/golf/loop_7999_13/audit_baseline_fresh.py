#!/usr/bin/env python3
"""Lightweight domain-generator audit used only to prioritize search lanes."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import signal
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402

MAP = json.loads((ROOT / "docs" / "golf" / "task_hash_map.json").read_text())


def parse_tasks(specification: str) -> list[int]:
    tasks: set[int] = set()
    for item in specification.split(","):
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            tasks.update(range(start, end + 1))
        elif item:
            tasks.add(int(item))
    return sorted(task for task in tasks if 1 <= task <= 400)


def timeout_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("task audit timeout")


def session(model: onnx.ModelProto) -> ort.InferenceSession | None:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        return None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--tasks", default="1-400")
    parser.add_argument("--task-timeout", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ort.set_default_logger_severity(3)
    rows: list[dict[str, object]] = []
    signal.signal(signal.SIGALRM, timeout_handler)
    tasks = parse_tasks(args.tasks)
    for ordinal, task in enumerate(tasks, 1):
        row: dict[str, object] = {"task": task}
        try:
            signal.alarm(args.task_timeout)
            model = onnx.load(args.models / f"task{task:03d}.onnx")
            gen = importlib.import_module(f"task_{MAP[f'{task:03d}']}")
            random.seed(7_770_000 + task)
            examples = []
            generation_errors = 0
            for _ in range(args.k):
                try:
                    example = gen.generate()
                    if isinstance(example, dict) and "input" in example and "output" in example:
                        examples.append(example)
                except Exception:
                    generation_errors += 1
            current = session(model)
            if current is None:
                raise RuntimeError("sanitize failed")
            right, wrong, _ = scoring.verify_subset(current, examples)
            row.update({
                "generated": len(examples),
                "generation_errors": generation_errors,
                "right": right,
                "wrong": wrong,
                "rate": right / (right + wrong) if right + wrong else None,
                "perfect": bool(right + wrong == len(examples) and wrong == 0),
            })
        except Exception as exc:
            row.update({"perfect": False, "error": repr(exc)})
        finally:
            signal.alarm(0)
        rows.append(row)
        if ordinal % 25 == 0:
            print(f"audited {ordinal}/{len(tasks)} (task{task:03d})", flush=True)
    payload = {
        "models": str(args.models),
        "k": args.k,
        "tasks": args.tasks,
        "perfect_tasks": [row["task"] for row in rows if row.get("perfect")],
        "imperfect_tasks": [row["task"] for row in rows if not row.get("perfect")],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "perfect": len(payload["perfect_tasks"]),
        "imperfect": len(payload["imperfect_tasks"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
