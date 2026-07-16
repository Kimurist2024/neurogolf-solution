#!/usr/bin/env python3
"""Audit candidates on identical fresh cases with disabled and default ORT."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402

MAP = json.loads((ROOT / "docs" / "golf" / "task_hash_map.json").read_text())


def parse_tasks(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--pattern", default="task{task:03d}.onnx")
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13_700_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = args.models if args.models.is_absolute() else ROOT / args.models
    ort.set_default_logger_severity(3)
    rows: list[dict[str, object]] = []
    for task in parse_tasks(args.tasks):
        rendered = args.pattern.format(task=task)
        matches = sorted(models.glob(rendered)) if any(char in rendered for char in "*?[") else []
        path = matches[0] if matches else models / rendered
        row: dict[str, object] = {"task": task, "path": str(path.relative_to(ROOT))}
        try:
            model = onnx.load(path)
            module = importlib.import_module(f"task_{MAP[f'{task:03d}']}")
            random.seed(args.seed + task)
            examples: list[dict[str, object]] = []
            generation_errors = 0
            for _ in range(args.k):
                try:
                    example = module.generate()
                    if isinstance(example, dict) and "input" in example and "output" in example:
                        examples.append(example)
                    else:
                        generation_errors += 1
                except Exception:
                    generation_errors += 1
            for disabled, label in ((True, "disable_all"), (False, "default")):
                try:
                    session = make_session(model, disabled)
                    right, wrong, _ = scoring.verify_subset(session, examples)
                    row[label] = {
                        "right": right,
                        "wrong": wrong,
                        "runtime_or_output_failures": wrong,
                        "perfect": right == len(examples) and wrong == 0,
                    }
                except Exception as exc:
                    row[label] = {"perfect": False, "session_error": repr(exc)}
            row["generated"] = len(examples)
            row["generation_errors"] = generation_errors
            row["perfect"] = bool(
                generation_errors == 0
                and row["disable_all"].get("perfect")
                and row["default"].get("perfect")
            )
        except Exception as exc:
            row.update({"perfect": False, "error": repr(exc)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(row.get("perfect") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
