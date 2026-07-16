#!/usr/bin/env python3
"""Rescore all historical ONNX files that can beat an 8010.03 cost<=50 member."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import onnx
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8010.03.zip"
AUTHORITY_SHA256 = "d772399d4535176b95039690eca59808059add3c0ca2d42e2124f17c705ec2e6"
EVIDENCE = HERE / "evidence.json"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_from_path(path: str) -> int | None:
    match = re.search(r"task[_-]?(\d{3})(?!\d)", path, re.IGNORECASE)
    if match is None:
        match = re.search(r"task(\d{3})", path, re.IGNORECASE)
    return int(match.group(1)) if match else None


def current_costs() -> dict[int, int]:
    result: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if cost <= 50 and float(row["score"]) < 25.0:
                result[task] = cost
    return result


def parameter_count(model: onnx.ModelProto) -> int:
    value = scoring.calculate_params(model)
    return int(value) if value is not None else 10**18


def declared_lower_bound(model: onnx.ModelProto) -> int:
    """Cheap lower bound; unknown intermediates contribute zero and are reprofiled later."""
    total = parameter_count(model)
    graph_outputs = {value.name for value in model.graph.output}
    values = {
        value.name: value
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    counted: set[str] = set()
    for node in model.graph.node:
        for name in node.output:
            if not name or name in graph_outputs or name in counted:
                continue
            counted.add(name)
            value = values.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            tensor = value.type.tensor_type
            dims = [int(dim.dim_value) for dim in tensor.shape.dim]
            if not dims or any(dim <= 0 for dim in dims):
                continue
            try:
                itemsize = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type).itemsize
            except Exception:
                itemsize = 0
            total += math.prod(dims) * itemsize
    return int(total)


def known_exact_fast(model: onnx.ModelProto, task: int) -> tuple[bool, int]:
    """Fail on the first visible/generated mismatch before expensive profiling."""
    session = scoring._make_raw_session(model)
    if session is None:
        return False, 0
    checked = 0
    examples = scoring.load_examples(task)
    for subset in ("train", "test", "arc-gen"):
        for example in examples[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = scoring._raw_output(session, benchmark["input"])
            except Exception:
                return False, checked
            checked += 1
            if not np.array_equal((raw > 0.0).astype(np.float32), benchmark["output"]):
                return False, checked
    return True, checked


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs = current_costs()
    raw_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True
    ).splitlines()
    seen: set[tuple[int, str]] = set()
    candidates: list[dict[str, object]] = []
    for relpath in raw_paths:
        task = task_from_path(relpath)
        if task not in costs:
            continue
        path = ROOT / relpath
        try:
            data = path.read_bytes()
            digest = sha256(data)
            key = (task, digest)
            if key in seen:
                continue
            seen.add(key)
            model = onnx.load_model_from_string(data)
        except Exception:
            continue
        params = parameter_count(model)
        lower_bound = declared_lower_bound(model)
        if params >= costs[task] or lower_bound >= costs[task]:
            continue
        candidates.append({
            "task": task,
            "path": relpath,
            "sha256": digest,
            "authority_cost": costs[task],
            "params": params,
            "declared_lower_bound": lower_bound,
            "node_count": len(model.graph.node),
            "ops": [node.op_type for node in model.graph.node],
        })

    results: list[dict[str, object]] = []
    for index, row in enumerate(candidates, start=1):
        task = int(row["task"])
        path = ROOT / str(row["path"])
        model = onnx.load(path)
        fast_exact, fast_checked = known_exact_fast(model, task)
        profile = None
        if fast_exact:
            with tempfile.TemporaryDirectory(prefix=f"hist298_{task:03d}_", dir="/tmp") as work:
                profile = scoring.score_and_verify(
                    model, task, work, label=f"hist{index}", require_correct=False
                )
        item = dict(row)
        item["fast_known_exact"] = fast_exact
        item["fast_known_checked"] = fast_checked
        item["profile"] = profile
        item["strict_lower_actual"] = bool(
            profile is not None and int(profile["cost"]) < costs[task]
        )
        item["known_exact"] = bool(profile is not None and profile["correct"])
        item["historical_winner"] = bool(item["strict_lower_actual"] and item["known_exact"])
        results.append(item)
        print(json.dumps({
            "index": index,
            "total": len(candidates),
            "task": task,
            "cost": None if profile is None else profile["cost"],
            "correct": None if profile is None else profile["correct"],
            "winner": item["historical_winner"],
        }), flush=True)

    winners = [row for row in results if row["historical_winner"]]
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "scope": "all non-score25 authority tasks with cost<=50",
        "path_count": len(raw_paths),
        "unique_task_sha_count": len(seen),
        "theoretical_strict_lower_candidates": len(candidates),
        "known_exact_strict_lower_winners": len(winners),
        "winners": winners,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only this evidence directory",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidates": len(candidates),
        "winners": winners,
        "evidence": str(EVIDENCE.relative_to(ROOT)),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
