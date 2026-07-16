#!/usr/bin/env python3
"""Parallel exhaustive color-orbit audit for task328 candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


CONFIGS = (
    (True, 1, "disable_t1"),
    (True, 4, "disable_t4"),
)


def make_session(blob: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(blob)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options)


def specs() -> list[dict[str, object]]:
    rows = []
    for size in range(6, 19):
        corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        for count in range(2, 5):
            for selected in itertools.combinations(corners, count):
                rows.append({"size": size, "selected": selected,
                             "colors": tuple(range(1, count + 1))})
    return rows


def empty() -> dict[str, object]:
    return {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0,
            "small_positive": 0, "min_positive": None, "max_false": None,
            "max_abs": 0.0, "first_failure": None}


def worker(payload: tuple[int, list[dict[str, object]], bytes]) -> dict[str, object]:
    shard, rows, blob = payload
    generator = importlib.import_module("task_d22278a0")
    sessions = {label: make_session(blob, disabled, threads)
                for disabled, threads, label in CONFIGS}
    stats = {label: empty() for _, _, label in CONFIGS}
    for spec in rows:
        selected = spec["selected"]
        assert isinstance(selected, tuple)
        rr, cc = zip(*selected)
        example = generator.generate(size=int(spec["size"]), rows=rr, cols=cc,
                                     colors=spec["colors"])
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"unconvertible case: {spec}")
        target = benchmark["output"] > 0.0
        for label, session in sessions.items():
            row = stats[label]
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                finite = np.isfinite(raw)
                row["nonfinite"] = int(row["nonfinite"]) + int((~finite).sum())
                if finite.any():
                    row["max_abs"] = max(float(row["max_abs"]),
                                         float(np.abs(raw[finite]).max()))
                true = raw[target]
                false = raw[~target]
                row["small_positive"] = int(row["small_positive"]) + int(
                    ((raw > 0.0) & (raw < 0.25)).sum()
                )
                if true.size and np.isfinite(true).all():
                    value = float(true.min())
                    current = row["min_positive"]
                    row["min_positive"] = value if current is None else min(float(current), value)
                if false.size and np.isfinite(false).all():
                    value = float(false.max())
                    current = row["max_false"]
                    row["max_false"] = value if current is None else max(float(current), value)
                if np.array_equal(raw > 0.0, target):
                    row["right"] = int(row["right"]) + 1
                else:
                    row["wrong"] = int(row["wrong"]) + 1
                    if row["first_failure"] is None:
                        row["first_failure"] = spec
            except Exception as exc:
                row["errors"] = int(row["errors"]) + 1
                if row["first_failure"] is None:
                    row["first_failure"] = {**spec, "error": f"{type(exc).__name__}: {exc}"}
    return {"shard": shard, "cases": len(rows), "configs": stats}


def merge(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for _, _, label in CONFIGS:
        parts = [row["configs"][label] for row in rows]
        mins = [part["min_positive"] for part in parts if part["min_positive"] is not None]
        max_false = [part["max_false"] for part in parts if part["max_false"] is not None]
        result[label] = {
            key: sum(int(part[key]) for part in parts)
            for key in ("right", "wrong", "errors", "nonfinite", "small_positive")
        }
        result[label]["min_positive"] = min(float(value) for value in mins) if mins else None
        result[label]["max_false"] = max(float(value) for value in max_false) if max_false else None
        result[label]["max_abs"] = max(float(part["max_abs"]) for part in parts)
        result[label]["first_failure"] = next(
            (part["first_failure"] for part in parts if part["first_failure"] is not None), None
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", action="store_true")
    parser.add_argument("--worker-payload", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_payload:
        payload = json.loads(args.worker_payload)
        model = Path(payload["model"])
        blob = model.read_bytes()
        all_cases = specs()
        shard = int(payload["shard"])
        workers = int(payload["workers"])
        print(json.dumps(worker((shard, all_cases[shard::workers], blob))))
        return 0
    if args.model is None or args.output is None:
        parser.error("--model and --output are required")
    blob = args.model.read_bytes()
    cases = specs()
    shards = [cases[index::args.workers] for index in range(args.workers)]
    payloads = [(index, shard, blob) for index, shard in enumerate(shards) if shard]
    rows = []
    if args.processes:
        processes = [
            subprocess.Popen(
                [sys.executable, __file__, "--worker-payload", json.dumps({
                    "model": str(args.model.resolve()), "shard": index,
                    "workers": args.workers,
                })],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for index in range(args.workers)
        ]
        for process in processes:
            stdout, stderr = process.communicate()
            if process.returncode:
                raise RuntimeError(f"worker failed ({process.returncode}): {stderr}")
            row = json.loads(stdout)
            rows.append(row)
            print(json.dumps({"shard": row["shard"], "cases": row["cases"]}), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, payload) for payload in payloads]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(json.dumps({"shard": row["shard"], "cases": row["cases"]}), flush=True)
    configs = merge(rows)
    perfect = all(
        row["right"] == len(cases) and row["wrong"] == row["errors"] == 0
        and row["nonfinite"] == row["small_positive"] == 0
        and float(row["min_positive"]) >= 0.25 and float(row["max_false"]) <= 0.0
        for row in configs.values()
    )
    result = {
        "model": str(args.model.resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "orbit_representatives": len(cases),
        "full_generator_states": 71136,
        "workers": args.workers,
        "configs": configs,
        "strict_perfect": perfect,
        "shards": sorted(rows, key=lambda row: int(row["shard"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "shards"}, indent=2))
    return 0 if perfect else 1


if __name__ == "__main__":
    raise SystemExit(main())
