#!/usr/bin/env python3
"""Run the two 500-case task157 fresh seeds in parallel processes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71405/task157_improved(1).onnx"
SHA256 = "a1254f2619406b8db5d3fe5fdd1c42c917820fa51b91faef0f3ceed5d8b3662f"
SEEDS = (98_000_157, 98_100_157)
TASK = 157


def load_helpers():
    path = ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py"
    spec = importlib.util.spec_from_file_location("lane98_fresh_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import audit helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_seed(seed: int) -> dict:
    helpers = load_helpers()
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("task157 candidate hash mismatch")
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{TASK:03d}']}")
    configs = (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    )
    sessions = {name: helpers.make_session(data, disabled, threads) for disabled, threads, name in configs}
    stats = {name: {"right": 0, "wrong": 0, "errors": 0, "first_failure": None} for _, _, name in configs}
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    valid = attempts = generation_errors = conversion_skips = 0
    while valid < 500:
        attempts += 1
        try:
            benchmark = helpers.scoring.convert_to_numpy(generator.generate())
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if benchmark is None:
            conversion_skips += 1
            continue
        valid += 1
        want = benchmark["output"] > 0
        for _, _, name in configs:
            item = stats[name]
            try:
                session = sessions[name]
                raw = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
                if np.array_equal(raw > 0, want):
                    item["right"] += 1
                else:
                    item["wrong"] += 1
                    if item["first_failure"] is None:
                        item["first_failure"] = {
                            "valid_case": valid,
                            "different_cells": int(np.count_nonzero((raw > 0) != want)),
                        }
            except Exception as exc:  # noqa: BLE001
                item["errors"] += 1
                if item["first_failure"] is None:
                    item["first_failure"] = {"valid_case": valid, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "seed": seed,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "configs": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.seed is not None:
        run = run_seed(args.seed)
        path = HERE / f"audit/fresh_task157_seed{args.seed}.json"
        path.write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps({"task": TASK, "seed": args.seed, "valid": run["valid"]}, indent=2))
        return 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        runs = list(executor.map(run_seed, SEEDS))
    rates = [item["right"] / 500 for run in runs for item in run["configs"].values()]
    report = {
        "task": TASK,
        "sha256": SHA256,
        "source": str(SOURCE.relative_to(ROOT)),
        "count_per_seed": 500,
        "seeds": list(SEEDS),
        "runs": runs,
        "minimum_config_rate": min(rates),
        "maximum_config_rate": max(rates),
    }
    (HERE / "audit/fresh_task157.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "task": TASK,
        "minimum_config_rate": report["minimum_config_rate"],
        "maximum_config_rate": report["maximum_config_rate"],
        "runs": [{"seed": run["seed"], "valid": run["valid"]} for run in runs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
