#!/usr/bin/env python3
"""Audit one deterministic 500-case chunk of task275 fresh validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = 275
CANDIDATE = HERE / "task275_diagonal_reuse_cost419_c7ddaab77f6d.onnx"
EXPECTED_SHA256 = "c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2"
SEEDS = (275_419_001, 275_419_002)
CHUNK_SIZE = 500


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LANE = import_path(
    "task275_gold_chunk_support",
    ROOT / "scripts/golf/cost351_500_gold_loop/worker.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--chunk", type=int, choices=(0, 1, 2, 3), required=True)
    args = parser.parse_args()

    data = CANDIDATE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError((digest, EXPECTED_SHA256))
    seed = SEEDS[args.seed_index]
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    cases, generation = LANE.BASE.SUPPORT.fresh_cases(TASK, seed, task_map)
    if len(cases) != 2_000:
        raise RuntimeError(f"expected 2000 fresh cases, got {len(cases)}")
    start = args.chunk * CHUNK_SIZE
    selected = cases[start : start + CHUNK_SIZE]
    runtime = LANE.BASE.failfast_known(data, selected)
    passed = bool(
        runtime.get("early_reject_reason") is None and LANE.BASE.runtime_pass(runtime)
    )
    payload = {
        "task": TASK,
        "candidate_sha256": digest,
        "seed": seed,
        "seed_index": args.seed_index,
        "chunk": args.chunk,
        "start": start,
        "stop": start + len(selected),
        "case_count": len(selected),
        "generation": generation,
        "runtime": runtime,
        "pass": passed,
    }
    output = HERE / f"fresh_seed{seed}_chunk{args.chunk}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise RuntimeError("fresh chunk rejected candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
