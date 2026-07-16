#!/usr/bin/env python3
"""Run fresh 2×500 prioritization and collect exact-SHA text history."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (
    239, 222, 37, 226, 297, 14, 234, 92, 397, 264,
    394, 398, 200, 75, 392, 387, 225, 218, 36, 132,
)
SEEDS = (2026071401, 2026071402)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "expand20j_fresh_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
SCANNER.HERE = HERE
SCANNER.TARGETS = TARGETS
SCANNER.BASE_ZIP = AUTHORITY


def exact_text_history(digest: str) -> list[str]:
    command = [
        "rg", "-l", "--hidden", "--fixed-strings",
        "--glob", "!*.onnx", "--glob", "!*.zip", "--glob", "!*.pyc",
        "--glob", "!agent_expand20j_95/**", digest, ".",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return sorted(line.removeprefix("./") for line in result.stdout.splitlines() if line)


def main() -> int:
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")
    audited = json.loads((HERE / "audit" / "delta_official_known4.json").read_text())
    wanted = {
        row["sha256"]: row
        for row in audited["rows"]
        if row.get("known_perfect_all_configs")
        and (row.get("runtime_shape_trace") or {}).get("truthful")
    }
    candidates, _ = SCANNER.inventory()
    data_by_task: dict[int, list[dict]] = {}
    for task, task_rows in candidates.items():
        for digest, item in task_rows.items():
            if digest in wanted:
                data_by_task.setdefault(task, []).append(
                    {"sha256": digest, "data": item["data"], "sources": item["sources"]}
                )

    rows: list[dict] = []
    for task, task_candidates in sorted(data_by_task.items()):
        runs = []
        for seed in SEEDS:
            print(f"FRESH task{task:03d} seed={seed} candidates={len(task_candidates)}", flush=True)
            runs.append(SCANNER.fresh_dual(task, task_candidates, 500, seed))
        for candidate in task_candidates:
            digest = candidate["sha256"]
            row = {
                "task": task,
                "sha256": digest,
                "sources": sorted(set(candidate["sources"])),
                "exact_sha_text_history": exact_text_history(digest),
                "fresh_runs": [],
            }
            for seed, run in zip(SEEDS, runs):
                row["fresh_runs"].append(
                    {
                        "seed": seed,
                        "generation": {
                            "requested": run["requested"],
                            "valid": run["valid"],
                            "attempts": run["attempts"],
                            "generation_errors": run["generation_errors"],
                            "conversion_skips": run["conversion_skips"],
                        },
                        "modes": run["candidates"][digest],
                    }
                )
            rows.append(row)
    output = {
        "authority_zip_sha256": got,
        "fresh_is_prioritization_only": True,
        "seeds": list(SEEDS),
        "count_per_seed": 500,
        "candidate_count": len(rows),
        "rows": rows,
    }
    (HERE / "audit" / "fresh_history.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
