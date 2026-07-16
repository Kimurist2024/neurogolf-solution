#!/usr/bin/env python3
"""Deep audit the six highest-priority models from the new 71405 pool."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
CANDIDATES = {
    13: ROOT / "others/71405/task013_cost357.onnx",
    23: ROOT / "others/71405/task023_improved_v2.onnx",
    66: ROOT / "others/71405/task066_further_cost_reduced.onnx",
    69: ROOT / "others/71405/task069_further_improved.onnx",
    44: ROOT / "others/71405/task044_best_valid.onnx",
    46: ROOT / "others/71405/task046_reimproved.onnx",
}
SEEDS = (202607149701, 202607149702)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDITOR = load_module(
    "priority97_known_four",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
SWEEP = AUDITOR.SWEEP
SCANNER = load_module(
    "priority97_fresh",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)


def official_profile(task: int, data: bytes, label: str) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"priority97_{task:03d}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def exact_history(digest: str) -> list[str]:
    result = subprocess.run(
        [
            "rg", "-l", "--hidden", "--fixed-strings",
            "--glob", "!*.onnx", "--glob", "!*.zip", "--glob", "!*.pyc",
            digest, ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    own = str(HERE.relative_to(ROOT))
    return sorted(
        line.removeprefix("./")
        for line in result.stdout.splitlines()
        if line and own not in line
    )


def main() -> int:
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)

    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, path in CANDIDATES.items():
            authority_data = archive.read(f"task{task:03d}.onnx")
            candidate_data = path.read_bytes()
            authority_profile = official_profile(task, authority_data, "authority")
            candidate_profile = official_profile(task, candidate_data, "candidate")
            static = SWEEP.static_audit(candidate_data)
            configs = {
                label: AUDITOR.known_config(task, candidate_data, disable, threads)
                for disable, threads, label in AUDITOR.CONFIGS
            }
            known4 = all(item.get("perfect", False) for item in configs.values())
            try:
                trace = AUDITOR.direct_runtime_shape_trace(task, candidate_data) if known4 else None
                trace_error = None
            except BaseException as exc:
                trace = None
                trace_error = f"{type(exc).__name__}: {exc}"
            base_cost = int(authority_profile["cost"])
            cost = int(candidate_profile["cost"])
            lower = bool(candidate_profile.get("correct")) and 0 < cost < base_cost
            row = {
                "task": task,
                "source": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(candidate_data).hexdigest(),
                "serialized_bytes": len(candidate_data),
                "authority_sha256": hashlib.sha256(authority_data).hexdigest(),
                "authority_profile": authority_profile,
                "candidate_profile": candidate_profile,
                "strict_lower_correct": lower,
                "projected_gain": math.log(base_cost / cost) if lower else 0.0,
                "static": static,
                "known_four_configs": configs,
                "known_perfect_all_configs": known4,
                "runtime_shape_trace": trace,
                "runtime_shape_trace_error": trace_error,
            }
            rows.append(row)
            print(
                f"AUDIT task{task:03d} cost={cost}/{base_cost} lower={lower} "
                f"known4={known4} truthful={None if trace is None else trace.get('truthful')}",
                flush=True,
            )

    for row in rows:
        digest = row["sha256"]
        row["exact_sha_text_history"] = exact_history(digest)
        if not row["known_perfect_all_configs"]:
            row["fresh_not_run_reason"] = "known×4/runtime gate failed"
            continue
        data = CANDIDATES[row["task"]].read_bytes()
        candidate = [{"sha256": digest, "data": data, "sources": [row["source"]]}]
        fresh_runs = []
        for seed in SEEDS:
            print(f"FRESH task{row['task']:03d} seed={seed}", flush=True)
            fresh_runs.append(SCANNER.fresh_dual(row["task"], candidate, 500, seed))
        row["fresh_runs"] = fresh_runs

    output = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": got,
        "fresh_seeds": list(SEEDS),
        "fresh_count_per_seed": 500,
        "rows": rows,
    }
    (HERE / "audit" / "deep_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
