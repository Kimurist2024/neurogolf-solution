#!/usr/bin/env python3
"""Fail-closed deep audit for the selected 71405 tail candidates.

Every invocation handles one candidate so an ORT abort cannot erase evidence
for the other candidates.  The script never writes outside this lane.
"""

from __future__ import annotations

import argparse
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
CANDIDATES = (
    ("task310_improved_1", 310, "others/71405/task310_improved(1).onnx"),
    ("task354_improved", 354, "others/71405/task354_improved.onnx"),
    ("task361_cost844", 361, "others/71405/task361_cost844.onnx"),
    ("task363_improved", 363, "others/71405/task363_improved.onnx"),
    ("task365_cost1355", 365, "others/71405/task365_cost1355.onnx"),
    ("task370_improved_1", 370, "others/71405/task370_improved(1).onnx"),
    ("task378_improved", 378, "others/71405/task378_improved.onnx"),
    ("task396_cost_reduced", 396, "others/71405/task396_cost_reduced.onnx"),
    ("task396_improved_v2", 396, "others/71405/task396_improved_v2.onnx"),
    ("task268_improved_cost420", 268, "others/71405/task268_improved_cost420.onnx"),
    ("task270_improved", 270, "others/71405/task270_improved.onnx"),
    ("task284_improved", 284, "others/71405/task284_improved.onnx"),
)
SEEDS = (202607149901, 202607149902)

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
    "tail99_known_four",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
SWEEP = AUDITOR.SWEEP
SCANNER = load_module(
    "tail99_fresh",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)


def official_profile(task: int, data: bytes, label: str) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"tail99_{task:03d}_", dir="/tmp") as workdir:
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


def run_candidate(label: str, task: int, source: str, fresh_count: int) -> dict:
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")
    path = ROOT / source
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read(f"task{task:03d}.onnx")

    authority_profile = official_profile(task, authority_data, "authority")
    candidate_profile = official_profile(task, data, "candidate")
    static = SWEEP.static_audit(data)
    configs = {
        config_label: AUDITOR.known_config(task, data, disable, threads)
        for disable, threads, config_label in AUDITOR.CONFIGS
    }
    known4 = all(item.get("perfect", False) for item in configs.values())
    trace = None
    trace_error = None
    if known4:
        try:
            trace = AUDITOR.direct_runtime_shape_trace(task, data)
        except BaseException as exc:  # fail closed; invocation is subprocess-isolated
            trace_error = f"{type(exc).__name__}: {exc}"

    base_cost = int(authority_profile["cost"])
    cost = int(candidate_profile["cost"])
    lower = bool(candidate_profile.get("correct")) and 0 < cost < base_cost
    row = {
        "label": label,
        "task": task,
        "source": source,
        "sha256": digest,
        "serialized_bytes": len(data),
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
        "exact_sha_text_history": exact_history(digest),
        "fresh_seeds": list(SEEDS),
        "fresh_count_per_seed": fresh_count,
    }
    if lower and known4:
        candidate = [{"sha256": digest, "data": data, "sources": [source]}]
        row["fresh_runs"] = [
            SCANNER.fresh_dual(task, candidate, fresh_count, seed) for seed in SEEDS
        ]
    else:
        row["fresh_not_run_reason"] = "strict-lower or known×4 gate failed"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=[item[0] for item in CANDIDATES], required=True)
    parser.add_argument("--fresh", type=int, default=500)
    args = parser.parse_args()
    label, task, source = next(item for item in CANDIDATES if item[0] == args.label)
    row = run_candidate(label, task, source, args.fresh)
    outdir = HERE / "audit" / "rows"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{label}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"DONE {label} task{task:03d} "
        f"cost={row['candidate_profile']['cost']}/{row['authority_profile']['cost']} "
        f"lower={row['strict_lower_correct']} known4={row['known_perfect_all_configs']} "
        f"truthful={None if row['runtime_shape_trace'] is None else row['runtime_shape_trace']['truthful']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
