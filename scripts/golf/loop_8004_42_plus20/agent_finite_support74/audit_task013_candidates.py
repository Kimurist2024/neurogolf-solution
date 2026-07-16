#!/usr/bin/env python3
"""Audit eight gain-ranked task013 files without promoting any artifact."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract"
NAMES = (
    "task013_r001.onnx",
    "task013_r002.onnx",
    "task013_r003.onnx",
    "task013_r004.onnx",
    "task013_r005.onnx",
    "task013_r006.onnx",
    "task013_r009.onnx",
    "task013_r010.onnx",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ONE = load_module("finite_support_task013_one", HERE / "audit_task013_one.py")
SWEEP = load_module(
    "finite_support_wave30b",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        base_data = archive.read("task013.onnx")
    base = onnx.load_from_string(base_data)
    baseline_profile = SWEEP.profiler_cost(base_data, 13, "baseline_8005_17")

    rows = []
    for name in NAMES:
        path = SOURCE / name
        data = path.read_bytes()
        model = onnx.load_from_string(data)
        proof = ONE.exact_rewrite_proof(base, model)
        static = SWEEP.static_audit(data)
        profile = SWEEP.profiler_cost(data, 13, path.stem)
        sessions = {
            label: ONE.make_session(data, disable, threads)
            for disable, threads, label in ONE.CONFIGS
        }
        known = ONE.run_known(13, sessions)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(data),
                "profile": profile,
                "strictly_lower": profile["cost"] < baseline_profile["cost"],
                "static": static,
                "exact_rewrite_proof": proof,
                "known_four_configs": known,
                "known_perfect_all_configs": all(row["perfect"] for row in known.values()),
            }
        )
        print(name, profile["cost"], rows[-1]["known_perfect_all_configs"], flush=True)

    payload = {
        "baseline_zip": "submission_base_8005.17.zip",
        "baseline_zip_sha256": digest((ROOT / "submission_base_8005.17.zip").read_bytes()),
        "baseline_task013_sha256": digest(base_data),
        "baseline_profile": baseline_profile,
        "candidate_count": len(rows),
        "rows": rows,
    }
    (HERE / "candidate_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
