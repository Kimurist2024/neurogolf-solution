#!/usr/bin/env python3
"""Re-audit historical task161 cost176/184/188 candidates under POLICY90."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
FRESH_SEEDS = (418_100_161, 418_200_161)
CANDIDATE_PATHS = (
    "scripts/golf/scratch_codex/task161/probe_tensor176.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor176_s2.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor184_cp6_micro_s2.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor184_cp6_s1.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor184_cp6_s3.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor184_cp6_s4.onnx",
    "scripts/golf/scratch_codex/task161/probe_tensor184_cp6_s5.onnx",
    "scripts/golf/scratch_codex/task161/candidate_rank1_188.onnx",
    "scripts/golf/scratch_codex/task161/candidate_slice_rank1_188.onnx",
    "scripts/golf/root_task161_margin_repair_279/candidates/task161_cost186_margin8.onnx",
)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = import_path(
    "task161_policy90_worker_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
WORKER.SUPPORT.FRESH_PER_SEED = FRESH_PER_SEED


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(rows: dict[str, dict]) -> dict[str, dict]:
    return {name: WORKER.compact_runtime(row) for name, row in rows.items()}


def runtime_pass(row: dict) -> bool:
    return bool(
        float(row.get("accuracy", 0.0)) >= THRESHOLD
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    cases, known_counts = WORKER.SUPPORT.known_cases(161)
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task161.onnx")
    authority_model = onnx.load_model_from_string(authority_data)
    authority_profile = WORKER.POLICY.fast_profile(
        WORKER.SUPPORT, 161, authority_model, cases[0]
    )

    rows = []
    payloads: dict[str, bytes] = {}
    for relative in CANDIDATE_PATHS:
        path = ROOT / relative
        data = path.read_bytes()
        sha = digest(data)
        model = onnx.load_model_from_string(data)
        preflight = WORKER.quick_preflight(copy.deepcopy(model))
        profile = WORKER.POLICY.fast_profile(WORKER.SUPPORT, 161, model, cases[0])
        known = WORKER.failfast_known(data, cases) if not preflight else None
        passed = bool(
            not preflight
            and profile is not None
            and int(profile["cost"]) < int(authority_profile["cost"])
            and known is not None
            and known.get("early_reject_reason") is None
            and runtime_pass(known)
        )
        row = {
            "source": relative,
            "sha256": sha,
            "bytes": len(data),
            "profile": profile,
            "preflight": preflight,
            "known_disable_threads1": known,
            "screen_pass": passed,
        }
        rows.append(row)
        if passed:
            payloads[sha] = data
        print(json.dumps({
            "source": path.name,
            "cost": None if profile is None else profile["cost"],
            "known": None if known is None else known["accuracy"],
            "reason": None if known is None else known.get("early_reject_reason"),
            "pass": passed,
        }), flush=True)

    survivors = [row for row in rows if row["screen_pass"]]
    survivors.sort(key=lambda row: (
        int(row["profile"]["cost"]),
        -float(row["known_disable_threads1"]["accuracy"]),
        row["sha256"],
    ))
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text(encoding="utf-8"))
    full_audits = []
    admission = None
    for rank, row in enumerate(survivors, 1):
        data = payloads[row["sha256"]]
        model = onnx.load_model_from_string(data)
        structure = WORKER.POLICY.structure_audit(WORKER.SUPPORT, 161, model, data)
        known_raw = WORKER.SUPPORT.evaluate_four(data, cases)
        known_pass = structure.get("pass") and all(runtime_pass(item) for item in known_raw.values())
        fresh = []
        if known_pass:
            for seed in FRESH_SEEDS:
                fresh_cases, generation = WORKER.SUPPORT.fresh_cases(161, seed, task_map)
                runtime_raw = WORKER.SUPPORT.evaluate_four(data, fresh_cases)
                fresh.append({
                    "seed": seed,
                    "generation": generation,
                    "runtime": compact(runtime_raw),
                    "pass": all(runtime_pass(item) for item in runtime_raw.values()),
                })
        passed = bool(known_pass and len(fresh) == 2 and all(item["pass"] for item in fresh))
        audit = {
            **row,
            "rank": rank,
            "structure": structure,
            "known_four": compact(known_raw),
            "known_four_pass": known_pass,
            "fresh": fresh,
            "policy90_pass": passed,
        }
        full_audits.append(audit)
        print(json.dumps({
            "rank": rank,
            "cost": row["profile"]["cost"],
            "known": {name: item["accuracy"] for name, item in known_raw.items()},
            "fresh": [item["runtime"]["disable_threads1"]["accuracy"] for item in fresh],
            "structure_pass": structure.get("pass"),
            "pass": passed,
        }), flush=True)
        if passed:
            target = HERE / "candidates" / (
                f"task161_POLICY90_cost{int(row['profile']['cost'])}_{row['sha256'][:12]}.onnx"
            )
            target.write_bytes(data)
            admission = {
                **audit,
                "saved_path": str(target.relative_to(ROOT)),
                "authority_cost": int(authority_profile["cost"]),
                "candidate_cost": int(row["profile"]["cost"]),
                "gain": math.log(int(authority_profile["cost"]) / int(row["profile"]["cost"])),
            }
            break

    evidence = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": AUTHORITY_SHA256,
            "task161_sha256": digest(authority_data),
            "profile": authority_profile,
        },
        "threshold": THRESHOLD,
        "fresh_per_seed": FRESH_PER_SEED,
        "fresh_seeds": list(FRESH_SEEDS),
        "known_counts": known_counts,
        "rows": rows,
        "full_audits": full_audits,
        "admission": admission,
        "protected_writes": "lane only; root submission/all_scores/others untouched",
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admission": admission}, indent=2))
    return 0 if admission is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
