#!/usr/bin/env python3
"""Enumerate lower-rank task175 r001 contractions and audit POLICY90 finalists."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
SOURCE = ROOT / "scripts/golf/loop_8003_40/agent_exact_scanners/prune_latent/task175_r001.onnx"
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
FRESH_SEEDS = (416_300_175, 416_400_175)
MARGIN_POWER = 12


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = import_path(
    "task175_rank_prune_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
WORKER.SUPPORT.FRESH_PER_SEED = FRESH_PER_SEED


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def subsets(size: int) -> list[tuple[int, ...]]:
    return [
        choice
        for width in range(1, size + 1)
        for choice in itertools.combinations(range(size), width)
    ]


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    index = next(i for i, item in enumerate(model.graph.initializer) if item.name == name)
    model.graph.initializer[index].CopyFrom(
        numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)
    )


def variant(m_indices: tuple[int, ...], n_indices: tuple[int, ...]) -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    replace(model, "G1", arrays["G1"][:, :, list(m_indices)])
    replace(
        model,
        "G2",
        arrays["G2"][list(m_indices), :, :][:, :, list(n_indices)],
    )
    replace(model, "K", arrays["K"][list(n_indices), :, :])
    replace(model, "C0", np.ldexp(arrays["C0"], MARGIN_POWER))
    return model


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
    cases, known_counts = WORKER.SUPPORT.known_cases(175)
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task175.onnx")
    authority_model = onnx.load_model_from_string(authority_data)
    authority_profile = WORKER.POLICY.fast_profile(
        WORKER.SUPPORT, 175, authority_model, cases[0]
    )

    rows = []
    survivor_payloads: dict[str, bytes] = {}
    for m_indices in subsets(3):
        for n_indices in subsets(3):
            if len(m_indices) == 3 and len(n_indices) == 3:
                continue
            model = variant(m_indices, n_indices)
            data = model.SerializeToString()
            sha = digest(data)
            preflight = WORKER.quick_preflight(copy.deepcopy(model))
            profile = WORKER.POLICY.fast_profile(WORKER.SUPPORT, 175, model, cases[0])
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
                "m_indices": list(m_indices),
                "n_indices": list(n_indices),
                "m_rank": len(m_indices),
                "n_rank": len(n_indices),
                "sha256": sha,
                "profile": profile,
                "preflight": preflight,
                "known_disable_threads1": known,
                "screen_pass": passed,
            }
            rows.append(row)
            if passed:
                survivor_payloads[sha] = data
                print(json.dumps({
                    "m": m_indices,
                    "n": n_indices,
                    "cost": profile["cost"],
                    "known": known["accuracy"],
                    "sha": sha[:12],
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
    for rank, row in enumerate(survivors[:8], 1):
        data = survivor_payloads[row["sha256"]]
        model = onnx.load_model_from_string(data)
        structure = WORKER.POLICY.structure_audit(WORKER.SUPPORT, 175, model, data)
        known_raw = WORKER.SUPPORT.evaluate_four(data, cases)
        known_pass = structure.get("pass") and all(runtime_pass(item) for item in known_raw.values())
        fresh = []
        if known_pass:
            for seed in FRESH_SEEDS:
                fresh_cases, generation = WORKER.SUPPORT.fresh_cases(175, seed, task_map)
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
            "pass": passed,
        }), flush=True)
        if passed:
            target = HERE / "candidates" / (
                f"task175_POLICY90_rankprune_cost{int(row['profile']['cost'])}_{row['sha256'][:12]}.onnx"
            )
            target.write_bytes(data)
            admission = {
                **audit,
                "saved_path": str(target.relative_to(ROOT)),
                "authority_cost": int(authority_profile["cost"]),
                "candidate_cost": int(row["profile"]["cost"]),
                "gain": math.log(int(authority_profile["cost"]) / int(row["profile"]["cost"])),
                "construction": (
                    "r001 analytic Einsum with matching M/N latent axes sliced and C0 "
                    "scaled by an exact power of two for margin; no parameters added"
                ),
            }
            break

    evidence = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": AUTHORITY_SHA256,
            "task175_sha256": digest(authority_data),
            "profile": authority_profile,
        },
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": digest(SOURCE.read_bytes())},
        "threshold": THRESHOLD,
        "fresh_per_seed": FRESH_PER_SEED,
        "fresh_seeds": list(FRESH_SEEDS),
        "margin_power_of_two": MARGIN_POWER,
        "known_counts": known_counts,
        "enumerated": len(rows),
        "screen_survivor_count": len(survivors),
        "rows": rows,
        "full_audits": full_audits,
        "admission": admission,
        "protected_writes": "lane only; root submission/all_scores/others untouched",
    }
    (HERE / "rank_prune_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"admission": admission}, indent=2))
    return 0 if admission is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
