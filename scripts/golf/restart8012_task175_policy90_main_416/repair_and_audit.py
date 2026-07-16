#!/usr/bin/env python3
"""Repair task175 r007's near-zero margin by exact sign-preserving scaling."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
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
SOURCE = ROOT / "scripts/golf/loop_8003_40/agent_exact_scanners/prune_latent/task175_r007.onnx"
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
FRESH_SEEDS = (416_100_175, 416_200_175)
POWERS = (0, 8, 12, 16, 20, 24)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = import_path(
    "task175_repair_worker_support",
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


def scaled_model(power: int) -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    index = next(i for i, item in enumerate(model.graph.initializer) if item.name == "C0")
    original = numpy_helper.to_array(model.graph.initializer[index]).astype(np.float32, copy=True)
    scaled = np.ldexp(original, power).astype(np.float32)
    replacement = numpy_helper.from_array(scaled, name="C0")
    model.graph.initializer[index].CopyFrom(replacement)
    return model


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

    screens = []
    finalists: list[tuple[int, onnx.ModelProto, bytes, dict, dict]] = []
    for power in POWERS:
        model = scaled_model(power)
        data = model.SerializeToString()
        preflight = WORKER.quick_preflight(copy.deepcopy(model))
        profile = WORKER.POLICY.fast_profile(WORKER.SUPPORT, 175, model, cases[0])
        structure = WORKER.POLICY.structure_audit(WORKER.SUPPORT, 175, model, data)
        known_raw = WORKER.SUPPORT.evaluate_four(data, cases)
        known = compact(known_raw)
        passed = bool(
            not preflight
            and profile is not None
            and int(profile["cost"]) < int(authority_profile["cost"])
            and structure.get("pass")
            and all(runtime_pass(row) for row in known_raw.values())
        )
        row = {
            "power_of_two": power,
            "scale": math.ldexp(1.0, power),
            "sha256": digest(data),
            "profile": profile,
            "preflight": preflight,
            "structure": structure,
            "known": known,
            "known_pass": passed,
        }
        screens.append(row)
        print(json.dumps({
            "power": power,
            "cost": None if profile is None else profile["cost"],
            "accuracy": {name: item["accuracy"] for name, item in known_raw.items()},
            "small": {name: item["small_positive_elements_0_to_0_25"] for name, item in known_raw.items()},
            "pass": passed,
        }), flush=True)
        if passed:
            finalists.append((power, model, data, profile, structure))

    fresh_audits = []
    admission = None
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text(encoding="utf-8"))
    for power, model, data, profile, structure in finalists:
        fresh = []
        for seed in FRESH_SEEDS:
            fresh_cases, generation = WORKER.SUPPORT.fresh_cases(175, seed, task_map)
            runtime_raw = WORKER.SUPPORT.evaluate_four(data, fresh_cases)
            fresh.append({
                "seed": seed,
                "generation": generation,
                "runtime": compact(runtime_raw),
                "pass": all(runtime_pass(row) for row in runtime_raw.values()),
            })
        passed = len(fresh) == 2 and all(row["pass"] for row in fresh)
        audit = {
            "power_of_two": power,
            "scale": math.ldexp(1.0, power),
            "sha256": digest(data),
            "profile": profile,
            "structure": structure,
            "fresh": fresh,
            "fresh_pass": passed,
        }
        fresh_audits.append(audit)
        print(json.dumps({
            "power": power,
            "fresh": [row["runtime"]["disable_threads1"]["accuracy"] for row in fresh],
            "pass": passed,
        }), flush=True)
        if passed:
            target = HERE / "candidates" / f"task175_POLICY90_cost{int(profile['cost'])}_{digest(data)[:12]}.onnx"
            target.write_bytes(data)
            admission = {
                **audit,
                "saved_path": str(target.relative_to(ROOT)),
                "authority_cost": int(authority_profile["cost"]),
                "candidate_cost": int(profile["cost"]),
                "gain": math.log(int(authority_profile["cost"]) / int(profile["cost"])),
                "construction": (
                    "historical r007 analytic Einsum with C0 multiplied by an exact power of two; "
                    "the transformation increases output margin without adding parameters"
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
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": digest(SOURCE.read_bytes()),
        },
        "threshold": THRESHOLD,
        "fresh_per_seed": FRESH_PER_SEED,
        "known_counts": known_counts,
        "screens": screens,
        "fresh_audits": fresh_audits,
        "admission": admission,
        "protected_writes": "lane only; root submission/all_scores/others untouched",
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admission": admission}, indent=2))
    return 0 if admission is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
