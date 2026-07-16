#!/usr/bin/env python3
"""Screen all historical task175 latent-prune depths under POLICY90."""

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
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
SOURCE_DIR = ROOT / "scripts/golf/loop_8003_40/agent_exact_scanners/prune_latent"
THRESHOLD = 0.90


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = import_path(
    "task175_policy90_worker_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in (
            "total",
            "evaluated",
            "right",
            "wrong",
            "accuracy",
            "accuracy_is_upper_bound",
            "errors",
            "nonfinite_cases",
            "nonfinite_elements",
            "runtime_shape_mismatches",
            "small_positive_elements_0_to_0_25",
            "minimum_positive",
            "maximum_nonpositive",
            "early_reject_reason",
        )
    }


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    cases, counts = WORKER.SUPPORT.known_cases(175)
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task175.onnx")
    authority_model = onnx.load_model_from_string(authority_data)
    authority_profile = WORKER.POLICY.fast_profile(
        WORKER.SUPPORT, 175, authority_model, cases[0]
    )
    rows = []
    for depth in range(1, 9):
        path = SOURCE_DIR / f"task175_r{depth:03d}.onnx"
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        preflight = WORKER.quick_preflight(copy.deepcopy(model))
        profile = WORKER.POLICY.fast_profile(WORKER.SUPPORT, 175, model, cases[0])
        known = WORKER.failfast_known(data, cases) if not preflight else None
        structure = (
            WORKER.POLICY.structure_audit(WORKER.SUPPORT, 175, model, data)
            if not preflight and known is not None and known.get("early_reject_reason") is None
            and float(known.get("accuracy", 0.0)) >= THRESHOLD
            else None
        )
        admitted_screen = bool(
            not preflight
            and profile is not None
            and int(profile["cost"]) < int(authority_profile["cost"])
            and known is not None
            and known.get("early_reject_reason") is None
            and float(known.get("accuracy", 0.0)) >= THRESHOLD
            and known.get("errors") == 0
            and known.get("nonfinite_cases") == 0
            and known.get("runtime_shape_mismatches") == 0
            and known.get("small_positive_elements_0_to_0_25") == 0
            and structure is not None
            and structure.get("pass")
        )
        row = {
            "depth": depth,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "bytes": len(data),
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "preflight": preflight,
            "profile": profile,
            "known": compact(known) if known is not None else None,
            "structure": structure,
            "screen_pass": admitted_screen,
        }
        rows.append(row)
        print(json.dumps({
            "depth": depth,
            "cost": None if profile is None else profile["cost"],
            "known": None if known is None else known["accuracy"],
            "screen_pass": admitted_screen,
        }), flush=True)
    passing = [row for row in rows if row["screen_pass"]]
    passing.sort(key=lambda row: (int(row["profile"]["cost"]), -float(row["known"]["accuracy"])))
    result = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": AUTHORITY_SHA256,
            "task175_sha256": digest(authority_data),
            "profile": authority_profile,
        },
        "threshold": THRESHOLD,
        "known_counts": counts,
        "rows": rows,
        "best_screen": passing[0] if passing else None,
        "protected_writes": "lane only; root submission/all_scores/others untouched",
    }
    (HERE / "screen.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_screen": result["best_screen"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
