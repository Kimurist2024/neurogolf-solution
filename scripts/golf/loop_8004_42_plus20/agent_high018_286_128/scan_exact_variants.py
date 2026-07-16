#!/usr/bin/env python3
"""Generate only semantics-preserving local rewrites from current task286."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "current/task286.onnx"
CANDIDATES = HERE / "candidates"
PROBES = HERE / "exact_probes"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high128_exact_scan_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    base_data = BASE.read_bytes()
    base = onnx.load_model_from_string(base_data)
    base_profile = SCAN.official_cost(base_data, "high128_task286_base")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    PROBES.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "task": 286,
        "baseline_sha256": digest(base_data),
        "baseline_profile": base_profile,
        "variants": [],
        "strict_lower": [],
    }
    seen = {digest(base_data)}
    transforms = [
        ("alias_initializers", SCAN.transform_alias_initializers),
        ("dead_unused", SCAN.transform_dead),
        ("common_subexpression", SCAN.transform_cse),
        ("optional_outputs", SCAN.transform_optional),
        ("manual_noops", SCAN.transform_manual_noops),
        ("scalar_broadcast", SCAN.transform_scalar_broadcast),
    ]
    generated = []
    for label, transform in transforms:
        try:
            candidate = transform(copy.deepcopy(base))
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: transform error {type(exc).__name__}: {exc}", flush=True)
            candidate = None
        if candidate is not None:
            generated.append((label, candidate))
    for label, candidate in generated:
        data = candidate.SerializeToString()
        sha = digest(data)
        if sha in seen:
            continue
        seen.add(sha)
        static = SCAN.structural(copy.deepcopy(candidate))
        row: dict[str, object] = {
            "label": label,
            "sha256": sha,
            "serialized_bytes": len(data),
            "structural": static,
        }
        if static.get("pass"):
            path = PROBES / f"task286_{label}_{sha[:12]}.onnx"
            path.write_bytes(data)
            row["path"] = str(path.relative_to(ROOT))
            row["status"] = "needs_isolated_cost_profile"
        report["variants"].append(row)
        print(
            f"{label}: structural={static.get('pass')} "
            f"path={row.get('path')} ",
            flush=True,
        )
        del candidate
        gc.collect()
    (HERE / "exact_variant_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"strict_lower={len(report['strict_lower'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
