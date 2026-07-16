#!/usr/bin/env python3
"""Deduplicated, fail-closed archive rescreen against current 8009.46 members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high136_archive_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high136_archive_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official_profile(task: int, data: bytes, label: str) -> dict[str, object]:
    try:
        model = onnx.load_model_from_string(data)
        with tempfile.TemporaryDirectory(prefix=f"high136_archive_{task:03d}_", dir="/tmp") as workdir:
            result = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label=label, require_correct=False
            )
        return result or {"memory": -1, "params": -1, "cost": -1, "correct": False, "error": "None"}
    except Exception as exc:  # noqa: BLE001
        return {"memory": -1, "params": -1, "cost": -1, "correct": False, "error": f"{type(exc).__name__}: {exc}"}


def candidates() -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {216: [], 285: [], 226: []}
    prior = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound216_255_103/inventory.json").read_text()
    )
    for row in prior["tasks"]["216"]["historical_static_below_current"]:
        for raw in row["sample_sources"]:
            if "::" not in raw and raw.endswith(".onnx"):
                path = ROOT / raw
                if path.is_file():
                    out[216].append(path)
    roots285 = (
        ROOT / "scripts/golf/loop_7999_13/lane_a21",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound285_105",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_deep68",
        ROOT / "scripts/golf/scratch/task285",
    )
    for root in roots285:
        if root.exists():
            out[285].extend(path for path in root.rglob("*.onnx") if "task285" in path.name)
    roots226 = (
        ROOT / "scripts/golf/scratch_codex/task226",
        ROOT / "scripts/golf/scratch_kimi/task226",
        ROOT / "scripts/golf/scratch_codex_plus10/wave1_c/zip_candidates",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_target_mid19",
        ROOT / "scripts/golf/loop_8003_40",
        ROOT / "scripts/golf/loop_8000_46",
    )
    for root in roots226:
        if root.exists():
            out[226].extend(path for path in root.rglob("*.onnx") if "task226" in path.name)
    return out


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    authority = json.loads((HERE / "authority_audit.json").read_text())
    report: dict[str, object] = {
        "authority_sha256": AUTHORITY_SHA256, "tasks": {}, "stage_survivors": []
    }
    for task, paths in candidates().items():
        baseline = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        baseline_sha = digest(baseline)
        baseline_cost = authority["tasks"][str(task)]["official_profile"]["cost"]
        unique: dict[str, tuple[bytes, list[str]]] = {}
        for path in paths:
            try:
                data = path.read_bytes()
                onnx.load_model_from_string(data)
            except Exception:  # noqa: BLE001
                continue
            sha = digest(data)
            if sha == baseline_sha:
                continue
            rel = str(path.relative_to(ROOT))
            if sha in unique:
                unique[sha][1].append(rel)
            else:
                unique[sha] = (data, [rel])
        rows = []
        for index, (sha, (data, sources)) in enumerate(sorted(unique.items())):
            model = onnx.load_model_from_string(data)
            static = SCAN.structural(copy.deepcopy(model))
            try:
                declared = SCAN.official_cost(data, f"high136_archive_declared_{task:03d}_{index}")
            except Exception as exc:  # noqa: BLE001
                declared = {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}
            row: dict[str, object] = {
                "task": task, "sha256": sha, "sources": sources,
                "declared_profile": declared, "structural": static,
                "official_profile": None, "strict_lower": False,
                "runtime_shape_trace": None, "known_four_configs": {},
                "authority_raw_equal": False, "accepted": False,
                "stage": "DECLARED_NOT_LOWER_OR_UNSCORABLE",
            }
            if declared["cost"] >= 0 and declared["cost"] < baseline_cost:
                profile = official_profile(task, data, f"high136_archive_{task:03d}_{index}")
                row["official_profile"] = profile
                row["strict_lower"] = profile["cost"] >= 0 and profile["cost"] < baseline_cost
                row["stage"] = "OFFICIAL_NOT_LOWER_OR_UNSCORABLE"
                if row["strict_lower"]:
                    try:
                        trace = AUDIT.direct_trace(task, data)
                    except Exception as exc:  # noqa: BLE001
                        trace = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
                    row["runtime_shape_trace"] = trace
                    row["stage"] = "STRUCTURE_OR_SHAPE_REJECT"
                    if static.get("pass", False) and trace.get("truthful", False):
                        known = {
                            label: AUDIT.known_config(task, baseline, data, disable, threads)
                            for disable, threads, label in CONFIGS
                        }
                        row["known_four_configs"] = known
                        row["authority_raw_equal"] = all(item.get("perfect", False) for item in known.values())
                        row["stage"] = "KNOWN_OR_RAW_REJECT"
                        if row["authority_raw_equal"]:
                            row["stage"] = "FRESH_REQUIRED"
                            report["stage_survivors"].append(row)
            rows.append(row)
            print(
                f"task{task:03d} {index+1}/{len(unique)} sha={sha[:12]} "
                f"declared={declared['cost']} official={row['official_profile'] and row['official_profile']['cost']} "
                f"stage={row['stage']}", flush=True,
            )
        report["tasks"][str(task)] = {
            "baseline_sha256": baseline_sha,
            "baseline_cost": baseline_cost,
            "unique_nonauthority": len(unique),
            "rows": rows,
        }
    (HERE / "archive_rescreen.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"stage_survivors={len(report['stage_survivors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
