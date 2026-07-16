#!/usr/bin/env python3
"""Official-profile and known×4 audit of incremental SHA candidates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (
    239, 222, 37, 226, 297, 14, 234, 92, 397, 264,
    394, 398, 200, 75, 392, 387, 225, 218, 36, 132,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "expand20j_screen_inventory",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
AUDITOR = load_module(
    "expand20j_known_four",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
SWEEP = AUDITOR.SWEEP
SCANNER.HERE = HERE
SCANNER.TARGETS = TARGETS
SCANNER.BASE_ZIP = AUTHORITY


def main() -> int:
    delta_report = json.loads((HERE / "inventory_delta.json").read_text())
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256 or delta_report["authority_sha256"] != got:
        raise SystemExit(f"authority drift: {got}")

    wanted = {row["sha256"]: row for row in delta_report["delta"]}
    candidates, inventory_report = SCANNER.inventory()
    data_by_sha = {
        digest: item["data"]
        for task_rows in candidates.values()
        for digest, item in task_rows.items()
        if digest in wanted
    }
    if set(data_by_sha) != set(wanted):
        missing = sorted(set(wanted) - set(data_by_sha))
        raise SystemExit(f"inventory changed; missing {missing}")

    authority: dict[int, dict] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TARGETS:
            data = archive.read(f"task{task:03d}.onnx")
            authority[task] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "profile": SWEEP.profiler_cost(data, task, "authority"),
            }
            print(f"BASE task{task:03d} cost={authority[task]['profile']['cost']}", flush=True)

    rows: list[dict] = []
    for index, (digest, source) in enumerate(
        sorted(wanted.items(), key=lambda item: (item[1]["task"], item[0])), 1
    ):
        task = int(source["task"])
        data = data_by_sha[digest]
        static = SWEEP.static_audit(data)
        try:
            profile = SWEEP.profiler_cost(data, task, f"candidate_{digest[:12]}")
            profile_error = None
        except BaseException as exc:  # fail closed even on native runtime exits reported as exceptions
            profile = None
            profile_error = f"{type(exc).__name__}: {exc}"
        base_cost = int(authority[task]["profile"]["cost"])
        actual_cost = None if profile is None else int(profile["cost"])
        lower = actual_cost is not None and 0 < actual_cost < base_cost
        gain = math.log(base_cost / actual_cost) if lower else 0.0
        row = {
            **source,
            "authority_sha256": authority[task]["sha256"],
            "authority_profile": authority[task]["profile"],
            "static": static,
            "official_profile": profile,
            "official_profile_error": profile_error,
            "strictly_lower": lower,
            "projected_gain": gain,
        }
        if lower and static.get("pre_runtime_structural_pass", False):
            configs = {
                label: AUDITOR.known_config(task, data, disable, threads)
                for disable, threads, label in AUDITOR.CONFIGS
            }
            known4 = all(item.get("perfect", False) for item in configs.values())
            row["known_four_configs"] = configs
            row["known_perfect_all_configs"] = known4
            if known4:
                try:
                    row["runtime_shape_trace"] = AUDITOR.direct_runtime_shape_trace(task, data)
                except BaseException as exc:
                    row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(
            f"AUDIT {index}/{len(wanted)} task{task:03d} cost={actual_cost}/{base_cost} "
            f"static={static.get('pre_runtime_structural_pass')} "
            f"known4={row.get('known_perfect_all_configs')}",
            flush=True,
        )

    output = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": got,
        "targets": list(TARGETS),
        "inventory_counts": inventory_report["counts"],
        "delta_count": len(rows),
        "strict_lower_count": sum(row["strictly_lower"] for row in rows),
        "structural_strict_lower_count": sum(
            row["strictly_lower"] and row["static"].get("pre_runtime_structural_pass", False)
            for row in rows
        ),
        "known_four_complete_count": sum(row.get("known_perfect_all_configs", False) for row in rows),
        "truthful_count": sum(
            bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in rows
        ),
        "authority_tasks": {str(task): authority[task] for task in TARGETS},
        "rows": rows,
    }
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "delta_official_known4.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
