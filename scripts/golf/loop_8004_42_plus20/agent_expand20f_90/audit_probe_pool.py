#!/usr/bin/env python3
"""Re-open giant/lookup/private candidates for LB-probe ranking.

These policy classes are not fixed-safe, but they are not discarded merely for
being giant/lookup/private. Actual-lower, known×4-complete, truthful candidates
are retained as LB_PROBE_REQUIRED.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    75, 392, 225, 218, 159, 185, 263, 370, 182, 330,
    361, 157, 280, 382, 201, 251, 12, 107, 131, 364,
)
BASE_ZIP = ROOT / "submission_base_8006.61.zip"
COSTS_PATH = HERE / "baseline_costs_8006_61.json"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
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
    "expand20f_probe_scanner",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
AUDIT = load_module(
    "expand20f_probe_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
SWEEP = load_module(
    "expand20f_probe_static",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)


def selected(row: dict[str, Any]) -> bool:
    reasons = row.get("reasons", [])
    if row.get("stage") == "structure_reject":
        return len(reasons) == 1 and reasons[0].startswith("giant_einsum:")
    if row.get("stage") == "policy_reject":
        return bool(reasons) and all(reason in {"lookup", "private_zero_lineage"} for reason in reasons)
    return False


def boolean_known_complete(configs: dict[str, dict[str, Any]]) -> bool:
    return all(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("runtime_errors") == 0
        and not row.get("session_error")
        for row in configs.values()
    )


def probe_structure_ok(static: dict[str, Any]) -> bool:
    allowed = {"lookup", "giant_einsum", "giant_initializer"}
    return (
        bool(static.get("full_check"))
        and bool(static.get("strict_data_prop"))
        and bool(static.get("all_node_outputs_static_positive"))
        and bool(static.get("standard_domains"))
        and bool(static.get("conv_bias_ub0"))
        and set(static.get("reasons", [])) <= allowed
    )


def main() -> int:
    rescreen = json.loads((HERE / "rescreen.json").read_text())
    costs = json.loads(COSTS_PATH.read_text())["costs"]
    pool = [row for row in rescreen["rows"] if selected(row)]
    wanted = {row["sha256"] for row in pool}
    SCANNER.HERE = HERE
    SCANNER.TARGETS = TARGETS
    SCANNER.BASE_ZIP = BASE_ZIP
    SCANNER.CURRENT_COSTS_JSON = COSTS_PATH
    inventory, inventory_report = SCANNER.inventory()
    data_by_sha = {
        digest: item["data"]
        for per_task in inventory.values()
        for digest, item in per_task.items()
        if digest in wanted
    }
    rows = []
    lower = 0
    known_jobs = 0
    for index, source in enumerate(sorted(pool, key=lambda row: (row["task"], row["sha256"])), 1):
        task = int(source["task"])
        digest = source["sha256"]
        data = data_by_sha[digest]
        try:
            profile = SWEEP.profiler_cost(data, task, f"probe_{index:03d}")
            profile_error = None
        except Exception as exc:  # noqa: BLE001
            profile = None
            profile_error = f"{type(exc).__name__}: {exc}"
        row: dict[str, Any] = {
            "task": task,
            "sha256": digest,
            "sources": source["sources"],
            "original_reasons": source["reasons"],
            "baseline_cost": int(costs[str(task)]),
            "profile": profile,
            "profile_error": profile_error,
        }
        actual_cost = None if profile is None else int(profile["cost"])
        row["actual_strict_lower"] = actual_cost is not None and 0 < actual_cost < row["baseline_cost"]
        if not row["actual_strict_lower"]:
            row.update(classification="HARD_REJECT", final_reason="actual_not_strict_lower_or_profile_error")
            rows.append(row)
            print(f"PROBE {index}/{len(pool)} task{task:03d} actual={actual_cost} lower=0", flush=True)
            continue
        lower += 1
        static = SWEEP.static_audit(data)
        configs = {
            label: AUDIT.known_config(task, data, disable, threads)
            for disable, threads, label in CONFIGS
        }
        known_jobs += 1
        boolean_complete = boolean_known_complete(configs)
        try:
            trace = AUDIT.direct_runtime_shape_trace(task, data) if boolean_complete else None
            trace_error = None
        except Exception as exc:  # noqa: BLE001
            trace = None
            trace_error = f"{type(exc).__name__}: {exc}"
        structure_ok = probe_structure_ok(static)
        truthful = bool((trace or {}).get("truthful"))
        probe_required = boolean_complete and structure_ok and truthful
        row.update(
            static=static,
            known_four_configs=configs,
            known_boolean_complete_all_configs=boolean_complete,
            runtime_shape_trace=trace,
            runtime_shape_trace_error=trace_error,
            probe_structure_ok=structure_ok,
            classification="LB_PROBE_REQUIRED" if probe_required else "HARD_REJECT",
            final_reason=(
                "known_x4_truthful_but_nonfixed_policy_class"
                if probe_required
                else "known_runtime_schema_or_truthful_shape_failure"
            ),
        )
        rows.append(row)
        print(
            f"PROBE {index}/{len(pool)} task{task:03d} actual={actual_cost} "
            f"known4={boolean_complete} truthful={truthful} class={row['classification']}",
            flush=True,
        )
    result = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "inventory_counts": inventory_report["counts"],
        "reopened_count": len(rows),
        "actual_strict_lower_count": lower,
        "known_x4_jobs": known_jobs,
        "lb_probe_required_count": sum(row["classification"] == "LB_PROBE_REQUIRED" for row in rows),
        "rows": rows,
    }
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "reopened_giant_lookup_private.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
