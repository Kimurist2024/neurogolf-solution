#!/usr/bin/env python3
"""Competition-profile rescreen of all delta SHAs.

``rank_dir.cost_of`` is a useful truthful-runtime diagnostic, but the campaign
authority includes declared/runtime shape discrepancies.  The comparison that
decides strict-lower must therefore use the same ``score_and_verify`` profile
as the competition tooling.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (
    239, 222, 37, 226, 297, 14, 234, 92, 397, 264,
    394, 398, 200, 75, 392, 387, 225, 218, 36, 132,
)

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


SCANNER = load_module(
    "expand20j_official_inventory",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
AUDITOR = load_module(
    "expand20j_official_known_four",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
SCANNER.HERE = HERE
SCANNER.TARGETS = TARGETS
SCANNER.BASE_ZIP = AUTHORITY


def official_profile(task: int, data: bytes, label: str) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"expand20j_{task:03d}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def main() -> int:
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")
    delta = json.loads((HERE / "inventory_delta.json").read_text())
    truth = json.loads((HERE / "audit" / "delta_official_known4.json").read_text())
    truth_by_sha = {row["sha256"]: row for row in truth["rows"]}
    wanted = {row["sha256"]: row for row in delta["delta"]}
    inventory, inventory_report = SCANNER.inventory()
    data_by_sha = {
        digest: item["data"]
        for task_rows in inventory.values()
        for digest, item in task_rows.items()
        if digest in wanted
    }

    authority: dict[int, dict] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TARGETS:
            data = archive.read(f"task{task:03d}.onnx")
            authority[task] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "official_full_profile": official_profile(task, data, "authority"),
            }
            print(
                f"OFFICIAL BASE task{task:03d} cost={authority[task]['official_full_profile']['cost']}",
                flush=True,
            )

    rows: list[dict] = []
    for index, (digest, source) in enumerate(
        sorted(wanted.items(), key=lambda item: (item[1]["task"], item[0])), 1
    ):
        task = int(source["task"])
        data = data_by_sha[digest]
        try:
            profile = official_profile(task, data, f"candidate_{digest[:10]}")
            error = None
        except BaseException as exc:
            profile = None
            error = f"{type(exc).__name__}: {exc}"
        base_cost = int(authority[task]["official_full_profile"]["cost"])
        cost = int(profile["cost"]) if profile and profile.get("cost") is not None else None
        correct = bool(profile and profile.get("correct"))
        lower = correct and cost is not None and 0 < cost < base_cost
        row = {
            **source,
            "authority_sha256": authority[task]["sha256"],
            "authority_official_full_profile": authority[task]["official_full_profile"],
            "candidate_official_full_profile": profile,
            "official_profile_error": error,
            "official_correct_strict_lower": lower,
            "projected_gain": math.log(base_cost / cost) if lower else 0.0,
            "truthful_runtime_diagnostic": {
                "profile": truth_by_sha[digest].get("official_profile"),
                "static": truth_by_sha[digest].get("static"),
                "runtime_shape_trace": truth_by_sha[digest].get("runtime_shape_trace"),
                "known_four_configs": truth_by_sha[digest].get("known_four_configs"),
            },
        }
        static_ok = bool((row["truthful_runtime_diagnostic"]["static"] or {}).get("pre_runtime_structural_pass"))
        if lower and static_ok:
            existing = row["truthful_runtime_diagnostic"].get("known_four_configs")
            if existing is None:
                existing = {
                    label: AUDITOR.known_config(task, data, disable, threads)
                    for disable, threads, label in AUDITOR.CONFIGS
                }
            row["known_four_configs"] = existing
            row["known_perfect_all_configs"] = all(item.get("perfect", False) for item in existing.values())
            if row["known_perfect_all_configs"]:
                try:
                    row["runtime_shape_trace"] = AUDITOR.direct_runtime_shape_trace(task, data)
                except BaseException as exc:
                    row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(
            f"OFFICIAL {index}/{len(wanted)} task{task:03d} cost={cost}/{base_cost} "
            f"correct={correct} lower={lower} known4={row.get('known_perfect_all_configs')}",
            flush=True,
        )

    output = {
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": got,
        "targets": list(TARGETS),
        "inventory_counts": inventory_report["counts"],
        "delta_count": len(rows),
        "official_correct_strict_lower_count": sum(row["official_correct_strict_lower"] for row in rows),
        "known_four_complete_count": sum(row.get("known_perfect_all_configs", False) for row in rows),
        "truthful_count": sum(bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in rows),
        "authority_tasks": {str(task): authority[task] for task in TARGETS},
        "rows": rows,
    }
    (HERE / "audit" / "official_rescreen.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
