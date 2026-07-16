#!/usr/bin/env python3
"""Coordinate isolated history audits and assemble manifests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def crash_row(index: int, lead: dict[str, Any], authority_cost: int, process: subprocess.CompletedProcess[str] | None, error: str) -> dict[str, Any]:
    return {
        "audit_index": index,
        "task": int(lead["task"]),
        "path": lead["candidate_path"],
        "sha256": lead["sha256"],
        "sources": lead["sources"],
        "source_count": lead["source_count"],
        "authority_cost": authority_cost,
        "static_cost_floor": lead.get("static_cost_floor"),
        "actual_cost": None,
        "actual_strict_lower": False,
        "gain_if_valid": 0.0,
        "exact_computational_graph_equivalent": bool(
            lead.get("exact_computational_graph_equivalent")
        ),
        "known_complete_four_configs": {},
        "known_four_complete_pass": False,
        "structure": {"pass": False, "worker_error": error},
        "risk_classification": ["runtime_process_crash"],
        "decision": "REJECT_RUNTIME_PROCESS_CRASH",
        "worker": {
            "returncode": process.returncode if process else None,
            "stdout_tail": process.stdout[-2000:] if process else "",
            "stderr_tail": process.stderr[-4000:] if process else "",
        },
    }


def main() -> int:
    inventory = json.loads((HERE / "inventory.json").read_text())
    tasks = [int(task) for task in inventory["targets"]]
    authority_costs = {
        int(row["task"]): int(row["authority_cost"])
        for row in inventory["summary"]
    }
    leads = [
        row
        for task in tasks
        for row in inventory["rows_by_task"][str(task)]
        if row.get("could_be_actual_strict_lower") and row.get("candidate_path")
    ]
    worker_dir = HERE / "evidence/workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    partial_path = HERE / "audit_partial.json"
    if partial_path.exists():
        previous = json.loads(partial_path.read_text()).get("rows", [])
        for index, row in enumerate(previous, start=1):
            if index > len(leads) or row.get("sha256") != leads[index - 1]["sha256"]:
                break
            row["audit_index"] = index
            existing.append(row)
    rows = list(existing)
    print(f"reusing {len(existing)} completed in-process audits", flush=True)
    for index in range(len(existing) + 1, len(leads) + 1):
        lead = leads[index - 1]
        output = worker_dir / f"worker_{index:03d}_{lead['sha256'][:12]}.json"
        if output.exists():
            try:
                row = json.loads(output.read_text())
                if row.get("sha256") != lead["sha256"]:
                    raise ValueError("worker SHA mismatch")
                rows.append(row)
                print(
                    f"[{index}/{len(leads)}] cached task{row['task']:03d} {row['decision']}",
                    flush=True,
                )
                continue
            except Exception:
                output.unlink()
        command = [
            sys.executable,
            str(HERE / "audit_worker.py"),
            "--index",
            str(index),
            "--output",
            str(output),
        ]
        process: subprocess.CompletedProcess[str] | None = None
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=300,
            )
            if process.returncode == 0 and output.exists():
                row = json.loads(output.read_text())
            else:
                row = crash_row(
                    index,
                    lead,
                    authority_costs[int(lead["task"])],
                    process,
                    f"worker exited {process.returncode}",
                )
        except subprocess.TimeoutExpired as exc:
            row = crash_row(
                index,
                lead,
                authority_costs[int(lead["task"])],
                process,
                f"worker timeout after {exc.timeout}s",
            )
        except Exception as exc:  # noqa: BLE001
            row = crash_row(
                index,
                lead,
                authority_costs[int(lead["task"])],
                process,
                f"coordinator:{type(exc).__name__}:{exc}",
            )
        rows.append(row)
        print(
            f"[{index}/{len(leads)}] task{row['task']:03d} "
            f"cost={(row.get('actual_cost') or {}).get('cost')}/{row['authority_cost']} "
            f"{row['decision']}",
            flush=True,
        )
        if index % 5 == 0:
            partial_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    partial_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    probes = [row for row in rows if row["decision"] == "LB_PROBE_REQUIRED"]
    exact = [row for row in rows if row["decision"] == "EXACT_FIXED_CANDIDATE"]
    result = {
        "authority": inventory["authority"],
        "authority_sha256": inventory["authority_sha256"],
        "targets": tasks,
        "history_inventory": inventory["inventory"],
        "leads_profiled": len(rows),
        "rows": rows,
        "exact_fixed_candidates": exact,
        "lb_probe_required": probes,
        "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "black12_net_specific": [18, 48, 112, 134, 168, 198, 233, 251, 277, 286, 365, 366],
        "fixed7": [13, 70, 158, 254, 267, 323, 379],
        "policy_notes": {
            "black12_is_net_specific_not_task_permanent": True,
            "giant_lookup_private_nonfinite_fresh_below_90_not_probe_rejects": True,
            "non_exact_requires_lb_probe": True,
            "local_fresh_is_ranking_only": True,
            "runtime_process_crash_is_reject": True,
        },
        "protected_files_modified": [],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(
        json.dumps(
            {
                "authority": inventory["authority"],
                "authority_sha256": inventory["authority_sha256"],
                "status": "LB_PROBE_REQUIRED",
                "candidates": [
                    {
                        "task": row["task"],
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "authority_cost": row["authority_cost"],
                        "candidate_cost": row["actual_cost"]["cost"],
                        "projected_gain": row["gain_if_valid"],
                        "risk_classification": row["risk_classification"],
                        "sources": row["sources"],
                    }
                    for row in probes
                ],
                "merge_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "authority": inventory["authority"],
                "authority_sha256": inventory["authority_sha256"],
                "winners": [
                    {
                        "task": row["task"],
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "authority_cost": row["authority_cost"],
                        "candidate_cost": row["actual_cost"]["cost"],
                        "gain": math.log(row["authority_cost"] / row["actual_cost"]["cost"]),
                        "proof": "strong graph equivalence after metadata/unused-init normalization",
                    }
                    for row in exact
                ],
                "merge_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result["decision_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
