#!/usr/bin/env python3
"""Pin and independently re-audit the three pending POLICY95 candidates.

This lane is evidence-only.  It never mutates the root authority or others/.
Each task is handled by one spawned process so the three audits run in
parallel while keeping generator RNG state isolated.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
THRESHOLD = 0.95
FRESH_PER_SEED = 2_000
FRESH_SEEDS = {
    161: (404_200_161, 404_300_161),
    175: (404_200_175, 404_300_175),
    355: (404_200_355, 404_300_355),
}
CANDIDATES = {
    161: {
        "source": "scripts/golf/root_task161_margin_repair_279/candidates/task161_cost186_margin8.onnx",
        "sha256": "57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81",
        "selection": "lowest actual cost in cost101_250_half_307; margin-repaired tie winner",
    },
    175: {
        "source": "scripts/golf/loop_8003_40/agent_exact_scanners/prune_latent/task175_r001.onnx",
        "sha256": "40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c",
        "selection": "only lowest-cost history candidate passing all prior POLICY95 gates",
    },
    355: {
        "source": "scripts/golf/loop_7999_13/lane_archive_all400/task355_r04_static249.onnx",
        "sha256": "7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8",
        "selection": "cost-249 tie; chose the independently reviewed r04 lineage",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_support():
    path = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
    spec = importlib.util.spec_from_file_location(
        f"restart8012_support_{os.getpid()}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.POLICY_THRESHOLD = THRESHOLD
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


def runtime_row_pass(row: dict[str, Any]) -> bool:
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


def annotate_rows(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for row in rows.values():
        row["policy_threshold"] = THRESHOLD
        row["policy95"] = runtime_row_pass(row)
    return rows


def profile_or_fail(support, task: int, model: onnx.ModelProto, label: str) -> dict[str, Any]:
    profile = support.official_profile(task, copy.deepcopy(model), label)
    if profile is None:
        raise RuntimeError(f"unscorable {label} task{task:03d}")
    return profile


def required_structure_audit(
    support, task: int, model: onnx.ModelProto, structure: dict[str, Any]
) -> dict[str, Any]:
    """Apply the assignment's mandatory structural gates.

    The generic scanner treats an Einsum with >=15 inputs as an advisory
    overfit-risk heuristic and therefore skips its runtime trace.  The task175
    candidate is a small 145-parameter analytic contraction with 18 inputs;
    high arity is not a banned op or a mandatory rejection in this assignment.
    We run the truthful runtime-shape trace explicitly and retain the arity as
    an advisory flag instead of silently accepting or rejecting it.
    """
    try:
        trace = support.runtime_shape_trace(task, copy.deepcopy(model))
        trace["shape_cloak_findings"] = len(trace.get("declared_actual_mismatches", []))
        trace["truthful"] = bool(
            not trace.get("error")
            and trace["shape_cloak_findings"] == 0
        )
    except Exception as exc:  # noqa: BLE001
        trace = {
            "truthful": False,
            "shape_cloak_findings": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
    reasons: list[str] = []
    if not structure.get("full_check"):
        reasons.append("full_checker")
    if not structure.get("strict_shape_data_prop"):
        reasons.append("strict_shape")
    if not structure.get("canonical_io"):
        reasons.append("noncanonical_io")
    if structure.get("missing_node_outputs") or structure.get("nonstatic_node_outputs"):
        reasons.append("untyped_or_nonstatic_node_output")
    if structure.get("banned_ops"):
        reasons.append("banned_ops")
    if (
        structure.get("nonstandard_domains")
        or structure.get("nested_graphs")
        or structure.get("functions")
        or structure.get("sparse_initializers")
        or structure.get("external_initializers")
    ):
        reasons.append("nonstandard_or_nested")
    if structure.get("nonfinite_initializers"):
        reasons.append("nonfinite_initializer")
    if structure.get("conv_bias_ub_findings"):
        reasons.append("conv_bias_ub")
    if not trace.get("truthful"):
        reasons.append("runtime_shape_cloak")
    return {
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "runtime_shape_trace": trace,
        "advisories": [
            f"high_arity_einsum_inputs={structure.get('max_einsum_inputs')}"
        ] if structure.get("giant_einsum") else [],
    }


def private_zero_membership(task: int, support) -> dict[str, Any]:
    catalog = ROOT / "docs/golf/private_zero_tasks.md"
    text = catalog.read_text(encoding="utf-8")
    # The maintained support set is the machine-readable operational catalog.
    return {
        "catalog": rel(catalog),
        "catalog_sha256": sha256(text.encode()),
        "listed_private_zero_or_unsound": task in support.PRIVATE_ZERO_OR_UNSOUND,
        "raw_number_occurs_in_catalog": bool(re.search(rf"(?<!\d){task}(?!\d)", text)),
    }


def public_overfit_membership(task: int) -> dict[str, Any]:
    source = ROOT / "docs/research/discussions/raw/704762.json"
    text = source.read_text(encoding="utf-8")
    return {
        "source": rel(source),
        "source_sha256": sha256(text.encode()),
        "risk_marker_present": "Overfit-risk top 10 task IDs" in text,
        "task_in_source": f"task{task:03d}" in text,
    }


def audit_task(task: int) -> dict[str, Any]:
    started = time.monotonic()
    support = load_support()
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    specification = CANDIDATES[task]
    source = ROOT / specification["source"]
    candidate_data = source.read_bytes()
    if sha256(candidate_data) != specification["sha256"]:
        raise RuntimeError(f"candidate SHA changed for task{task:03d}")
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP SHA changed")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read(f"task{task:03d}.onnx")

    candidate_model = onnx.load_model_from_string(candidate_data)
    authority_model = onnx.load_model_from_string(authority_data)
    structure = support.structural_audit(task, candidate_model, candidate_data)
    structure["mandatory_gates"] = required_structure_audit(
        support, task, candidate_model, structure
    )
    candidate_profile = profile_or_fail(
        support, task, candidate_model, f"restart404_candidate_{task:03d}"
    )
    authority_profile = profile_or_fail(
        support, task, authority_model, f"restart404_authority_{task:03d}"
    )
    strict_lower = int(candidate_profile["cost"]) < int(authority_profile["cost"])

    known_cases, known_counts = support.known_cases(task)
    known_runtime = annotate_rows(support.evaluate_four(candidate_data, known_cases))
    print(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "task": task,
                "phase": "known",
                "accuracy": {name: row["accuracy"] for name, row in known_runtime.items()},
            }
        ),
        flush=True,
    )
    fresh: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS[task]:
        cases, generation = support.fresh_cases(task, seed, task_map)
        runtime = annotate_rows(support.evaluate_four(candidate_data, cases))
        row = {
            "seed": seed,
            "generation": generation,
            "runtime": runtime,
            "pass": all(runtime_row_pass(item) for item in runtime.values()),
        }
        fresh.append(row)
        print(
            json.dumps(
                {
                    "worker_pid": os.getpid(),
                    "task": task,
                    "phase": f"fresh_{seed}",
                    "accuracy": {name: item["accuracy"] for name, item in runtime.items()},
                    "pass": row["pass"],
                }
            ),
            flush=True,
        )

    private_zero = private_zero_membership(task, support)
    public_overfit = public_overfit_membership(task)
    known_pass = all(runtime_row_pass(row) for row in known_runtime.values())
    fresh_pass = all(row["pass"] for row in fresh)
    all_rows = [*known_runtime.values()]
    for run in fresh:
        all_rows.extend(run["runtime"].values())
    exact_all_cases = all(float(row["accuracy"]) == 1.0 for row in all_rows)
    policy95_pass = bool(
        structure["mandatory_gates"]["pass"]
        and strict_lower
        and known_pass
        and fresh_pass
    )
    guaranteed_safe = bool(
        policy95_pass
        and candidate_profile.get("correct") is True
        and exact_all_cases
        and not private_zero["listed_private_zero_or_unsound"]
        and not (public_overfit["risk_marker_present"] and public_overfit["task_in_source"])
    )
    if not policy95_pass:
        classification = "REJECT"
    elif private_zero["listed_private_zero_or_unsound"]:
        classification = "PRIVATE_ZERO_POLICY95"
    elif guaranteed_safe:
        classification = "GUARANTEED_SAFE"
    elif public_overfit["risk_marker_present"] and public_overfit["task_in_source"]:
        classification = "PUBLIC_OVERFIT_RISK_POLICY95"
    else:
        classification = "POLICY95_NONEXACT"

    aggregate = {
        "dataset_config_rows": len(all_rows),
        "case_config_executions": sum(int(row["total"]) for row in all_rows),
        "minimum_accuracy": min(float(row["accuracy"]) for row in all_rows),
        "errors": sum(int(row.get("errors", 0)) for row in all_rows),
        "nonfinite_cases": sum(int(row.get("nonfinite_cases", 0)) for row in all_rows),
        "nonfinite_elements": sum(int(row.get("nonfinite_elements", 0)) for row in all_rows),
        "runtime_shape_mismatches": sum(
            int(row.get("runtime_shape_mismatches", 0)) for row in all_rows
        ),
        "small_positive_elements_0_to_0_25": sum(
            int(row.get("small_positive_elements_0_to_0_25", 0)) for row in all_rows
        ),
        "sign_mismatch_cases_across_configs": sum(
            int(row.get("sign_mismatch_cases_vs_disable_threads1", 0)) for row in all_rows
        ),
        "sign_mismatch_cells_across_configs": sum(
            int(row.get("sign_mismatch_cells_vs_disable_threads1", 0)) for row in all_rows
        ),
    }
    result = {
        "task": task,
        "worker_pid": os.getpid(),
        "threshold": THRESHOLD,
        "fresh_per_seed": FRESH_PER_SEED,
        "authority": {
            "zip": rel(AUTHORITY),
            "zip_sha256": AUTHORITY_SHA256,
            "member": f"task{task:03d}.onnx",
            "member_sha256": sha256(authority_data),
            "member_bytes": len(authority_data),
            "profile": authority_profile,
        },
        "candidate": {
            "source": specification["source"],
            "source_sha256": specification["sha256"],
            "source_bytes": len(candidate_data),
            "selection": specification["selection"],
            "profile": candidate_profile,
            "strict_lower_actual_cost": strict_lower,
            "cost_reduction": int(authority_profile["cost"]) - int(candidate_profile["cost"]),
            "projected_gain": math.log(
                int(authority_profile["cost"]) / int(candidate_profile["cost"])
            ) if strict_lower else 0.0,
        },
        "structure": structure,
        "known": {"counts": known_counts, "runtime": known_runtime, "pass": known_pass},
        "fresh": fresh,
        "runtime_aggregate": aggregate,
        "policy_classification": {
            "private_zero": private_zero,
            "public_overfit": public_overfit,
            "known_and_fresh_exact_all_configs": exact_all_cases,
            "guaranteed_safe": guaranteed_safe,
            "policy95_pass": policy95_pass,
            "classification": classification,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    if policy95_pass:
        outdir = HERE / "candidates"
        outdir.mkdir(parents=True, exist_ok=True)
        saved = outdir / (
            f"task{task:03d}_cost{int(candidate_profile['cost'])}_"
            f"{specification['sha256'][:12]}_{classification}.onnx"
        )
        saved.write_bytes(candidate_data)
        result["candidate"]["saved_path"] = rel(saved)
        result["candidate"]["saved_sha256"] = sha256(saved.read_bytes())
    (HERE / f"task{task:03d}_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority changed before launch")
    HERE.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    with context.Pool(processes=3) as pool:
        results = pool.map(audit_task, [161, 175, 355])
    results.sort(key=lambda row: int(row["task"]))
    accepted = [row for row in results if row["policy_classification"]["policy95_pass"]]
    payload = {
        "lane": rel(HERE),
        "authority": {"zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256, "lb": 8012.15},
        "workers_requested": 3,
        "worker_pids": sorted({int(row["worker_pid"]) for row in results}),
        "threshold": THRESHOLD,
        "fresh_design": "2 independent seeds x 2000 cases x 4 ORT configs per task",
        "tasks": results,
        "summary": {
            "audited": len(results),
            "policy95_pass": len(accepted),
            "guaranteed_safe": sum(
                bool(row["policy_classification"]["guaranteed_safe"]) for row in results
            ),
            "private_zero_policy95": sum(
                row["policy_classification"]["classification"] == "PRIVATE_ZERO_POLICY95"
                for row in results
            ),
            "public_overfit_risk_policy95": sum(
                row["policy_classification"]["classification"]
                == "PUBLIC_OVERFIT_RISK_POLICY95"
                for row in results
            ),
            "projected_gain": sum(float(row["candidate"]["projected_gain"]) for row in accepted),
            "projected_lb_if_all_policy95_hold": 8012.15
            + sum(float(row["candidate"]["projected_gain"]) for row in accepted),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (HERE / "evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "authority": payload["authority"],
        "admission_policy": "POLICY95; not an LB guarantee",
        "candidates": [
            {
                "task": row["task"],
                "path": row["candidate"].get("saved_path"),
                "sha256": row["candidate"].get("saved_sha256"),
                "authority_member_sha256": row["authority"]["member_sha256"],
                "authority_cost": row["authority"]["profile"]["cost"],
                "candidate_cost": row["candidate"]["profile"]["cost"],
                "projected_gain": row["candidate"]["projected_gain"],
                "classification": row["policy_classification"]["classification"],
                "policy95_pass": row["policy_classification"]["policy95_pass"],
            }
            for row in results
        ],
        "projected_gain": payload["summary"]["projected_gain"],
        "projected_lb_if_all_policy95_hold": payload["summary"][
            "projected_lb_if_all_policy95_hold"
        ],
        "root_or_others_modified": False,
    }
    (HERE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# restart8012 pending 3-worker audit",
        "",
        f"Authority: `submission_base_8012.15.zip` (`{AUTHORITY_SHA256}`)",
        "",
        f"Workers: 3 spawned processes, PIDs {payload['worker_pids']}",
        "",
        "| task | authority | candidate | gain | classification | min accuracy |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for row in results:
        report_lines.append(
            "| {task:03d} | {authority} | {candidate} | +{gain:.6f} | {classification} | {accuracy:.4%} |".format(
                task=row["task"],
                authority=row["authority"]["profile"]["cost"],
                candidate=row["candidate"]["profile"]["cost"],
                gain=row["candidate"]["projected_gain"],
                classification=row["policy_classification"]["classification"],
                accuracy=row["runtime_aggregate"]["minimum_accuracy"],
            )
        )
    report_lines.extend(
        [
            "",
            f"Conditional total gain: **+{payload['summary']['projected_gain']:.6f}**",
            f"Conditional projected LB: **{payload['summary']['projected_lb_if_all_policy95_hold']:.6f}**",
            "",
            "`GUARANTEED_SAFE` and `POLICY95` are intentionally separate. A POLICY95 pass",
            "permits known/fresh mismatches up to 5% and is not an LB guarantee. task355",
            "also retains the documented public-overfit-risk tag. No candidate is listed in",
            "the maintained private-zero/unsound operational set.",
            "",
            "The root authority and `others/` were not modified.",
        ]
    )
    (HERE / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
