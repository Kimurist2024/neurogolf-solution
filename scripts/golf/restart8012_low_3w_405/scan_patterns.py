#!/usr/bin/env python3
"""Worker 2: transplant finite cost<=10 structures into cost 101..166 tasks."""

from __future__ import annotations

import csv
import json
import tempfile
import time
import zipfile
from collections import Counter

import onnx

import common


OUT = common.HERE / "pattern_scan.json"


def main() -> int:
    started = time.monotonic()
    common.HERE.mkdir(parents=True, exist_ok=True)
    common.CANDIDATES.mkdir(parents=True, exist_ok=True)
    common.validate_authority()
    p = common.PATTERN
    costs = common.current_costs(101, 166)
    cases = {task: p.known_cases(task) for task in costs}
    report = {
        "authority": str(common.AUTHORITY.relative_to(common.ROOT)),
        "authority_sha256": common.AUTHORITY_SHA256,
        "authority_diff": common.authority_diff(),
        "scope": "all 8012.15 authority tasks at cost 101..166; <=100 inherited from lane 401 because every member is byte-identical",
        "scope_task_count": len(costs), "templates": [], "rejected_templates": [],
        "tasks": [], "finalists": [], "counters": {},
    }
    counters: Counter[str] = Counter()
    with zipfile.ZipFile(common.AUTHORITY) as archive:
        low = []
        # Template references include the four score-25 members at cost 0/1;
        # the optimization target census excludes score-25 tasks, but the
        # user's instruction explicitly asks us to learn from *all* cost<=10.
        low_costs = {}
        with (common.ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
            for csv_row in csv.DictReader(handle):
                cost = int(csv_row["cost"])
                if cost <= 10:
                    low_costs[int(csv_row["task"].removeprefix("task"))] = cost
        for task, cost in sorted(low_costs.items()):
            model = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            row = p.source(model, f"authority_task{task:03d}_cost{cost}", "literal_cost_le10", "8012.15 member")
            row["source_task"] = task
            row["source_cost"] = cost
            low.append(row)
        low = p.unique_sources(low)
        admitted = []
        for row in low:
            audit = p.structure(row["_model"])
            compact = common.compact_source(row)
            compact["structure"] = audit
            if audit["pass"]:
                admitted.append(row)
                report["templates"].append(compact)
            else:
                report["rejected_templates"].append(compact)
        generic = []
        for legacy in p.TEMPLATES.generic_variants():
            row = p.source(legacy["_model"], legacy["name"], legacy["family"], legacy["proof"])
            if p.structure(row["_model"])["pass"]:
                generic.append(row)
        generic = p.unique_sources([*admitted, *generic])
        generic_runtime = [(row, p.make_session(row["_model"])) for row in generic]

        for task in sorted(costs, key=lambda value: (costs[value], value)):
            base = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            specific = []
            synth = p.color_gather(task, cases[task])
            if synth is not None:
                specific.append(synth)
            specific.extend(p.einsum_initializer_subsets(task, base))
            specific.extend(p.conv_crop_variants(task, base))
            specific = p.unique_sources(specific)
            variants = [*generic_runtime, *((row, p.make_session(row["_model"])) for row in specific)]
            task_row = {
                "task": task, "authority_cost": costs[task], "half_target": costs[task] // 2,
                "known_count": len(cases[task]), "variant_count": len(variants),
                "quick_exact": 0, "known_exact": 0, "strict_lower": 0,
                "half_cost": 0, "best_quick": {"right": -1}, "survivors": [],
            }
            for row, runtime in variants:
                counters["candidate_task_evaluations"] += 1
                if runtime is None:
                    counters["session_reject"] += 1
                    continue
                quick = p.evaluate(runtime, cases[task], min(12, len(cases[task])))
                if quick["right"] > task_row["best_quick"]["right"]:
                    task_row["best_quick"] = {"name": row["name"], "family": row["family"], **quick}
                if not p.clean(quick):
                    continue
                task_row["quick_exact"] += 1
                full = p.evaluate(runtime, cases[task])
                if not p.clean(full):
                    continue
                task_row["known_exact"] += 1
                audit = p.structure(row["_model"])
                if not audit["pass"]:
                    counters["structural_reject"] += 1
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix=f"low405_pattern_{task:03d}_", dir="/tmp") as work:
                        profile = p.scoring.score_and_verify(row["_model"], task, work, label="candidate", require_correct=True)
                except Exception:
                    profile = None
                if profile is None or int(profile["cost"]) >= costs[task]:
                    continue
                stable, margin = p.scoring.model_margin_stable(row["_model"], task)
                if not stable:
                    counters["margin_reject"] += 1
                    continue
                candidate_cost = int(profile["cost"])
                half = candidate_cost * 2 <= costs[task]
                path = common.CANDIDATES / f"task{task:03d}_pattern_cost{candidate_cost}_{row['sha256'][:12]}.onnx"
                path.write_bytes(row["_data"])
                finalist = {
                    "task": task, "name": row["name"], "family": row["family"],
                    "detail": row["detail"], "sha256": row["sha256"],
                    "authority_cost": costs[task], "candidate_cost": candidate_cost,
                    "half_target_met": half, "known": full, "structure": audit,
                    "margin_stable": bool(stable), "margin_min": margin,
                    "path": str(path.relative_to(common.ROOT)),
                }
                task_row["strict_lower"] += 1
                task_row["half_cost"] += int(half)
                task_row["survivors"].append(finalist)
                report["finalists"].append(finalist)
            report["tasks"].append(task_row)
            OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "task": task, "cost": costs[task], "variants": len(variants),
                "best_right": task_row["best_quick"]["right"],
                "strict": task_row["strict_lower"], "half": task_row["half_cost"],
            }), flush=True)
    counters["lowcost_unique"] = len(report["templates"]) + len(report["rejected_templates"])
    counters["lowcost_admitted"] = len(report["templates"])
    counters["lowcost_rejected"] = len(report["rejected_templates"])
    report["counters"] = dict(counters)
    report["elapsed_seconds"] = time.monotonic() - started
    report["protected_writes"] = "lane only; authority/root/others untouched"
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finalists": len(report["finalists"]), "counters": report["counters"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
