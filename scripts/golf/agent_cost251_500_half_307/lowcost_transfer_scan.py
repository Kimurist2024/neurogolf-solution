#!/usr/bin/env python3
"""Transfer every authority cost<=10 and score-25 generic structure to 251..500.

This is a cheap broad screen.  Exact-known and >=95%-known survivors are
officially reprofiled; fresh validation is deliberately a separate admission
gate so this file cannot silently label a fitted formula safe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUTHORITY = ROOT / "submission_base_8011.05.zip"

import sys
sys.path.insert(0, str(ROOT))
from scripts.golf.score25_similarity_le100_304 import scan_score25 as base  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def main() -> None:
    started = time.monotonic()
    costs: dict[int, int] = {}
    all_costs: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            all_costs[task] = cost
            if 251 <= cost <= 500:
                costs[task] = cost

    known = {task: base.cases(task) for task in costs}
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_low = []
        for task, cost in sorted(all_costs.items()):
            if cost > 10:
                continue
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_from_string(data)
            authority_low.append({
                "name": f"authority_cost{cost}_task{task:03d}",
                "family": "authority_cost_le10_transfer",
                "proof": "exact byte model from immutable 8011.05 authority",
                "_model": model,
                "_data": data,
            })

        generic = base.dedupe([*base.TEMPLATES.generic_variants(), *authority_low])
        generic_sessions = [(source, base.session(source["_model"])) for source in generic]
        report = {
            "authority": str(AUTHORITY.relative_to(ROOT)),
            "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
            "scope": sorted(costs),
            "generic_variant_count": len(generic),
            "results": [],
            "finalists": [],
        }

        for task in sorted(costs, key=lambda value: (costs[value], value)):
            incumbent = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            specific = base.dedupe(base.safe_specific_variants(task, incumbent))
            variants = generic_sessions + [
                (source, base.session(source["_model"])) for source in specific
            ]
            row = {
                "task": task,
                "authority_cost": costs[task],
                "known_count": len(known[task]),
                "variant_count": len(variants),
                "best_quick": {"right": -1},
                "full_survivors": [],
            }
            for source, runtime in variants:
                if runtime is None:
                    continue
                quick = base.evaluate(runtime, known[task], 20)
                if quick["right"] > row["best_quick"]["right"]:
                    row["best_quick"] = {
                        "name": source["name"], "family": source["family"], **quick,
                    }
                # Preserve every plausible POLICY95 lead.  A 90% quick threshold
                # avoids rejecting a true 95% formula due to one unlucky case.
                if quick["errors"] or quick["nonfinite_cases"] or quick["small_margin_cases"]:
                    continue
                if quick["total"] and quick["right"] / quick["total"] < 0.90:
                    continue
                full = base.evaluate(runtime, known[task], None)
                if full["errors"] or full["nonfinite_cases"] or full["small_margin_cases"]:
                    continue
                accuracy = full["right"] / full["total"] if full["total"] else 0.0
                if accuracy < 0.95:
                    continue
                candidate = source["_model"]
                try:
                    profile = scoring.score_and_verify(
                        candidate, task, str(HERE / "profiles"), source["name"],
                        require_correct=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    profile = None
                    profile_error = f"{type(exc).__name__}: {exc}"
                else:
                    profile_error = None
                summary = {
                    "name": source["name"],
                    "family": source["family"],
                    "sha256": hashlib.sha256(source["_data"]).hexdigest(),
                    "quick": quick,
                    "full": full,
                    "known_accuracy": accuracy,
                    "official_profile": profile,
                    "profile_error": profile_error,
                }
                if profile is None or int(profile["cost"]) * 2 > costs[task]:
                    continue
                path = HERE / "candidates" / (
                    f"task{task:03d}_{source['name']}_cost{profile['cost']}.onnx"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source["_data"])
                summary["path"] = str(path.relative_to(ROOT))
                row["full_survivors"].append(summary)
                report["finalists"].append({"task": task, **summary})
            report["results"].append(row)
            print(json.dumps({
                "task": task,
                "cost": costs[task],
                "variants": len(variants),
                "best": row["best_quick"],
                "finalists": len(row["full_survivors"]),
            }), flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    (HERE / "lowcost_transfer_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "finalist_count": len(report["finalists"]),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
