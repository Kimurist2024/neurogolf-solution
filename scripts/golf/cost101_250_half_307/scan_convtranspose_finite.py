#!/usr/bin/env python3
"""Cross-apply finite rebuilds of the four nonfinite cost<=10 ConvTranspose nets."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
sys.path.insert(0, str(HERE))
import scan_patterns as common  # noqa: E402


def main() -> None:
    started = time.monotonic()
    costs = {}
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            task, cost = int(row["task"][4:]), int(row["cost"])
            if 101 <= cost <= 250:
                costs[task] = cost
    known = {task: common.SCAN.cases(task) for task in costs}
    values = (0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0)
    variants = []
    seen = set()
    with zipfile.ZipFile(AUTHORITY) as archive:
        for source_task in (53, 135, 326, 373):
            base = onnx.load_model_from_string(archive.read(f"task{source_task:03d}.onnx"))
            for initializer in base.graph.initializer:
                dense = onnx.numpy_helper.to_array(initializer)
                bad = list(zip(*np.where(~np.isfinite(dense))))
                if not bad:
                    continue
                for replacements in itertools.product(values, repeat=len(bad)):
                    model = copy.deepcopy(base)
                    array = dense.copy()
                    for index, value in zip(bad, replacements):
                        array[index] = value
                    model.graph.initializer[0].CopyFrom(
                        onnx.numpy_helper.from_array(np.ascontiguousarray(array), initializer.name)
                    )
                    data = model.SerializeToString()
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    safe, reasons = common.finite_structure(model)
                    runtime = common.SCAN.session(model) if safe else None
                    if runtime is not None:
                        variants.append({
                            "name": f"task{source_task:03d}_finite_" + "_".join(f"{x:g}" for x in replacements),
                            "source_task": source_task, "replacements": replacements,
                            "sha256": digest, "model": model, "data": data, "runtime": runtime,
                        })
    report = {"variant_count": len(variants), "tasks": [], "finalists": []}
    outdir = HERE / "convtranspose_finite_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    for task in sorted(costs):
        row = {"task": task, "authority_cost": costs[task], "half_limit": costs[task] // 2,
               "best_quick": {"right": -1}, "survivors": []}
        for source in variants:
            quick = common.SCAN.evaluate(source["runtime"], known[task], 4)
            if quick["right"] > row["best_quick"]["right"]:
                row["best_quick"] = {"name": source["name"], **quick}
            if not common.SCAN.clean(quick):
                continue
            full = common.SCAN.evaluate(source["runtime"], known[task], None)
            if not common.SCAN.clean(full):
                continue
            try:
                profile = common.SCAN.scoring.score_and_verify(
                    source["model"], task, str(HERE / "convtranspose_finite_profiles"),
                    source["name"], require_correct=True,
                )
            except Exception:
                profile = None
            if profile is None or int(profile["cost"]) > row["half_limit"]:
                continue
            stable, margin = common.SCAN.scoring.model_margin_stable(source["model"], task)
            if not stable:
                continue
            path = outdir / f"task{task:03d}_{source['name']}_cost{profile['cost']}.onnx"
            path.write_bytes(source["data"])
            item = {"task": task, "name": source["name"], "source_task": source["source_task"],
                    "sha256": source["sha256"], "path": str(path.relative_to(ROOT)),
                    "profile": profile, "known": full, "margin_min": margin}
            row["survivors"].append(item)
            report["finalists"].append(item)
        report["tasks"].append(row)
        print(json.dumps({"task": task, "best": row["best_quick"],
                          "survivors": len(row["survivors"])}), flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    (HERE / "convtranspose_finite_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variant_count": len(variants), "finalists": report["finalists"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
