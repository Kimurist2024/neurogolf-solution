#!/usr/bin/env python3
"""Apply every finite cost<=10 authority pattern plus generic one-node forms.

This is a cheap exact prescreen across all current authority-cost 101..250 tasks.
Only models at <=half actual cost, exact on every known case, finite, static and
margin-clean are saved for the later fresh gate.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
OUT = HERE / "pattern_evidence.json"


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = import_path("pattern307_scan", ROOT / "scripts/golf/score25_similarity_le100_304/scan_score25.py")
TEMPLATES = SCAN.TEMPLATES


def finite_structure(model: onnx.ModelProto) -> tuple[bool, list[str]]:
    reasons = []
    if model.functions or model.graph.sparse_initializer:
        reasons.append("function_or_sparse")
    for tensor in model.graph.initializer:
        try:
            if not np.all(np.isfinite(onnx.numpy_helper.to_array(tensor))):
                reasons.append("nonfinite_initializer")
        except Exception:
            reasons.append("bad_initializer")
    for node in model.graph.node:
        if node.op_type in {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}:
            reasons.append(f"banned:{node.op_type}")
        if "Sequence" in node.op_type:
            reasons.append(f"banned:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                reasons.append("nested_graph")
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
            for dim in value.type.tensor_type.shape.dim:
                if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                    reasons.append("nonstatic_shape")
                    break
    except Exception as exc:
        reasons.append(f"checker:{type(exc).__name__}")
    return not reasons, sorted(set(reasons))


def main() -> None:
    started = time.monotonic()
    costs = {}
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if 101 <= cost <= 250 and float(row["score"]) < 25:
                costs[task] = cost
    known = {task: SCAN.cases(task) for task in costs}
    with zipfile.ZipFile(AUTHORITY) as archive:
        sources = list(TEMPLATES.generic_variants())
        reference_tasks = []
        for task in range(1, 401):
            row_cost = None
            with (ROOT / "all_scores.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row["task"][4:]) == task:
                        row_cost = int(row["cost"])
                        break
            if row_cost is None or row_cost > 10:
                continue
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            safe, reasons = finite_structure(model)
            reference_tasks.append({"task": task, "cost": row_cost, "safe": safe, "reasons": reasons})
            if safe:
                sources.append({
                    "name": f"authority_cost_le10_task{task:03d}",
                    "family": "authority_finite_cost_le10_pattern",
                    "proof": "exact finite pattern from immutable 8011.05 authority",
                    "_model": model, "_data": data,
                })
        sources = SCAN.dedupe(sources)
        vetted = []
        source_rejects = []
        for source in sources:
            safe, reasons = finite_structure(source["_model"])
            runtime = SCAN.session(source["_model"]) if safe else None
            if runtime is None:
                source_rejects.append({"name": source["name"], "reasons": reasons or ["session"]})
            else:
                vetted.append((source, runtime))
        report = {
            "authority": AUTHORITY.name,
            "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
            "scope": sorted(costs), "cost_le10_references": reference_tasks,
            "generic_and_reference_count": len(sources), "vetted_source_count": len(vetted),
            "source_rejects": source_rejects, "task_rows": [], "finalists": [],
        }
        outdir = HERE / "pattern_candidates"
        outdir.mkdir(parents=True, exist_ok=True)
        for task in sorted(costs, key=lambda value: (costs[value], value)):
            base = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            specific = SCAN.dedupe(SCAN.safe_specific_variants(task, base))
            variants = list(vetted)
            for source in specific:
                safe, reasons = finite_structure(source["_model"])
                runtime = SCAN.session(source["_model"]) if safe else None
                if runtime is not None:
                    variants.append((source, runtime))
            row = {
                "task": task, "authority_cost": costs[task], "half_limit": costs[task] // 2,
                "known_count": len(known[task]), "variant_count": len(variants),
                "best_quick": {"right": -1}, "survivors": [],
            }
            for source, runtime in variants:
                quick = SCAN.evaluate(runtime, known[task], 4)
                if quick["right"] > row["best_quick"]["right"]:
                    row["best_quick"] = {"name": source["name"], "family": source["family"], **quick}
                if not SCAN.clean(quick):
                    continue
                full = SCAN.evaluate(runtime, known[task], None)
                if not SCAN.clean(full):
                    continue
                try:
                    profile = SCAN.scoring.score_and_verify(
                        source["_model"], task, str(HERE / "pattern_profiles"),
                        source["name"], require_correct=True,
                    )
                except Exception:
                    profile = None
                summary = {
                    "name": source["name"], "family": source["family"],
                    "sha256": hashlib.sha256(source["_data"]).hexdigest(),
                    "full_known": full, "profile": profile,
                }
                if profile is None or int(profile["cost"]) > row["half_limit"]:
                    continue
                stable, margin = SCAN.scoring.model_margin_stable(source["_model"], task)
                summary.update({"margin_stable": stable, "margin_min": margin})
                if not stable:
                    continue
                path = outdir / f"task{task:03d}_{source['name']}_cost{profile['cost']}.onnx"
                path.write_bytes(source["_data"])
                summary["path"] = str(path.relative_to(ROOT))
                row["survivors"].append(summary)
                report["finalists"].append({"task": task, "authority_cost": costs[task], **summary})
            report["task_rows"].append(row)
            print(json.dumps({"task": task, "cost": costs[task], "variants": len(variants),
                              "best": row["best_quick"], "survivors": len(row["survivors"])}), flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"finalists": report["finalists"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
