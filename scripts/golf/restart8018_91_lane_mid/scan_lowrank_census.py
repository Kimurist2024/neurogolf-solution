#!/usr/bin/env python3
"""Exact-rational R>=2 Einsum initializer census for the 8018.91 mid band."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8018.91.zip"
AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def main() -> int:
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    path = ROOT / "scripts/golf/root_einsum_lowrank_factor_scan_271/scan.py"
    spec = importlib.util.spec_from_file_location("restart8018_mid_lowrank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        tasks = [int(row["task"].removeprefix("task")) for row in csv.DictReader(handle)
                 if 250 <= int(row["cost"]) <= 399
                 and int(row["task"].removeprefix("task")) not in EXCLUDED]
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            rows.append(module.scan_model(
                onnx.load_model_from_string(data), task=task,
                source_kind="authority8018_91", source_path=f"{AUTHORITY.name}::{name}",
                source_sha256=hashlib.sha256(data).hexdigest(),
            ))
    partitions = [part for row in rows for part in row["partition_rows"]]
    saving = [part for part in partitions if part["rge2_parameter_saving"]]
    structural = [part for part in partitions if part["structural_candidate"]]
    payload = {
        "authority": AUTHORITY.name, "authority_sha256": AUTHORITY_SHA256,
        "tasks": len(rows),
        "einsum_nodes": sum(row["einsum_nodes"] for row in rows),
        "einsum_constant_initializers": sum(row["einsum_constant_initializers"] for row in rows),
        "partitions": len(partitions),
        "rge2_parameter_saving": saving,
        "structural_candidates": structural,
        "rows": rows,
    }
    (HERE / "lowrank_census.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "tasks", "einsum_nodes", "einsum_constant_initializers", "partitions"
    )} | {"rge2_saving": len(saving), "structural": len(structural)}), flush=True)
    return 0 if not structural else 2


if __name__ == "__main__":
    raise SystemExit(main())
