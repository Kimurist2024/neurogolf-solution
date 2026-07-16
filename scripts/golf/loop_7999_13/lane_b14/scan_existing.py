#!/usr/bin/env python3
"""Resume the exact-base headroom scan for task005/task080 in lane B14."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper


onnxruntime.set_default_logger_severity(4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/scratch_codex_plus10/wave1_b/scan_candidates.py"

spec = importlib.util.spec_from_file_location("b14_scanner", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

scanner.HERE = HERE
scanner.ROOT = ROOT
scanner.TASKS = (5, 80)
scanner.BASE_ZIP = ROOT / "submission_base_7999.13.zip"
scanner.POOL_ROOTS = tuple(
    ROOT / path
    for path in (
        "others/1/70205",
        "others/1/70208",
        "others/1/70209",
        "others/2/1200",
        "others/2/1201",
        "others/2/1202",
        "others/2/1203",
        "others/2/7901",
        "others/2/7902",
        "others/2/7903",
        "others/2/7904",
        "others/2/7905",
        "others/2/7906",
        "others/2/7907",
        "others/2/7908",
        "others/7902",
        "others/7903",
        "others/7905",
        "others/7906",
        "others/7907",
        "artifacts",
        "scripts/golf",
    )
)
scanner.TREE_ROOTS = ()

with zipfile.ZipFile(scanner.BASE_ZIP) as archive:
    BASE_HASHES = {
        hashlib.sha256(archive.read(f"task{task:03d}.onnx")).hexdigest()
        for task in scanner.TASKS
    }
KNOWN_BLACK_SHAS = {
    hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT / "artifacts/quarantine").glob("*.onnx")
}

original_collect = scanner.collect
original_static_check = scanner.static_check


def collect():
    base, candidates, errors = original_collect()
    for task in candidates:
        candidates[task] = {
            digest: item
            for digest, item in candidates[task].items()
            if digest not in KNOWN_BLACK_SHAS
        }
    return base, candidates, errors


def static_check(data: bytes):
    digest = hashlib.sha256(data).hexdigest()
    if digest in KNOWN_BLACK_SHAS:
        return False, "known_black_sha", None
    try:
        ok, reason, model = original_static_check(data)
    except Exception as exc:  # noqa: BLE001
        return False, f"static_exception:{type(exc).__name__}:{exc}", None
    if not ok or model is None:
        return ok, reason, model
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"strict_data_prop:{type(exc).__name__}:{exc}", None
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        return False, "custom_node_domain", None
    if any(
        node.op_type == "Einsum" and len(node.input) > 16
        for node in model.graph.node
    ):
        return False, "giant_einsum", None
    if any(
        item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
        for item in model.graph.initializer
    ):
        return False, "external_initializer", None
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            return False, "nonfinite_initializer", None
    return True, reason, model


scanner.collect = collect
scanner.static_check = static_check


def seed_from_headroom() -> None:
    target = HERE / "scan_results.json"
    if target.exists():
        return
    source = ROOT / "scripts/golf/loop_7999_13/lane_headroom/scan_results.json"
    if not source.exists():
        return
    old = json.loads(source.read_text())
    rows = [row for row in old.get("rows", []) if row.get("task") in scanner.TASKS]
    base = {
        key: value
        for key, value in old.get("base", {}).items()
        if int(key) in scanner.TASKS
    }
    seeded = {
        "base": base,
        "rows": rows,
        "winners": [],
        "aggregate_gain_before_fresh": 0.0,
        "seed_source": str(source.relative_to(ROOT)),
    }
    target.write_text(json.dumps(seeded, indent=2) + "\n")


if __name__ == "__main__":
    (HERE / "base").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    seed_from_headroom()
    raise SystemExit(scanner.main())
