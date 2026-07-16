#!/usr/bin/env python3
"""Deduplicate and actual-score all C4 task candidates without promotion."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import onnxruntime

onnxruntime.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/scratch_codex_plus10/wave1_b/scan_candidates.py"

spec = importlib.util.spec_from_file_location("c4_scanner", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

scanner.HERE = HERE
scanner.ROOT = ROOT
scanner.TASKS = (12, 46, 66, 117, 165, 238, 270)
scanner.BASE_ZIP = ROOT / "submission_base_7999.13.zip"
scanner.POOL_ROOTS = tuple(
    ROOT / path
    for path in (
        "others/1/70205", "others/1/70208", "others/1/70209",
        "others/2/1200", "others/2/1201", "others/2/1202", "others/2/1203",
        "others/2/7901", "others/2/7902", "others/2/7903", "others/2/7904",
        "others/2/7905", "others/2/7906", "others/2/7907", "others/2/7908",
        "others/7902", "others/7903", "others/7905", "others/7906", "others/7907",
        "artifacts", "scripts/golf",
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
    except Exception as exc:
        return False, f"static_exception:{type(exc).__name__}:{exc}", None
    if not ok or model is None:
        return ok, reason, model
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            return False, f"custom_node_domain:{node.domain}", None
        if node.op_type == "Einsum" and len(node.input) >= 15 and digest not in BASE_HASHES:
            return False, f"giant_einsum:{len(node.input)}", None
    return True, reason, model


scanner.collect = collect
scanner.static_check = static_check


if __name__ == "__main__":
    (HERE / "base").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    raise SystemExit(scanner.main())
