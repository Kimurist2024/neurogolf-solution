#!/usr/bin/env python3
"""Reproduce why the tempting task223 sparse-initializer model is unsafe."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "artifacts/submission_sparse_test.zip"
OUTPUT = HERE / "sparse_task223_audit.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_support():
    spec = importlib.util.spec_from_file_location("sparse223_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    support = load_support()
    with zipfile.ZipFile(SOURCE) as archive:
        data = archive.read("task223.onnx")
    model = onnx.load_model_from_string(data)
    full_check_error = strict_error = official_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        full_check_error = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"
    official_profile = None
    try:
        with tempfile.TemporaryDirectory(prefix="sparse223_", dir="/tmp") as work:
            official_profile = scoring.score_and_verify(
                copy.deepcopy(model), 223, work, label="sparse223", require_correct=False
            )
    except Exception as exc:  # noqa: BLE001
        official_error = f"{type(exc).__name__}: {exc}"
    cases, counts = support.known_cases(223)
    known = support.evaluate_four(data, cases)
    payload = {
        "task": 223,
        "source": str(SOURCE.relative_to(ROOT)) + "::task223.onnx",
        "sha256": hashlib.sha256(data).hexdigest(),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "sparse_stored_values": sum(
            int(item.values.dims[0]) for item in model.graph.sparse_initializer
        ),
        "full_checker_pass": full_check_error is None,
        "full_checker_error": full_check_error,
        "strict_shape_pass": strict_error is None,
        "strict_shape_error": strict_error,
        "known_counts": counts,
        "known_four": known,
        "official_profile": official_profile,
        "official_scorer_error": official_error,
        "decision": "REJECT_OFFICIAL_SCORER_CRASH",
        "reason": (
            "calculate_memory iterates graph.sparse_initializer and reads init.name, "
            "but SparseTensorProto has no name field; this raises AttributeError and "
            "would be an error task even though ORT signs are exact"
        ),
        "fresh_not_run": "official scorer compatibility is a prior hard gate",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "sha256": payload["sha256"],
        "full_checker_pass": payload["full_checker_pass"],
        "strict_shape_pass": payload["strict_shape_pass"],
        "official_scorer_error": payload["official_scorer_error"],
        "known_right": {name: row["right"] for name, row in known.items()},
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
