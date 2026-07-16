#!/usr/bin/env python3
"""Screen task132 scale variants against one cached arbitrary-grid corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "submission_base_8000.46.zip"
VARIANT_DIR = Path(__file__).resolve().parent / "lane_task132_scale"
VALIDATOR = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"


def load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("ngolf_validator_task132_screen", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load validator from {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_run(module: Any, session: Any, one_hot: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    try:
        return module.run_output(session, one_hot), None
    except Exception as exc:  # pragma: no cover - records candidate runtime failures
        return None, repr(exc)


def screen_one(
    module: Any,
    path: Path,
    inputs: list[np.ndarray],
    baseline_outputs: list[np.ndarray | None],
    baseline_errors: list[str | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "requested": len(inputs),
        "executable": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "mismatches": 0,
        "skipped_both_failed": 0,
        "skipped_one_failed": 0,
        "max_abs_difference": 0.0,
        "first_mismatch": None,
    }
    try:
        model = module.sanitize_model(onnx.load(path))
        if model is None:
            raise RuntimeError("candidate sanitization failed")
        session = module.make_session(model)
    except Exception as exc:
        row["session_error"] = repr(exc)
        return row

    for case_index, (one_hot, baseline_raw, baseline_error) in enumerate(
        zip(inputs, baseline_outputs, baseline_errors, strict=True)
    ):
        candidate_raw, candidate_error = safe_run(module, session, one_hot)
        if baseline_error is not None and candidate_error is not None:
            row["skipped_both_failed"] += 1
            continue
        if baseline_error is not None or candidate_error is not None:
            row["skipped_one_failed"] += 1
            if row["first_mismatch"] is None:
                row["first_mismatch"] = {
                    "case": case_index,
                    "baseline_error": baseline_error,
                    "candidate_error": candidate_error,
                }
            continue

        assert baseline_raw is not None and candidate_raw is not None
        row["executable"] += 1
        raw_equal = bool(np.array_equal(baseline_raw, candidate_raw, equal_nan=True))
        threshold_equal = bool(np.array_equal(baseline_raw > 0, candidate_raw > 0))
        row["raw_equal"] += int(raw_equal)
        row["threshold_equal"] += int(threshold_equal)
        if baseline_raw.shape == candidate_raw.shape:
            finite = np.isfinite(baseline_raw) & np.isfinite(candidate_raw)
            if finite.any():
                difference = float(np.max(np.abs(baseline_raw[finite] - candidate_raw[finite])))
                row["max_abs_difference"] = max(row["max_abs_difference"], difference)
        if not threshold_equal:
            row["mismatches"] += 1
            if row["first_mismatch"] is None:
                row["first_mismatch"] = {
                    "case": case_index,
                    "different_threshold_cells": int(
                        np.count_nonzero((baseline_raw > 0) != (candidate_raw > 0))
                    ),
                    "raw_equal": raw_equal,
                }
    row["strict_pass"] = (
        row["mismatches"] == 0
        and row["skipped_one_failed"] == 0
        and row["executable"] + row["skipped_both_failed"] == row["requested"]
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=80004601)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variant-dir", type=Path, default=VARIANT_DIR)
    parser.add_argument("--glob", default="task132_e*.onnx")
    parser.add_argument("--filter-report", type=Path)
    args = parser.parse_args()

    module = load_validator()
    with zipfile.ZipFile(BASELINE) as archive:
        baseline_model = module.sanitize_model(onnx.load_model_from_string(archive.read("task132.onnx")))
    if baseline_model is None:
        raise RuntimeError("baseline sanitization failed")
    baseline_session = module.make_session(baseline_model)

    rng = np.random.default_rng(args.seed)
    inputs: list[np.ndarray] = []
    baseline_outputs: list[np.ndarray | None] = []
    baseline_errors: list[str | None] = []
    for case_index in range(args.cases):
        grid = module._random_grid(rng, case_index % 4)
        one_hot = module.grid_to_one_hot(grid)
        baseline_raw, baseline_error = safe_run(module, baseline_session, one_hot)
        inputs.append(one_hot)
        baseline_outputs.append(baseline_raw)
        baseline_errors.append(baseline_error)

    variant_dir = args.variant_dir.resolve()
    paths = sorted(variant_dir.glob(args.glob))
    if args.filter_report is not None:
        prior = json.loads(args.filter_report.read_text())
        allowed = {
            (ROOT / row["path"]).resolve()
            for row in prior["results"]
            if row.get("strict_pass")
        }
        paths = [path for path in paths if path.resolve() in allowed]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(screen_one, module, path, inputs, baseline_outputs, baseline_errors): path
            for path in paths
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["path"])

    report = {
        "baseline": str(BASELINE.relative_to(ROOT)),
        "task": 132,
        "cases": args.cases,
        "seed": args.seed,
        "baseline_errors": sum(error is not None for error in baseline_errors),
        "candidate_count": len(results),
        "strict_pass_count": sum(bool(row.get("strict_pass")) for row in results),
        "results": results,
    }
    output = variant_dir / f"screen{args.cases}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "candidate_count": report["candidate_count"],
                "strict_pass_count": report["strict_pass_count"],
                "baseline_errors": report["baseline_errors"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
