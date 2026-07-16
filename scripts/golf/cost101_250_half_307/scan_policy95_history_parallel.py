#!/usr/bin/env python3
"""Parallel POLICY95 screen for every structurally safe strict-history row.

Rows are grouped by task before dispatch so that each worker loads the known
fixtures and generator support for a task only once.  This is evidence-only:
it writes candidates and JSON inside this lane and never edits an authority
archive or score ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx

import scan_policy95_history as base


def compact_item(item: dict) -> dict:
    """Return a JSON/pickle-friendly row without transient model bytes."""
    return {key: value for key, value in item.items() if key != "data"}


def evaluate_known_fail_fast(support, runtime, cases: list[dict]) -> dict:
    """Evaluate fully only while POLICY95 remains mathematically reachable."""
    total = len(cases)
    max_failures = total - math.ceil(base.THRESHOLD * total)
    right = wrong = errors = 0
    nonfinite_cases = nonfinite_elements = shape_mismatches = small_positive = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    sign_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    first_wrong = first_error = first_shape = None
    started = time.monotonic()
    early_reason = None
    evaluated = 0
    for index, example in enumerate(cases):
        evaluated += 1
        benchmark = support.scoring.convert_to_numpy(example)
        if benchmark is None:
            errors += 1
            first_error = {"case": index, "error": "case became unconvertible"}
            early_reason = "conversion_error"
            break
        expected = benchmark["output"] > 0
        try:
            raw = np.asarray(runtime.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:
            errors += 1
            first_error = {"case": index, "error": f"{type(exc).__name__}: {exc}"}
            early_reason = "runtime_error"
            break
        if tuple(raw.shape) != (1, 10, 30, 30):
            shape_mismatches += 1
            first_shape = {"case": index, "shape": list(raw.shape)}
            early_reason = "runtime_shape_mismatch"
            break
        finite = np.isfinite(raw)
        bad = int(np.count_nonzero(~finite))
        nonfinite_cases += int(bad > 0)
        nonfinite_elements += bad
        if bad:
            early_reason = "nonfinite"
            break
        positive = raw > 0
        packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
        sign_digest.update(packed)
        raw_digest.update(np.ascontiguousarray(raw).tobytes())
        correct = bool(np.array_equal(positive, expected))
        right += int(correct)
        wrong += int(not correct)
        if not correct and first_wrong is None:
            first_wrong = {
                "case": index,
                "different_cells": int(np.count_nonzero(positive != expected)),
            }
        count_small = int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        small_positive += count_small
        if count_small:
            early_reason = "small_positive"
            break
        if np.any(positive):
            minimum_positive = min(minimum_positive, float(raw[positive].min()))
        nonpositive = finite & ~positive
        if np.any(nonpositive):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[nonpositive].max()))
        if wrong > max_failures:
            early_reason = "accuracy_upper_bound_below_95"
            break
    if early_reason is None:
        accuracy = right / total
        right_reported = right
    else:
        # Treat every unvisited case as right.  This is the most favorable
        # possible bound, so a threshold rejection remains fail-closed.
        right_reported = right + (total - evaluated)
        accuracy = right_reported / total
    return {
        "total": total,
        "evaluated": evaluated,
        "right": right_reported,
        "observed_right": right,
        "wrong": wrong,
        "accuracy": accuracy,
        "accuracy_is_upper_bound": early_reason is not None,
        "early_reject_reason": early_reason,
        "errors": errors,
        "nonfinite_cases": nonfinite_cases,
        "nonfinite_elements": nonfinite_elements,
        "runtime_shape_mismatches": shape_mismatches,
        "small_positive_elements_0_to_0_25": small_positive,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "sign_mismatch_cases_vs_disable_threads1": 0,
        "sign_mismatch_cells_vs_disable_threads1": 0,
        "sign_sha256": sign_digest.hexdigest(),
        "raw_sha256": raw_digest.hexdigest(),
        "first_wrong": first_wrong,
        "first_error": first_error,
        "first_shape_mismatch": first_shape,
        "elapsed_seconds": time.monotonic() - started,
    }


def screen_approx_chunk(indexed_rows: list[tuple[int, dict]]) -> list[tuple[int, dict, bytes | None]]:
    support = base.load_support()
    support.POLICY_THRESHOLD = base.THRESHOLD
    known_cache: dict[int, tuple[list, dict]] = {}
    results: list[tuple[int, dict, bytes | None]] = []
    for index, row in indexed_rows:
        task = int(row["task"])
        item = dict(row)
        data: bytes | None = None
        try:
            data = base.resolve(str(row["source"]))
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("sha mismatch")
            runtime = support.make_session(data, True, 1)
            if task not in known_cache:
                known_cache[task] = support.known_cases(task)
            cases, counts = known_cache[task]
            known_row = evaluate_known_fail_fast(support, runtime, cases)
            item["known_counts"] = counts
            item["known_disable_threads1"] = base.compact(known_row)
            for key in (
                "evaluated",
                "observed_right",
                "accuracy_is_upper_bound",
                "early_reject_reason",
                "first_shape_mismatch",
                "elapsed_seconds",
            ):
                item["known_disable_threads1"][key] = known_row.get(key)
        except Exception as exc:
            item["known_disable_threads1"] = {
                "total": 0,
                "right": 0,
                "accuracy": 0.0,
                "errors": 1,
                "session_error": f"{type(exc).__name__}: {exc}",
            }
            known_row = item["known_disable_threads1"]
            data = None
        item["known_policy95"] = base.row_pass(known_row)
        eligible_data = None
        if item["known_policy95"] and data is not None:
            model = onnx.load_model_from_string(data)
            cases, _ = known_cache[task]
            profile = base.fast_profile(support, task, model, cases[0])
            item["profile"] = profile
            if profile and int(profile["cost"]) < int(row["authority_cost"]):
                structure = base.structure_audit(support, task, model, data)
                item["structure"] = structure
                if structure["pass"]:
                    eligible_data = data
        print(
            json.dumps(
                {
                    "worker": os.getpid(),
                    "i": index,
                    "task": task,
                    "accuracy": known_row.get("accuracy"),
                    "eligible": eligible_data is not None,
                }
            ),
            flush=True,
        )
        results.append((index, compact_item(item), eligible_data))
    return results


def audit_task(args: tuple[int, list[tuple[dict, bytes]], str]) -> dict:
    task, rows, outdir_text = args
    support = base.load_support()
    support.POLICY_THRESHOLD = base.THRESHOLD
    task_map = json.loads((base.ROOT / "docs/golf/task_hash_map.json").read_text())
    known_cases, _ = support.known_cases(task)
    fresh_results = []
    finalist = None
    rows.sort(key=lambda pair: (int(pair[0]["profile"]["cost"]), pair[0]["sha256"]))
    for rank, (item, data) in enumerate(rows[:5], 1):
        known_four = support.evaluate_four(data, known_cases)
        fresh_runs = []
        for seed in (307_200_000 + task, 307_300_000 + task):
            cases, generation = support.fresh_cases(task, seed, task_map)
            runtime = support.evaluate_four(data, cases)
            fresh_runs.append(
                {
                    "seed": seed,
                    "generation": generation,
                    "runtime": {name: base.compact(row) for name, row in runtime.items()},
                    "pass": all(base.row_pass(row) for row in runtime.values()),
                }
            )
        result = compact_item(item)
        result["known_four"] = {name: base.compact(row) for name, row in known_four.items()}
        result["known_four_pass"] = all(base.row_pass(row) for row in known_four.values())
        result["fresh"] = fresh_runs
        result["policy95_pass"] = bool(
            result["known_four_pass"] and all(run["pass"] for run in fresh_runs)
        )
        result["meets_half"] = int(result["profile"]["cost"]) * 2 <= int(
            result["authority_cost"]
        )
        fresh_results.append(result)
        print(
            json.dumps(
                {
                    "fresh_worker": os.getpid(),
                    "task": task,
                    "rank": rank,
                    "cost": result["profile"]["cost"],
                    "fresh": [
                        run["runtime"]["disable_threads1"]["accuracy"] for run in fresh_runs
                    ],
                    "pass": result["policy95_pass"],
                }
            ),
            flush=True,
        )
        if result["policy95_pass"]:
            outdir = Path(outdir_text)
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / (
                f"task{task:03d}_cost{result['profile']['cost']}_"
                f"{result['sha256'][:12]}_POLICY95.onnx"
            )
            path.write_bytes(data)
            result["saved_path"] = str(path.relative_to(base.ROOT))
            finalist = result
            break
    return {"task": task, "fresh_results": fresh_results, "finalist": finalist}


def greedy_task_chunks(indexed_rows: list[tuple[int, dict]], workers: int) -> list[list[tuple[int, dict]]]:
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for pair in indexed_rows:
        grouped[int(pair[1]["task"])].append(pair)
    chunks: list[list[tuple[int, dict]]] = [[] for _ in range(workers)]
    sizes = [0] * workers
    for _task, rows in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        target = min(range(workers), key=lambda i: sizes[i])
        chunks[target].extend(rows)
        sizes[target] += len(rows)
    return [chunk for chunk in chunks if chunk]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()
    workers = max(1, args.workers)
    started = time.monotonic()
    support = base.load_support()
    support.POLICY_THRESHOLD = base.THRESHOLD
    ledger = json.loads(base.EVIDENCE.read_text())
    records = [row for row in ledger["results"] if not row["structural_reasons"]]

    known_screen_by_index: dict[int, dict] = {}
    eligible_by_index: dict[int, tuple[dict, bytes]] = {}
    approximate: list[tuple[int, dict]] = []
    for index, row in enumerate(records, 1):
        existing_profile = row.get("profile")
        if not (row.get("known_exact") and existing_profile):
            approximate.append((index, row))
            continue
        item = dict(row)
        item["known_disable_threads1"] = {
            "total": int(row.get("checked", 0)),
            "right": int(row.get("checked", 0)),
            "wrong": 0,
            "accuracy": 1.0,
            "errors": 0,
            "nonfinite_cases": 0,
            "nonfinite_elements": 0,
            "runtime_shape_mismatches": 0,
            "small_positive_elements_0_to_0_25": 0,
            "sign_mismatch_cases_vs_disable_threads1": 0,
            "sign_mismatch_cells_vs_disable_threads1": 0,
        }
        item["known_policy95"] = True
        if int(existing_profile["cost"]) < int(row["authority_cost"]):
            data = base.resolve(str(row["source"]))
            model = onnx.load_model_from_string(data)
            structure = base.structure_audit(support, int(row["task"]), model, data)
            item["structure"] = structure
            if structure["pass"]:
                eligible_by_index[index] = (compact_item(item), data)
        known_screen_by_index[index] = compact_item(item)

    chunks = greedy_task_chunks(approximate, workers)
    context = mp.get_context("spawn")
    with context.Pool(processes=len(chunks)) as pool:
        for chunk_result in pool.map(screen_approx_chunk, chunks):
            for index, item, data in chunk_result:
                known_screen_by_index[index] = item
                if data is not None:
                    eligible_by_index[index] = (item, data)

    known_screen = [known_screen_by_index[index] for index in range(1, len(records) + 1)]
    eligible = [eligible_by_index[index] for index in sorted(eligible_by_index)]
    grouped: dict[int, list[tuple[dict, bytes]]] = defaultdict(list)
    for item, data in eligible:
        grouped[int(item["task"])].append((item, data))

    outdir = base.HERE / "policy95_history_candidates"
    task_args = [(task, rows, str(outdir)) for task, rows in sorted(grouped.items())]
    if task_args:
        with context.Pool(processes=min(workers, len(task_args))) as pool:
            task_audits = pool.map(audit_task, task_args)
    else:
        task_audits = []
    fresh_results = [row for audit in task_audits for row in audit["fresh_results"]]
    finalists = [audit["finalist"] for audit in task_audits if audit["finalist"] is not None]
    finalists.sort(key=lambda row: int(row["task"]))
    lane_gain = sum(
        math.log(int(row["authority_cost"]) / int(row["profile"]["cost"])) for row in finalists
    )
    payload = {
        "threshold": base.THRESHOLD,
        "workers": workers,
        "history_records": len(records),
        "known_policy95_count": sum(bool(row["known_policy95"]) for row in known_screen),
        "profiled_structure_eligible_count": len(eligible),
        "fresh_audited_count": len(fresh_results),
        "finalist_count": len(finalists),
        "half_finalists": [row for row in finalists if row["meets_half"]],
        "strict_nonhalf_finalists": [row for row in finalists if not row["meets_half"]],
        "finalists": finalists,
        "fresh_results": fresh_results,
        "known_screen": known_screen,
        "lane_score_gain": lane_gain,
        "elapsed_seconds": time.monotonic() - started,
    }
    base.OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "history_records": len(records),
                "known_policy95": payload["known_policy95_count"],
                "eligible": len(eligible),
                "finalists": [
                    {
                        "task": row["task"],
                        "cost": row["profile"]["cost"],
                        "authority_cost": row["authority_cost"],
                        "meets_half": row["meets_half"],
                    }
                    for row in finalists
                ],
                "lane_score_gain": lane_gain,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
