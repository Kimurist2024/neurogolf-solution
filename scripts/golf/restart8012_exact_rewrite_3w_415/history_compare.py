#!/usr/bin/env python3
"""Three-process rebase of profiled history against the 8012.15 raw authority."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
SOURCE = ROOT / "scripts/golf/agent_cost251_500_half_307/cost_first_rank.json"
MODELS = ROOT / "scripts/golf/agent_cost251_500_half_307/profile_models"
WORKERS = 3
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102,
    112, 133, 134, 138, 145, 157, 158, 161, 169, 170, 173, 174, 175,
    178, 185, 187, 192, 196, 198, 202, 205, 208, 209, 216, 219, 222,
    233, 246, 255, 277, 285, 286, 302, 319, 325, 333, 343, 344, 346,
    354, 355, 361, 365, 366, 372, 377, 379, 391, 393, 396,
}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    f"restart415_history_exact_{os.getpid()}",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/scan_and_build.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def current_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="") as stream:
        return {int(row["task"][4:]): int(row["cost"]) for row in csv.DictReader(stream)}


def rows() -> list[dict[str, Any]]:
    costs = current_costs()
    source = json.loads(SOURCE.read_text())
    result = []
    for row in source["provisional_strict_lower"]:
        task = int(row["task"])
        if task in EXCLUDED or not 100 <= costs[task] <= 500:
            continue
        prior = row.get("provisional_profile") or {}
        if int(prior.get("cost", 10**18)) >= costs[task]:
            continue
        result.append({**row, "current_authority_cost": costs[task]})
    return result


def session(model: onnx.ModelProto, level: ort.GraphOptimizationLevel, threads: int):
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(sanitized.SerializeToString(), options,
                                providers=["CPUExecutionProvider"])


def known_inputs(task: int) -> list[np.ndarray]:
    result = []
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                result.append(benchmark["input"])
    return result


def compare(authority: onnx.ModelProto, candidate: onnx.ModelProto, task: int) -> dict[str, Any]:
    try:
        left = session(authority, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1)
        right = session(candidate, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1)
    except Exception as exc:
        return {"pass": False, "reason": f"session:{type(exc).__name__}:{exc}"}
    checked = raw_mismatch = runtime_errors = nonfinite = shape = small = 0
    for array in known_inputs(task):
        try:
            a = left.run(["output"], {"input": array})[0]
            b = right.run(["output"], {"input": array})[0]
        except Exception:
            runtime_errors += 1
            break
        checked += 1
        raw_mismatch += int(not np.array_equal(a, b))
        nonfinite += int(not np.all(np.isfinite(b)))
        shape += int(tuple(b.shape) != (1, 10, 30, 30))
        small += int(np.count_nonzero((b > 0) & (b < 0.25)))
        if raw_mismatch or nonfinite or shape or small:
            break
    return {
        "pass": checked > 0 and raw_mismatch == runtime_errors == nonfinite == shape == small == 0,
        "checked": checked, "raw_mismatch": raw_mismatch,
        "runtime_errors": runtime_errors, "nonfinite_cases": nonfinite,
        "shape_mismatch_cases": shape, "small_positive_values": small,
    }


def worker(worker_id: int, assigned: list[dict[str, Any]]) -> dict[str, Any]:
    output = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for row in assigned:
            task = int(row["task"])
            digest = str(row["sha256"])
            path = MODELS / f"task{task:03d}_{digest}.onnx"
            item = {key: value for key, value in row.items() if key not in {"ops"}}
            if not path.exists() or sha256(path.read_bytes()) != digest:
                item.update({"status": "materialization_mismatch"})
                output.append(item)
                continue
            candidate = onnx.load(path)
            audit, _ = EXACT.structural_audit(candidate)
            if not audit.get("pass"):
                item.update({"status": "strict_reject", "structural": audit})
                output.append(item)
                continue
            authority = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            comparison = compare(authority, candidate, task)
            item["known_raw_comparison"] = comparison
            if not comparison["pass"]:
                item["status"] = "known_raw_reject"
                output.append(item)
                continue
            with tempfile.TemporaryDirectory(prefix=f"restart415_hist_t{task:03d}_", dir="/tmp") as tmp:
                profile = scoring.score_and_verify(candidate, task, tmp, "history", require_correct=False)
            item["official_profile"] = profile
            if profile is None or int(profile["cost"]) >= int(row["current_authority_cost"]):
                item["status"] = "not_current_lower"
                output.append(item)
                continue
            destination = HERE / "candidates" / f"history_worker_{worker_id}" / f"task{task:03d}_{digest[:12]}.onnx"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            item.update({
                "status": "history_known_raw_exact_finalist",
                "path": str(destination.relative_to(ROOT)),
            })
            output.append(item)
    payload = {
        "worker": worker_id, "pid": os.getpid(), "assigned": len(assigned),
        "results": output,
        "finalists": [item for item in output if item["status"] == "history_known_raw_exact_finalist"],
    }
    (HERE / f"history_worker_{worker_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA")
    source_rows = rows()
    parts = [source_rows[index::WORKERS] for index in range(WORKERS)]
    if args.worker is not None:
        result = worker(args.worker, parts[args.worker])
        print(json.dumps({"worker": args.worker, "assigned": len(parts[args.worker]),
                          "finalists": len(result["finalists"])}))
        return 0
    processes = []
    handles = []
    for worker_id in range(WORKERS):
        handle = (HERE / f"history_worker_{worker_id}.log").open("w")
        handles.append(handle)
        processes.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(worker_id)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        ))
    failures = []
    for worker_id, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append({"worker": worker_id, "exit": code})
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError(failures)
    results = [json.loads((HERE / f"history_worker_{worker_id}.json").read_text())
               for worker_id in range(WORKERS)]
    finalists = [row for result in results for row in result["finalists"]]
    payload = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": AUTHORITY_SHA256},
        "workers": WORKERS, "source": str(SOURCE.relative_to(ROOT)),
        "source_rows": len(source_rows), "partitions": [len(part) for part in parts],
        "finalists": finalists,
    }
    (HERE / "history_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"source_rows": len(source_rows), "workers": WORKERS,
                      "finalists": len(finalists),
                      "tasks": sorted({int(row["task"]) for row in finalists})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
