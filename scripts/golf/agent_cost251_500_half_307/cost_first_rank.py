#!/usr/bin/env python3
"""Cost-first ranking of historical strict-reduction artifacts.

The older exhaustive scanner evaluated every known example before learning
whether a model was actually cheaper.  Some deliberately pathological graphs
make that ordering prohibitively slow.  This script profiles one canonical
known input in an isolated subprocess, with a hard timeout, and ranks only the
models whose provisional actual cost is below the pinned authority.  Any such
lead still requires the full four-configuration/two-seed audit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INVENTORY = HERE / "strict_inventory.json"
OUT = HERE / "cost_first_rank.json"
MODELS = HERE / "profile_models"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def candidate_bytes(source: str) -> bytes:
    if "!" in source:
        archive_path, member = source.split("!", 1)
        with zipfile.ZipFile(ROOT / archive_path) as archive:
            return archive.read(member)
    return (ROOT / source).read_bytes()


def first_case(task: int):
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                return benchmark
    raise RuntimeError("no convertible known case")


def worker(task: int, model_path: Path) -> int:
    model = onnx.load(model_path)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize rejected")
    with tempfile.TemporaryDirectory(prefix=f"costfirst_{task:03d}_", dir="/tmp") as tmp:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.profile_file_prefix = str(Path(tmp) / "trace")
        session = ort.InferenceSession(
            sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        benchmark = first_case(task)
        scoring._raw_output(session, benchmark["input"])
        trace = session.end_profiling()
        try:
            memory, params = scoring.score_network(sanitized, trace)
        finally:
            Path(trace).unlink(missing_ok=True)
    if memory is None or params is None or memory < 0 or params < 0:
        raise RuntimeError("unscorable")
    print(json.dumps({"memory": int(memory), "params": int(params),
                      "cost": int(memory + params)}))
    return 0


def profile(row: dict[str, object]) -> dict[str, object]:
    task = int(row["task"])
    sha = str(row["sha256"])
    model_path = MODELS / f"task{task:03d}_{sha}.onnx"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--task", str(task), "--model", str(model_path)]
    try:
        completed = subprocess.run(command, text=True, capture_output=True,
                                   timeout=12, check=False)
    except subprocess.TimeoutExpired:
        return {**row, "profile_status": "timeout", "provisional_profile": None}
    if completed.returncode != 0:
        return {**row, "profile_status": "error", "provisional_profile": None,
                "profile_stderr": completed.stderr[-2000:]}
    try:
        measured = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {**row, "profile_status": "parse_error", "provisional_profile": None,
                "profile_stdout": completed.stdout[-2000:],
                "profile_stderr": completed.stderr[-2000:]}
    return {**row, "profile_status": "ok", "provisional_profile": measured,
            "provisional_strict_lower": measured["cost"] < int(row["authority_cost"])}


def orchestrate(workers: int) -> int:
    inventory = json.loads(INVENTORY.read_text())
    rows = [row for row in inventory["results"] if row["structurally_safe"]]
    MODELS.mkdir(exist_ok=True)
    materialized = []
    for row in rows:
        data = candidate_bytes(str(row["source"]))
        digest = hashlib.sha256(data).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"SHA mismatch: {row['source']}")
        path = MODELS / f"task{int(row['task']):03d}_{digest}.onnx"
        if not path.exists():
            path.write_bytes(data)
        materialized.append(row)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(profile, row): row for row in materialized}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or result.get("provisional_strict_lower"):
                print(json.dumps({"i": index, "n": len(materialized),
                                  "task": result["task"],
                                  "status": result["profile_status"],
                                  "cost": (result.get("provisional_profile") or {}).get("cost"),
                                  "authority": result["authority_cost"],
                                  "lower": result.get("provisional_strict_lower", False)}),
                      flush=True)
    results.sort(key=lambda row: (int(row["task"]),
                                  (row.get("provisional_profile") or {}).get("cost", 10**18),
                                  str(row["sha256"])))
    leads = [row for row in results if row.get("provisional_strict_lower")]
    payload = {
        "inventory": str(INVENTORY.relative_to(ROOT)),
        "candidate_count": len(materialized),
        "workers": workers,
        "timeout_seconds": 12,
        "method": "one canonical known input; provisional only",
        "provisional_strict_lower_count": len(leads),
        "provisional_strict_lower": leads,
        "results": results,
        "protected_writes": "none; lane evidence only",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(materialized), "leads": len(leads),
                      "evidence": str(OUT.relative_to(ROOT))}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", type=int)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        if args.task is None or args.model is None:
            parser.error("worker requires --task and --model")
        return worker(args.task, args.model)
    return orchestrate(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
