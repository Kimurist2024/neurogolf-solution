#!/usr/bin/env python3
"""Harvest recent others pools against exact submission_base_7999.13 members.

Scratch-only: all outputs stay beside this script.  The scanner uses the exact
baseline ONNX bytes as cost authority, performs strict static safety checks,
profiles with ORT_DISABLE_ALL, and runs complete known-gold verification before
copying a task winner.  It never reads all_scores.csv and never writes a root
submission, score pointer, ledger, or artifacts tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


ort.set_default_logger_severity(4)

BASE_ZIP = ROOT / "submission_base_7999.13.zip"
SOURCE_ROOTS = (
    ROOT / "others/2/7908",
    ROOT / "others/2/1200",
    ROOT / "others/2/1201",
    ROOT / "others/2/1202",
    ROOT / "others/2/1203",
    ROOT / "others/2/1294",
    ROOT / "others/2/1300",
    ROOT / "others/7130",
    ROOT / "others/7131",
    ROOT / "others/7132",
    ROOT / "others/7905",
    ROOT / "others/7906",
    ROOT / "others/7907",
)

# Exact task groups excluded by the latest error-minimized baseline audit,
# plus task158's repeated processing/private-zero history and task333's
# unresolved 198/333 private-zero ambiguity / non-terminating safety audit.
BLACK_OR_PRIVATE_ZERO = {
    9, 76, 77, 101, 112, 134, 158, 168, 185, 192, 196, 198,
    201, 208, 219, 251, 286, 333, 343, 344, 391, 396,
}
UB_EXCLUDED = {96}
EXCLUDED = BLACK_OR_PRIVATE_ZERO | UB_EXCLUDED

MAX_BYTES = 1_440_000
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
FILE_RE = re.compile(r"^task(\d{3})(?:[^0-9].*)?\.onnx$", re.IGNORECASE)
MEMBER_RE = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base_payloads() -> dict[int, bytes]:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        return {task: archive.read(f"task{task:03d}.onnx") for task in range(1, 401)}


def task_from_file(path: Path) -> int | None:
    match = FILE_RE.match(path.name)
    if match:
        task = int(match.group(1))
        return task if 1 <= task <= 400 else None
    return None


def task_from_member(name: str) -> int | None:
    match = MEMBER_RE.search(name)
    if not match:
        return None
    task = int(match.group(1))
    return task if 1 <= task <= 400 else None


def inventory(base: dict[int, bytes]) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    base_hash = {task: digest(data) for task, data in base.items()}
    candidates: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: defaultdict[str, int] = defaultdict(int)
    errors: list[dict[str, str]] = []

    def add(task: int, data: bytes, source: str, kind: str) -> None:
        counts[f"{kind}_observations"] += 1
        if task in EXCLUDED:
            counts["excluded_task_observations"] += 1
            return
        sha = digest(data)
        if sha == base_hash[task]:
            counts["exact_baseline_duplicates"] += 1
            return
        slot = candidates[task].setdefault(
            sha, {"data": data, "sha256": sha, "sources": [], "source_kinds": []}
        )
        slot["sources"].append(source)
        slot["source_kinds"].append(kind)

    for source_root in SOURCE_ROOTS:
        if not source_root.exists():
            counts["missing_roots"] += 1
            continue
        for path in sorted(source_root.rglob("*.onnx")):
            task = task_from_file(path)
            if task is None:
                continue
            rel = str(path.relative_to(ROOT))
            try:
                add(task, path.read_bytes(), rel, "loose")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": rel, "error": repr(exc)})
        for path in sorted(source_root.rglob("*.zip")):
            rel = str(path.relative_to(ROOT))
            counts["zip_files_seen"] += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        task = task_from_member(name)
                        if task is None:
                            continue
                        add(task, archive.read(name), f"{rel}::{name}", "zip_member")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": rel, "error": repr(exc)})

    counts["unique_different"] = sum(len(items) for items in candidates.values())
    return dict(candidates), {
        "source_roots": [str(path.relative_to(ROOT)) for path in SOURCE_ROOTS],
        "counts": dict(counts),
        "unique_by_task": {str(task): len(items) for task, items in sorted(candidates.items())},
        "errors": errors,
    }


def exact_conv_bias_gate(model: onnx.ModelProto) -> tuple[bool, str | None, list[Any]]:
    official_findings = check_conv_bias(model)
    if official_findings:
        return False, f"check_conv_bias:{official_findings}", official_findings

    # Strengthen the project checker: a present bias must be an initializer and
    # exactly match the statically inferred output channel count.  This catches
    # dynamic-bias cases that the diagnostic checker intentionally cannot prove.
    initializers = {item.name: item for item in model.graph.initializer}
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"bias_inference:{type(exc).__name__}", official_findings
    out_channels: dict[str, int] = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) >= 2 and dims[1].HasField("dim_value") and dims[1].dim_value > 0:
            out_channels[value.name] = int(dims[1].dim_value)
    for node in model.graph.node:
        index = 8 if node.op_type == "QLinearConv" else (2 if node.op_type in {"Conv", "ConvTranspose"} else None)
        if index is None or len(node.input) <= index or not node.input[index]:
            continue
        bias = initializers.get(node.input[index])
        if bias is None:
            return False, f"dynamic_or_external_conv_bias:{node.output[0]}", official_findings
        expected = out_channels.get(node.output[0])
        if expected is None:
            return False, f"unknown_conv_output_channels:{node.output[0]}", official_findings
        try:
            actual = int(numpy_helper.to_array(bias).size)
        except Exception as exc:  # noqa: BLE001
            return False, f"unreadable_conv_bias:{type(exc).__name__}", official_findings
        if actual != expected:
            return False, f"conv_bias_length:{actual}!={expected}:{node.output[0]}", official_findings
    return True, None, official_findings


def structure_gate(data: bytes) -> tuple[onnx.ModelProto | None, str, int | None]:
    if len(data) > MAX_BYTES:
        return None, "file_too_large", None
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"checker_or_shape:{type(exc).__name__}:{exc}", None
    if model.functions:
        return None, "local_functions", None
    if model.graph.sparse_initializer:
        return None, "sparse_initializer", None
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return None, "noncanonical_io_count", None
    if model.graph.input[0].name != "input" or model.graph.output[0].name != "output":
        return None, "noncanonical_io_names", None
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL or initializer.external_data:
            return None, "external_initializer", None
    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            return None, f"custom_opset:{opset.domain}", None
    for node in model.graph.node:
        upper = node.op_type.upper()
        if node.domain not in ("", "ai.onnx"):
            return None, f"custom_node_domain:{node.domain}", None
        if upper in BANNED or "SEQUENCE" in upper:
            return None, f"banned_op:{node.op_type}", None
        if node.op_type == "Einsum" and len(node.input) >= 15:
            return None, f"giant_einsum:{len(node.input)}", None
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                return None, f"nested_graph:{node.op_type}", None
            if attr.type in (onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                return None, f"sparse_attribute:{node.op_type}", None
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        if not value.type.HasField("tensor_type"):
            return None, f"non_tensor:{value.name}", None
        dims = value.type.tensor_type.shape.dim
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
            return None, f"non_static_shape:{value.name}", None
    bias_ok, reason, _ = exact_conv_bias_gate(model)
    if not bias_ok:
        return None, reason or "conv_bias", None
    static_floor = static_cost_floor(inferred)
    return model, "pass", static_floor


def static_cost_floor(inferred: onnx.ModelProto) -> int | None:
    try:
        graph = inferred.graph
        initializer_names = {item.name for item in graph.initializer}
        tensor_map = {
            value.name: value
            for value in list(graph.input) + list(graph.value_info) + list(graph.output)
        }
        names = {name for node in graph.node for name in node.output if name}
        names.update(tensor_map)
        memory = 0
        for name in names:
            if name in {"input", "output"} or name in initializer_names:
                continue
            value = tensor_map.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                return None
            tensor = value.type.tensor_type
            elements = math.prod(dim.dim_value for dim in tensor.shape.dim)
            memory += elements * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        params = scoring.calculate_params(inferred)
        return None if params is None else int(memory + params)
    except Exception:  # noqa: BLE001
        return None


def known_score(data: bytes, task: int, require_correct: bool, label: str) -> dict[str, Any] | None:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"harvest_{task:03d}_") as workdir:
        return scoring.score_and_verify(
            model, task, workdir, label=label, require_correct=require_correct
        )


def actual_screen(data: bytes, task: int) -> int | None:
    trace: str | None = None
    try:
        model = onnx.load_model_from_string(data)
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return None
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        with tempfile.TemporaryDirectory(prefix=f"harvest_screen_{task:03d}_") as workdir:
            options.profile_file_prefix = str(Path(workdir) / "trace")
            session = ort.InferenceSession(sanitized.SerializeToString(), options)
            examples = scoring.load_examples(task)
            benchmark = next(
                converted
                for subset in ("train", "test", "arc-gen")
                for example in examples[subset]
                if (converted := scoring.convert_to_numpy(example)) is not None
            )
            session.run(["output"], {"input": benchmark["input"]})
            trace = session.end_profiling()
            memory = scoring.calculate_memory(sanitized, trace)
            params = scoring.calculate_params(sanitized)
            if memory is None or params is None:
                return None
            return int(memory + params)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if trace:
            Path(trace).unlink(missing_ok=True)


def baseline_worker(job: tuple[int, bytes]) -> dict[str, Any]:
    task, data = job
    try:
        result = known_score(data, task, False, "base7999_13")
        return {"task": task, "result": result}
    except BaseException:  # noqa: BLE001
        return {"task": task, "error": traceback.format_exc(limit=8)}


def screen_worker(job: tuple[str, int, bytes]) -> dict[str, Any]:
    sha, task, data = job
    try:
        return {"sha256": sha, "task": task, "cost": actual_screen(data, task)}
    except BaseException:  # noqa: BLE001
        return {"sha256": sha, "task": task, "error": traceback.format_exc(limit=8)}


def known_worker(job: tuple[str, int, bytes]) -> dict[str, Any]:
    sha, task, data = job
    try:
        return {
            "sha256": sha,
            "task": task,
            "result": known_score(data, task, True, f"candidate_{sha[:10]}"),
        }
    except BaseException:  # noqa: BLE001
        return {"sha256": sha, "task": task, "error": traceback.format_exc(limit=8)}


def run_bounded(
    jobs: list[Any], worker: Callable[[Any], dict[str, Any]], *, max_workers: int, timeout: float, label: str
) -> list[dict[str, Any]]:
    """Spawn-isolated bounded runner with a hard per-job timeout."""
    context = mp.get_context("spawn")
    pending = iter(jobs)
    active: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    exhausted = False
    completed = 0
    while active or not exhausted:
        while len(active) < max_workers and not exhausted:
            try:
                job = next(pending)
            except StopIteration:
                exhausted = True
                break
            output_queue = context.Queue(maxsize=1)
            process = context.Process(target=_worker_entry, args=(worker, job, output_queue))
            process.start()
            active.append({"job": job, "process": process, "queue": output_queue, "start": time.monotonic()})
        now = time.monotonic()
        for slot in list(active):
            process = slot["process"]
            elapsed = now - slot["start"]
            if process.is_alive() and elapsed <= timeout:
                continue
            if process.is_alive():
                process.terminate()
                process.join(2)
                result = {"timeout": True, "job_key": _job_key(slot["job"]), "elapsed": elapsed}
            else:
                process.join()
                try:
                    result = slot["queue"].get_nowait()
                except queue.Empty:
                    result = {"error": f"worker_exit_{process.exitcode}", "job_key": _job_key(slot["job"])}
            slot["queue"].close()
            active.remove(slot)
            results.append(result)
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"{label} {completed}/{len(jobs)}", flush=True)
        if active:
            time.sleep(0.03)
    return results


def _worker_entry(worker: Callable[[Any], dict[str, Any]], job: Any, output_queue: mp.Queue) -> None:
    ort.set_default_logger_severity(4)
    try:
        output_queue.put(worker(job))
    except BaseException:  # noqa: BLE001
        output_queue.put({"error": traceback.format_exc(limit=12), "job_key": _job_key(job)})


def _job_key(job: Any) -> Any:
    if isinstance(job, tuple):
        if len(job) == 2:
            return job[0]
        if len(job) >= 3:
            return {"sha256": job[0], "task": job[1]}
    return repr(job)


def main() -> int:
    started = time.time()
    base = base_payloads()
    candidates, inv = inventory(base)
    tasks = sorted(candidates)
    print(
        f"INVENTORY tasks={len(tasks)} unique={inv['counts'].get('unique_different', 0)}",
        flush=True,
    )

    baseline_jobs = [(task, base[task]) for task in tasks]
    baseline_raw = run_bounded(
        baseline_jobs, baseline_worker, max_workers=4, timeout=25.0, label="BASE"
    )
    baseline: dict[int, dict[str, Any]] = {}
    baseline_failures: list[dict[str, Any]] = []
    for item in baseline_raw:
        task = item.get("task")
        result = item.get("result")
        if task is None or result is None:
            baseline_failures.append(item)
            continue
        baseline[int(task)] = {
            **result,
            "sha256": digest(base[int(task)]),
            "bytes": len(base[int(task)]),
        }
    (HERE / "baseline_costs.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "baseline_zip_sha256": digest(BASE_ZIP.read_bytes()),
                "costs": {str(task): value for task, value in sorted(baseline.items())},
                "failures": baseline_failures,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"BASE DONE ok={len(baseline)} failed={len(baseline_failures)}", flush=True)

    rows: list[dict[str, Any]] = []
    screen_jobs: list[tuple[str, int, bytes]] = []
    row_by_sha: dict[str, dict[str, Any]] = {}
    data_by_sha: dict[str, bytes] = {}
    for task in tasks:
        for sha, item in candidates[task].items():
            row: dict[str, Any] = {
                "task": task,
                "sha256": sha,
                "bytes": len(item["data"]),
                "sources": item["sources"],
                "source_kinds": sorted(set(item["source_kinds"])),
            }
            rows.append(row)
            row_by_sha[sha] = row
            if task not in baseline:
                row.update(stage="baseline_unavailable", reason="exact_member_not_scorable_or_timeout")
                continue
            row["baseline_cost"] = int(baseline[task]["cost"])
            model, reason, floor = structure_gate(item["data"])
            row["static_cost_floor"] = floor
            if model is None:
                row.update(stage="structure_reject", reason=reason)
                continue
            if floor is None or floor >= row["baseline_cost"]:
                row.update(stage="static_reject", reason="static_floor_not_strictly_cheaper")
                continue
            row["stage"] = "screen_pending"
            screen_jobs.append((sha, task, item["data"]))
            data_by_sha[sha] = item["data"]

    print(f"STATIC DONE screen_jobs={len(screen_jobs)}", flush=True)
    screen_raw = run_bounded(
        screen_jobs, screen_worker, max_workers=4, timeout=18.0, label="SCREEN"
    )
    known_jobs: list[tuple[str, int, bytes]] = []
    for item in screen_raw:
        sha = item.get("sha256")
        if sha is None or sha not in row_by_sha:
            continue
        row = row_by_sha[sha]
        cost = item.get("cost")
        row["actual_screen_cost"] = cost
        if cost is None or int(cost) >= int(row["baseline_cost"]):
            row.update(stage="actual_reject", reason="actual_cost_not_strictly_cheaper_or_timeout")
            continue
        row["stage"] = "known_pending"
        known_jobs.append((sha, int(row["task"]), data_by_sha[sha]))

    print(f"SCREEN DONE known_jobs={len(known_jobs)}", flush=True)
    known_raw = run_bounded(
        known_jobs, known_worker, max_workers=4, timeout=35.0, label="KNOWN"
    )
    winners: dict[int, dict[str, Any]] = {}
    for item in known_raw:
        sha = item.get("sha256")
        if sha is None or sha not in row_by_sha:
            continue
        row = row_by_sha[sha]
        result = item.get("result")
        if result is None or not result.get("correct"):
            row.update(stage="known_reject", reason="complete_gold_or_runtime_failure")
            continue
        candidate_cost = int(result["cost"])
        row.update(
            candidate_cost=candidate_cost,
            candidate_memory=int(result["memory"]),
            candidate_params=int(result["params"]),
        )
        if candidate_cost >= int(row["baseline_cost"]):
            row.update(stage="known_reject", reason="complete_profile_not_strictly_cheaper")
            continue
        row["stage"] = "known_pass"
        row["gain"] = math.log(int(row["baseline_cost"]) / candidate_cost)
        task = int(row["task"])
        previous = winners.get(task)
        if previous is not None and int(previous["candidate_cost"]) <= candidate_cost:
            continue
        output = HERE / f"winner_task{task:03d}.onnx"
        output.write_bytes(data_by_sha[sha])
        winners[task] = {**row, "candidate": str(output.relative_to(ROOT))}
        print(
            f"WIN task{task:03d} {row['baseline_cost']}->{candidate_cost} gain={row['gain']:.9f}",
            flush=True,
        )

    report = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASE_ZIP.read_bytes()),
        "excluded_black_or_private_zero": sorted(BLACK_OR_PRIVATE_ZERO),
        "excluded_ub": sorted(UB_EXCLUDED),
        "inventory": inv,
        "baseline_failures": baseline_failures,
        "rows": rows,
        "known_winners": {str(task): value for task, value in sorted(winners.items())},
        "known_gain": sum(float(value["gain"]) for value in winners.values()),
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "scan_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "known_winner_manifest.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "baseline_zip_sha256": digest(BASE_ZIP.read_bytes()),
                "winners": {str(task): value for task, value in sorted(winners.items())},
                "known_gain": report["known_gain"],
                "pending": "random_differential_and_external_validation",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"DONE winners={len(winners)} gain={report['known_gain']:.9f} elapsed={report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
