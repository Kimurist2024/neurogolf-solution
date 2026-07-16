#!/usr/bin/env python3
"""Reprofile loose and ZIP history against the pinned 8012.15 mid/high scope."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
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
EVIDENCE = HERE / "history_evidence.json"
OUT = HERE / "history_candidates"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
KNOWN_LB_BLACK_TASKS = {70, 134, 202, 343}
TASK_RE = re.compile(r"task[_-]?(\d{3})(?!\d)", re.IGNORECASE)


def task_from(text: str) -> int | None:
    match = TASK_RE.search(text)
    return int(match.group(1)) if match else None


def params(model: onnx.ModelProto) -> int:
    value = scoring.calculate_params(model)
    return int(value) if value is not None else 10**18


def declared_floor(model: onnx.ModelProto) -> int:
    result = params(model)
    outputs = {value.name for value in model.graph.output}
    values = {
        value.name: value
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    seen: set[str] = set()
    for node in model.graph.node:
        for name in node.output:
            if not name or name in outputs or name in seen:
                continue
            seen.add(name)
            value = values.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            tensor = value.type.tensor_type
            dims = [int(dim.dim_value) for dim in tensor.shape.dim]
            if not dims or any(dim <= 0 for dim in dims):
                continue
            try:
                itemsize = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type).itemsize
            except Exception:
                continue
            result += math.prod(dims) * int(itemsize)
    return int(result)


def structural(model: onnx.ModelProto) -> tuple[bool, str | None]:
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        if model.functions or model.graph.sparse_initializer:
            raise ValueError("functions or sparse initializers")
        for init in model.graph.initializer:
            array = onnx.numpy_helper.to_array(init)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"nonfinite initializer {init.name}")
        for node in model.graph.node:
            if node.op_type in BANNED or "Sequence" in node.op_type:
                raise ValueError(f"banned op {node.op_type}")
            for attr in node.attribute:
                if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                    raise ValueError("nested graph")
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param:
                    raise ValueError("dynamic dim_param")
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("noncanonical graph I/O arity")
        for label, value in (("input", model.graph.input[0]), ("output", model.graph.output[0])):
            dims = [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
            if dims != [1, 10, 30, 30]:
                raise ValueError(f"{label} shape cloak/noncanonical declaration: {dims}")
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def runtime(model: onnx.ModelProto, optimization: ort.GraphOptimizationLevel, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise ValueError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def evaluate(
    session: ort.InferenceSession,
    cases: list[dict[str, np.ndarray]],
    limit: int | None,
    min_accuracy: float,
) -> dict[str, Any]:
    selected = cases if limit is None else cases[:limit]
    right = wrong = errors = nonfinite = smallpositive = shape_errors = 0
    allowed_wrong = int(math.floor(len(selected) * (1.0 - min_accuracy) + 1e-9))
    for item in selected:
        try:
            raw = session.run(["output"], {"input": item["input"]})[0]
            if raw.shape != item["output"].shape:
                shape_errors += 1
                wrong += 1
                break
            if not np.isfinite(raw).all():
                nonfinite += 1
                break
            if np.any((raw > 0.0) & (raw < 0.25)):
                smallpositive += 1
                break
            if np.array_equal(raw > 0.0, item["output"] > 0.0):
                right += 1
            else:
                wrong += 1
                if wrong > allowed_wrong:
                    break
        except Exception:  # fail closed
            errors += 1
            break
    return {
        "total": len(selected), "right": right, "wrong": wrong, "errors": errors,
        "nonfinite": nonfinite, "smallpositive": smallpositive,
        "shape_errors": shape_errors,
        "accuracy": right / len(selected) if selected else 0.0,
    }


def clean(row: dict[str, Any]) -> bool:
    return not any(row[key] for key in ("errors", "nonfinite", "smallpositive", "shape_errors"))


def read_source(row: dict[str, Any]) -> bytes:
    if row["kind"] == "loose":
        return (ROOT / row["path"]).read_bytes()
    with zipfile.ZipFile(ROOT / row["path"]) as archive:
        return archive.read(row["member"])


def evaluate_candidate(index: int, row: dict[str, Any], known: dict[int, list[dict[str, np.ndarray]]]) -> dict[str, Any]:
    item = dict(row)
    task = int(row["task"])
    try:
        data = read_source(row)
        model = onnx.load_model_from_string(data)
        ok, error = structural(model)
        item["structural_ok"] = ok
        item["structural_error"] = error
        if not ok:
            return item
        session = runtime(model, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1)
        quick = evaluate(session, known[task], 20, 0.70)
        item["quick"] = quick
        if not clean(quick) or quick["accuracy"] < 0.70:
            return item
        full = evaluate(session, known[task], None, 0.90)
        item["known"] = full
        if not clean(full) or full["accuracy"] < 0.90:
            return item
        with tempfile.TemporaryDirectory(prefix=f"hist406_{task:03d}_", dir="/tmp") as work:
            profile = scoring.score_and_verify(
                model, task, work, label=f"hist406_{index}", require_correct=False
            )
        item["profile"] = profile
        if profile is None:
            return item
        cost = int(profile["cost"])
        item["strict_lower"] = cost < int(row["authority_cost"])
        item["half"] = 2 * cost <= int(row["authority_cost"])
        item["known_exact"] = full["right"] == full["total"] and full["wrong"] == 0
        if not item["strict_lower"]:
            return item
        # Four runtime configurations are cheap on the official known set and
        # reject optimizer/thread dependent behavior before any fresh campaign.
        config_rows = []
        for name, optimization, threads in (
            ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
            ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
            ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
            ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
        ):
            tested = evaluate(runtime(model, optimization, threads), known[task], None, 0.90)
            config_rows.append({"name": name, **tested})
        item["known_four_configs"] = config_rows
        item["four_configs_pass90"] = all(clean(value) and value["accuracy"] >= 0.90 for value in config_rows)
        if not item["four_configs_pass90"]:
            return item
        OUT.mkdir(parents=True, exist_ok=True)
        output = OUT / f"task{task:03d}_cost{cost}_{row['sha256'][:12]}.onnx"
        output.write_bytes(data)
        item["candidate_path"] = str(output.relative_to(ROOT))
    except Exception as exc:  # noqa: BLE001
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def main() -> int:
    started = time.monotonic()
    ort.set_default_logger_severity(3)
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    authority = json.loads((HERE / "authority.json").read_text())
    full_scope = {int(row["task"]): int(row["cost"]) for row in authority["scope"]}
    scope = {task: cost for task, cost in full_scope.items() if task not in KNOWN_LB_BLACK_TASKS}
    known = {}
    for task in scope:
        examples = scoring.load_examples(task)
        known[task] = [
            converted
            for example in examples["train"] + examples["test"] + examples["arc-gen"]
            if (converted := scoring.convert_to_numpy(example)) is not None
        ]

    loose_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.onnx", "-g", "!scripts/golf/restart8012_midhigh_3w_406/**"],
        cwd=ROOT, text=True,
    ).splitlines()
    zip_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.zip", "-g", "!scripts/golf/restart8012_midhigh_3w_406/**"],
        cwd=ROOT, text=True,
    ).splitlines()

    seen: set[tuple[int, str]] = set()
    candidates: list[dict[str, Any]] = []
    inventory_errors = 0

    def consider(task: int | None, data: bytes, source: dict[str, Any]) -> None:
        nonlocal inventory_errors
        if task not in scope:
            return
        source_label = f"{source.get('path', '')}/{source.get('member', '')}".lower()
        if any(token in source_label for token in (
            "shape_cloak", "nonfinite", "runtime_error", "quarantine",
            "probe_only", "short_bias", "banned_op",
        )):
            return
        try:
            if len(data) > 1_440_000:
                return
            digest = hashlib.sha256(data).hexdigest()
            key = (int(task), digest)
            if key in seen:
                return
            seen.add(key)
            model = onnx.load_model_from_string(data)
            parameter_count = params(model)
            floor = declared_floor(model)
            if parameter_count >= scope[int(task)] or floor >= scope[int(task)]:
                return
            candidates.append({
                "task": int(task), "authority_cost": scope[int(task)],
                "sha256": digest, "params": parameter_count,
                "declared_floor": floor, "nodes": len(model.graph.node),
                "file_bytes": len(data),
                "ops": [node.op_type for node in model.graph.node], **source,
            })
        except Exception:
            inventory_errors += 1

    for relpath in loose_paths:
        try:
            consider(task_from(relpath), (ROOT / relpath).read_bytes(), {"kind": "loose", "path": relpath})
        except Exception:
            inventory_errors += 1

    for relpath in zip_paths:
        try:
            with zipfile.ZipFile(ROOT / relpath) as archive:
                for member in archive.namelist():
                    task = task_from(member)
                    if task not in scope or not member.lower().endswith(".onnx"):
                        continue
                    try:
                        data = archive.read(member)
                    except Exception:
                        inventory_errors += 1
                        continue
                    consider(task, data, {"kind": "zip", "path": relpath, "member": member})
        except Exception:
            inventory_errors += 1

    candidates.sort(key=lambda row: (
        int(row["nodes"]), int(row["file_bytes"]), int(row["declared_floor"]),
        int(row["task"]), str(row["sha256"]),
    ))
    workers = max(1, int(os.environ.get("NG_HISTORY_WORKERS", "3")))
    print(json.dumps({
        "loose_paths": len(loose_paths), "zip_paths": len(zip_paths),
        "unique_task_sha": len(seen), "candidates": len(candidates),
        "inventory_errors": inventory_errors, "workers": workers,
    }), flush=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_candidate, index, row, known): (index, row)
            for index, row in enumerate(candidates, 1)
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            index, row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {**row, "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            done += 1
            if result.get("candidate_path") or done % 100 == 0:
                print(json.dumps({
                    "done": done, "total": len(candidates), "task": result["task"],
                    "cost": (result.get("profile") or {}).get("cost"),
                    "known": (result.get("known") or {}).get("accuracy"),
                    "strict": bool(result.get("candidate_path")), "half": bool(result.get("half")),
                }), flush=True)

    winners = [row for row in results if row.get("candidate_path")]
    winners.sort(key=lambda row: (
        int(row["task"]), int((row.get("profile") or {})["cost"]),
        -float((row.get("known") or {})["accuracy"]), str(row["sha256"]),
    ))
    best: dict[int, dict[str, Any]] = {}
    for row in winners:
        best.setdefault(int(row["task"]), row)
    payload = {
        "authority": authority["authority"], "authority_sha256": AUTHORITY_SHA256,
        "scope_count": len(scope), "known_lb_black_tasks_excluded": sorted(KNOWN_LB_BLACK_TASKS),
        "policy_threshold": 0.90, "loose_path_count": len(loose_paths),
        "zip_path_count": len(zip_paths), "unique_task_sha_count": len(seen),
        "theoretical_strict_lower_count": len(candidates), "workers": workers,
        "inventory_errors": inventory_errors, "winner_count": len(winners),
        "winner_task_count": len(best), "best_by_task": list(best.values()),
        "winners": winners, "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(candidates), "winner_count": len(winners),
        "winner_tasks": sorted(best), "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
