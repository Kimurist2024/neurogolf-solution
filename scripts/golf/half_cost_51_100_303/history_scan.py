#!/usr/bin/env python3
"""Find already-built sound candidates at <= half the 8011.05 authority cost.

This is an evidence-only scan.  It never updates submission.zip, all_scores.csv,
or any shared candidate pool.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
EVIDENCE = HERE / "history_evidence.json"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


# Task-level monitor.  A new, independently reconstructed rule could be sound,
# but historical cheap models for these tasks must not be admitted here.
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 112, 133, 134, 138, 145, 157, 158, 169, 170, 173, 174,
    178, 185, 187, 192, 196, 198, 202, 205, 208, 209, 216, 219,
    222, 233, 246, 255, 277, 285, 286, 302, 319, 325, 333, 346,
    361, 365, 366, 372, 377, 379, 391, 393, 396,
}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_from_path(path: str) -> int | None:
    match = re.search(r"task[_-]?(\d{3})(?!\d)", path, re.IGNORECASE)
    return int(match.group(1)) if match else None


def authority_costs() -> dict[int, int]:
    result = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if 51 <= cost <= 100 and float(row["score"]) < 25.0:
                result[task] = cost
    return result


def params(model: onnx.ModelProto) -> int:
    value = scoring.calculate_params(model)
    return int(value) if value is not None else 10**18


def declared_lower_bound(model: onnx.ModelProto) -> int:
    total = params(model)
    graph_outputs = {value.name for value in model.graph.output}
    values = {value.name: value for value in (
        list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    )}
    counted = set()
    for node in model.graph.node:
        for name in node.output:
            if not name or name in graph_outputs or name in counted:
                continue
            counted.add(name)
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
            total += math.prod(dims) * itemsize
    return int(total)


def structurally_safe(model: onnx.ModelProto) -> tuple[bool, list[str]]:
    reasons = []
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializer")
    for node in model.graph.node:
        if node.op_type in BANNED or "Sequence" in node.op_type:
            reasons.append(f"banned:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                reasons.append("nested_graph")
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.type.HasField("tensor_type"):
            for dim in value.type.tensor_type.shape.dim:
                if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                    reasons.append("dynamic_shape")
                    break
    return not reasons, sorted(set(reasons))


def exact_known(model: onnx.ModelProto, task: int) -> tuple[bool, int, str | None]:
    try:
        session = scoring._make_raw_session(model)
    except Exception as exc:
        return False, 0, f"session:{type(exc).__name__}"
    if session is None:
        return False, 0, "session:none"
    checked = 0
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = scoring._raw_output(session, benchmark["input"])
            except Exception as exc:
                return False, checked, f"runtime:{type(exc).__name__}"
            checked += 1
            if not np.all(np.isfinite(raw)):
                return False, checked, "nonfinite"
            binary = (raw > 0.0).astype(np.float32)
            if not np.array_equal(binary, benchmark["output"]):
                return False, checked, "mismatch"
            positives = raw[binary > 0]
            near = raw[(raw > 0) & (raw < 0.25)]
            if near.size or (positives.size and float(positives.min()) < 0.25):
                return False, checked, "weak_margin"
    return True, checked, None


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs = authority_costs()
    paths = subprocess.check_output(["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True).splitlines()
    seen = set()
    candidates = []
    for relpath in paths:
        task = task_from_path(relpath)
        if task not in costs:
            continue
        path = ROOT / relpath
        try:
            data = path.read_bytes()
            digest = sha256(data)
            key = (task, digest)
            if key in seen:
                continue
            seen.add(key)
            model = onnx.load_model_from_string(data)
        except Exception:
            continue
        limit = costs[task] // 2
        pcount = params(model)
        lower = declared_lower_bound(model)
        if pcount > limit or lower > limit:
            continue
        safe, reasons = structurally_safe(model)
        candidates.append({
            "task": task, "path": relpath, "sha256": digest,
            "authority_cost": costs[task], "half_limit": limit,
            "params": pcount, "declared_lower_bound": lower,
            "ops": [n.op_type for n in model.graph.node],
            "catalog_monitored": task in PRIVATE_ZERO_OR_UNSOUND,
            "structurally_safe": safe, "structural_reasons": reasons,
        })

    results = []
    for index, row in enumerate(candidates, 1):
        item = dict(row)
        task = int(item["task"])
        if item["catalog_monitored"] or not item["structurally_safe"]:
            item.update({"known_exact": False, "checked": 0, "reject": "catalog_or_structure"})
        else:
            model = onnx.load(ROOT / str(item["path"]))
            exact, checked, reject = exact_known(model, task)
            item.update({"known_exact": exact, "checked": checked, "reject": reject})
            if exact:
                with tempfile.TemporaryDirectory(prefix=f"half303_{task:03d}_", dir="/tmp") as tmp:
                    item["profile"] = scoring.score_and_verify(model, task, tmp, label="history", require_correct=False)
            else:
                item["profile"] = None
        profile = item.get("profile")
        item["winner"] = bool(
            item["known_exact"] and profile and profile["correct"]
            and int(profile["cost"]) <= int(item["half_limit"])
        )
        results.append(item)
        if index % 20 == 0 or item["winner"]:
            print(json.dumps({"i": index, "n": len(candidates), "task": task,
                              "exact": item["known_exact"], "winner": item["winner"]}), flush=True)

    winners = [row for row in results if row["winner"]]
    payload = {
        "authority": AUTHORITY.name,
        "authority_sha256": AUTHORITY_SHA256,
        "scope": "cost 51..100, score<25, target <= floor(authority_cost/2)",
        "task_count": len(costs), "onnx_path_count": len(paths),
        "unique_task_sha_count": len(seen), "candidate_count": len(candidates),
        "winner_count": len(winners), "winners": winners, "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "winners": winners,
                      "evidence": str(EVIDENCE.relative_to(ROOT))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
