#!/usr/bin/env python3
"""Rebase every local loose/ZIP model for current cost 101..250 against 8011.05.

Evidence-only lane: this never modifies the authority submission or CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
OUT = HERE / "history_evidence.json"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
UNSAFE_OPS = {"TfIdfVectorizer", "CategoryMapper", "FeatureVectorizer", "ZipMap"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_from_source(name: str) -> int | None:
    match = re.search(r"task[_-]?(\d{3})(?!\d)", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def authority_costs() -> dict[int, int]:
    result = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if 101 <= cost <= 250 and float(row["score"]) < 25.0:
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


def structural_reasons(model: onnx.ModelProto) -> list[str]:
    reasons = []
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializer")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("noncanonical_io_count")
    else:
        def dims(value):
            return [int(d.dim_value) if d.HasField("dim_value") else None
                    for d in value.type.tensor_type.shape.dim]
        if dims(model.graph.input[0]) != [1, 10, 30, 30]:
            reasons.append("noncanonical_input_shape")
        if dims(model.graph.output[0]) != [1, 10, 30, 30]:
            reasons.append("noncanonical_output_shape")
    init = {tensor.name: tensor for tensor in model.graph.initializer}
    for tensor in model.graph.initializer:
        try:
            if not np.all(np.isfinite(onnx.numpy_helper.to_array(tensor))):
                reasons.append("nonfinite_initializer")
        except Exception:
            reasons.append("unreadable_initializer")
    for node in model.graph.node:
        if node.op_type in BANNED or "Sequence" in node.op_type:
            reasons.append(f"banned:{node.op_type}")
        if node.op_type in UNSAFE_OPS or (node.domain and node.domain != ""):
            reasons.append(f"lookup_or_nonstandard:{node.domain}:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                reasons.append("nested_graph")
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3 and node.input[2]:
            weight = init.get(node.input[1])
            bias = init.get(node.input[2])
            if weight is not None and bias is not None and weight.dims and bias.dims:
                channels = int(weight.dims[0] if node.op_type == "Conv" else weight.dims[1])
                if int(bias.dims[0]) != channels:
                    reasons.append("conv_bias_ub")
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if not value.type.HasField("tensor_type"):
            continue
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                reasons.append("dynamic_shape")
                break
    return sorted(set(reasons))


def known_exact(model: onnx.ModelProto, task: int) -> tuple[bool, int, str | None]:
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
            if np.any((raw > 0) & (raw < 0.25)):
                return False, checked, "weak_margin"
    return True, checked, None


def add_candidate(records, seen, task, source, data, costs, *, strict: bool):
    digest = sha256(data)
    key = (task, digest)
    if key in seen:
        return
    seen.add(key)
    try:
        model = onnx.load_model_from_string(data)
    except Exception:
        return
    limit = costs[task] - 1 if strict else costs[task] // 2
    pcount = params(model)
    lower = declared_lower_bound(model)
    if pcount > limit or lower > limit:
        return
    records.append({
        "task": task, "source": source, "sha256": digest,
        "authority_cost": costs[task], "half_limit": limit,
        "params": pcount, "declared_lower_bound": lower,
        "ops": [node.op_type for node in model.graph.node],
        "max_einsum_fanin": max([len(node.input) for node in model.graph.node
                                  if node.op_type == "Einsum"] or [0]),
        "structural_reasons": structural_reasons(model), "data": data,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="scan every strict reduction, not only <=half")
    args = parser.parse_args()
    started = time.monotonic()
    onnxruntime.set_default_logger_severity(3)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs = authority_costs()
    records, seen = [], set()
    loose_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True
    ).splitlines()
    for relpath in loose_paths:
        task = task_from_source(relpath)
        if task in costs:
            try:
                add_candidate(records, seen, task, relpath, (ROOT / relpath).read_bytes(), costs,
                              strict=args.strict)
            except Exception:
                pass
    zip_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.zip"], cwd=ROOT, text=True
    ).splitlines()
    member_count = 0
    for zindex, relzip in enumerate(zip_paths, 1):
        try:
            archive = zipfile.ZipFile(ROOT / relzip)
        except Exception:
            continue
        try:
            for name in archive.namelist():
                task = task_from_source(name)
                if task not in costs or not name.lower().endswith(".onnx"):
                    continue
                member_count += 1
                try:
                    add_candidate(records, seen, task, f"{relzip}!{name}", archive.read(name), costs,
                                  strict=args.strict)
                except Exception:
                    pass
        finally:
            archive.close()
        if zindex % 100 == 0:
            print(json.dumps({"zips": zindex, "records": len(records), "seen": len(seen)}), flush=True)

    results = []
    for index, row in enumerate(records, 1):
        data = row.pop("data")
        item = dict(row)
        task = int(item["task"])
        if item["structural_reasons"]:
            item.update({"known_exact": False, "checked": 0,
                         "reject": "structure", "profile": None})
        else:
            model = onnx.load_model_from_string(data)
            exact, checked, reject = known_exact(model, task)
            item.update({"known_exact": exact, "checked": checked, "reject": reject})
            if exact:
                with tempfile.TemporaryDirectory(prefix=f"hist307_{task:03d}_", dir="/tmp") as tmp:
                    try:
                        item["profile"] = scoring.score_and_verify(
                            model, task, tmp, label="history", require_correct=False
                        )
                    except Exception as exc:
                        item["profile"] = None
                        item["reject"] = f"profile:{type(exc).__name__}"
            else:
                item["profile"] = None
        profile = item["profile"]
        item["winner"] = bool(
            item["known_exact"] and profile and profile["correct"]
            and int(profile["cost"]) <= int(item["half_limit"])
        )
        results.append(item)
        if index % 25 == 0 or item["winner"]:
            print(json.dumps({"i": index, "n": len(records), "task": task,
                              "exact": item["known_exact"], "winner": item["winner"]}), flush=True)

    winners = [row for row in results if row["winner"]]
    payload = {
        "authority": AUTHORITY.name, "authority_sha256": AUTHORITY_SHA256,
        "scope": ("all loose and ZIP history, authority cost101..250, strict reduction"
                  if args.strict else
                  "all loose and ZIP history, authority cost101..250, <=floor(cost/2)"),
        "task_count": len(costs), "loose_path_count": len(loose_paths),
        "zip_count": len(zip_paths), "target_zip_member_count": member_count,
        "unique_task_sha_count": len(seen), "candidate_count": len(records),
        "winner_count": len(winners), "winners": winners, "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only scripts/golf/cost101_250_half_307",
    }
    out = HERE / ("strict_history_evidence.json" if args.strict else "history_evidence.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(records), "winner_count": len(winners),
                      "winners": winners, "evidence": str(out.relative_to(ROOT))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
