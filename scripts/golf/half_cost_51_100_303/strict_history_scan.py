#!/usr/bin/env python3
"""Broaden the archive pass from the half target to every strict reduction."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx

import history_scan as common


ROOT = common.ROOT
OUT = common.HERE / "strict_history_evidence.json"
UNSAFE_OPS = {"TfIdfVectorizer", "CategoryMapper", "FeatureVectorizer", "ZipMap"}


def task_from_member(name: str) -> int | None:
    match = re.search(r"(?:^|/)task[_-]?(\d{3})\.onnx$", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extra_structure(model: onnx.ModelProto) -> list[str]:
    reasons = []
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("noncanonical_io_count")
    else:
        inp = model.graph.input[0]
        out = model.graph.output[0]
        def dims(value):
            return [int(d.dim_value) if d.HasField("dim_value") else None
                    for d in value.type.tensor_type.shape.dim]
        if dims(inp) != [1, 10, 30, 30] or dims(out) != [1, 10, 30, 30]:
            reasons.append("noncanonical_io_shape")
    init = {tensor.name: tensor for tensor in model.graph.initializer}
    for tensor in model.graph.initializer:
        try:
            values = onnx.numpy_helper.to_array(tensor)
            if not np.all(np.isfinite(values)):
                reasons.append("nonfinite_initializer")
        except Exception:
            reasons.append("unreadable_initializer")
    for node in model.graph.node:
        if node.op_type in UNSAFE_OPS or (node.domain and node.domain != ""):
            reasons.append(f"lookup_or_nonstandard:{node.domain}:{node.op_type}")
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3 and node.input[2]:
            weight = init.get(node.input[1])
            bias = init.get(node.input[2])
            if weight is not None and bias is not None and weight.dims and bias.dims:
                channels = int(weight.dims[0] if node.op_type == "Conv" else weight.dims[1])
                if int(bias.dims[0]) != channels:
                    reasons.append("conv_bias_ub")
    return sorted(set(reasons))


def add_candidate(records, seen, task, source, data, costs):
    digest = hashlib.sha256(data).hexdigest()
    key = (task, digest)
    if key in seen:
        return
    seen.add(key)
    try:
        model = onnx.load_model_from_string(data)
    except Exception:
        return
    pcount = common.params(model)
    lower = common.declared_lower_bound(model)
    if pcount >= costs[task] or lower >= costs[task]:
        return
    safe, reasons = common.structurally_safe(model)
    reasons += extra_structure(model)
    records.append({
        "task": task, "source": source, "sha256": digest,
        "authority_cost": costs[task], "params": pcount,
        "declared_lower_bound": lower, "ops": [n.op_type for n in model.graph.node],
        "max_einsum_fanin": max([len(n.input) for n in model.graph.node if n.op_type == "Einsum"] or [0]),
        "catalog_monitored": task in common.PRIVATE_ZERO_OR_UNSOUND,
        "structurally_safe": safe and not reasons,
        "structural_reasons": sorted(set(reasons)), "data": data,
    })


def main() -> int:
    started = time.monotonic()
    if common.sha256(common.AUTHORITY.read_bytes()) != common.AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    costs = common.authority_costs()
    records, seen = [], set()
    loose_paths = subprocess.check_output(["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True).splitlines()
    for relpath in loose_paths:
        task = common.task_from_path(relpath)
        if task in costs:
            try:
                add_candidate(records, seen, task, relpath, (ROOT / relpath).read_bytes(), costs)
            except Exception:
                pass
    zip_paths = subprocess.check_output(["rg", "--files", "-g", "*.zip"], cwd=ROOT, text=True).splitlines()
    member_count = 0
    for relzip in zip_paths:
        try:
            archive = zipfile.ZipFile(ROOT / relzip)
        except Exception:
            continue
        try:
            for name in archive.namelist():
                task = task_from_member(name)
                if task not in costs:
                    continue
                member_count += 1
                try:
                    add_candidate(records, seen, task, f"{relzip}!{name}", archive.read(name), costs)
                except Exception:
                    pass
        finally:
            archive.close()

    results = []
    for index, row in enumerate(records, 1):
        data = row.pop("data")
        item = dict(row)
        task = int(item["task"])
        if item["catalog_monitored"] or not item["structurally_safe"]:
            item.update({"known_exact": False, "checked": 0, "reject": "catalog_or_structure", "profile": None})
        else:
            model = onnx.load_model_from_string(data)
            exact, checked, reject = common.exact_known(model, task)
            item.update({"known_exact": exact, "checked": checked, "reject": reject})
            if exact:
                with tempfile.TemporaryDirectory(prefix=f"strict303_{task:03d}_", dir="/tmp") as tmp:
                    item["profile"] = common.scoring.score_and_verify(
                        model, task, tmp, label="strict_history", require_correct=False
                    )
            else:
                item["profile"] = None
        profile = item["profile"]
        item["strict_lower_known"] = bool(
            item["known_exact"] and profile and profile["correct"]
            and int(profile["cost"]) < int(item["authority_cost"])
        )
        results.append(item)
        if index % 25 == 0 or item["strict_lower_known"]:
            print(json.dumps({"i": index, "n": len(records), "task": task,
                              "known": item["known_exact"], "lower": item["strict_lower_known"]}), flush=True)

    winners = [row for row in results if row["strict_lower_known"]]
    payload = {
        "authority": common.AUTHORITY.name, "authority_sha256": common.AUTHORITY_SHA256,
        "scope": "all loose and ZIP history, cost51..100, every strict reduction",
        "loose_path_count": len(loose_paths), "zip_count": len(zip_paths),
        "target_zip_member_count": member_count, "unique_task_sha_count": len(seen),
        "theoretical_candidate_count": len(records), "known_strict_lower_count": len(winners),
        "known_strict_lower": winners, "results": results,
        "elapsed_seconds": time.monotonic() - started, "protected_writes": "none",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(records), "known_strict_lower": winners,
                      "evidence": str(OUT.relative_to(ROOT))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
