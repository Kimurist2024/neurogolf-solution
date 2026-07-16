#!/usr/bin/env python3
"""Deduplicate and statically screen every loose high47 history model.

The earlier all-400 archive already covers ZIP members.  This pass re-scans
all loose models currently in the worktree so models created after that
archive are not missed.  It only copies numeric lower leads into this lane;
the isolated runtime auditor performs the decisive cost/correctness checks.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (44, 12, 198, 277, 117, 270, 19, 62)
BASE_COST = {44: 1086, 12: 710, 198: 661, 277: 631, 117: 606, 270: 594, 19: 536, 62: 465}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def static_cost(model: onnx.ModelProto) -> dict[str, int] | None:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        inferred = model
    infos = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    free = {item.name for item in inferred.graph.input}
    free.update(item.name for item in inferred.graph.output)
    free.update(item.name for item in inferred.graph.initializer)
    seen: set[str] = set()
    memory = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in seen or name in free:
                continue
            seen.add(name)
            value = infos.get(name)
            dims = tensor_shape(value) if value is not None else None
            if dims is None:
                return None
            try:
                dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            except Exception:
                return None
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structure(model: onnx.ModelProto) -> dict[str, object]:
    ops = collections.Counter(node.op_type for node in model.graph.node)
    giant = [
        {"index": index, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Einsum" and len(node.input) > 16
    ]
    lookup = [
        node.op_type
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax"}
    ]
    strict = True
    strict_error = None
    try:
        shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
    return {
        "ops": dict(sorted(ops.items())),
        "max_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "giant_einsum": giant,
        "lookup": lookup,
        "banned": sorted(
            {
                node.op_type
                for node in model.graph.node
                if node.op_type in BANNED or "Sequence" in node.op_type
            }
        ),
        "strict_data_prop": strict,
        "strict_error": strict_error,
    }


def main() -> None:
    baselines = {task: sha(HERE / "baselines" / f"task{task:03d}.onnx") for task in TARGETS}
    rows: dict[int, dict[str, dict[str, object]]] = {task: {} for task in TARGETS}
    observations = collections.Counter()

    for task in TARGETS:
        patterns = (f"task{task:03d}*.onnx", f"*task{task:03d}*.onnx")
        paths: set[Path] = set()
        for pattern in patterns:
            paths.update(ROOT.rglob(pattern))
        for path in sorted(paths):
            if HERE in path.parents:
                continue
            observations[task] += 1
            digest = sha(path)
            if digest == baselines[task]:
                continue
            relative = str(path.relative_to(ROOT))
            if digest in rows[task]:
                sources = rows[task][digest]["sources"]
                assert isinstance(sources, list)
                if len(sources) < 20:
                    sources.append(relative)
                rows[task][digest]["source_count"] = int(rows[task][digest]["source_count"]) + 1
                continue
            record: dict[str, object] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
                "sources": [relative],
                "source_count": 1,
            }
            try:
                model = onnx.load(path)
                record["static_cost"] = static_cost(model)
                record["structure"] = structure(model)
            except Exception as exc:
                record["parse_error"] = f"{type(exc).__name__}: {exc}"
            rows[task][digest] = record

    retained: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for task in TARGETS:
        unique = list(rows[task].values())
        lower = [
            row
            for row in unique
            if isinstance(row.get("static_cost"), dict)
            and int(row["static_cost"]["cost"]) < BASE_COST[task]
        ]
        lower.sort(key=lambda row: (int(row["static_cost"]["cost"]), str(row["sha256"])))
        for ordinal, row in enumerate(lower, 1):
            source = ROOT / str(row["sources"][0])
            destination = HERE / "candidates" / (
                f"task{task:03d}_history_r{ordinal:02d}_static{row['static_cost']['cost']}_"
                f"{str(row['sha256'])[:10]}.onnx"
            )
            shutil.copyfile(source, destination)
            retained.append(
                {
                    "task": task,
                    "baseline_cost": BASE_COST[task],
                    "candidate": str(destination.relative_to(ROOT)),
                    **row,
                }
            )
        summary.append(
            {
                "task": task,
                "observations": observations[task],
                "unique_nonbaseline": len(unique),
                "strict_numeric_lower": len(lower),
            }
        )

    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    output = {
        "baseline": "submission_base_8005.16.zip",
        "baseline_sha256": hashlib.sha256((ROOT / "submission_base_8005.16.zip").read_bytes()).hexdigest(),
        "all400_prior_stats": archive["stats"],
        "loose_current_summary": summary,
        "retained": retained,
        "retained_count": len(retained),
    }
    (HERE / "history_inventory.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"retained={len(retained)}")


if __name__ == "__main__":
    main()
