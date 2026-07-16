#!/usr/bin/env python3
"""Mine every repository ZIP for structurally cheaper task models.

This is a scratch-only discovery pass.  It uses the exact 7999.13 cost ledger
only as a prefilter, writes candidates below this directory, and deliberately
does not promote anything.  Correctness, fresh-generator, default-ORT, and
archive gates remain mandatory after this scan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_7999.13.zip"
SCORES = ROOT / "all_scores.csv"
OUT = HERE / "lane_archive_zip_sweep"
MEMBER = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)
LOOSE = re.compile(r"^task(\d{3})(?:[^0-9].*)?\.onnx$", re.IGNORECASE)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost)) if cost else 25.0


def load_costs() -> dict[int, int]:
    result: dict[int, int] = {}
    with SCORES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["task"].removeprefix("task"))] = int(row["cost"])
    return result


def static_cost(model: onnx.ModelProto) -> int | None:
    """Return a strict inferred-shape cost floor for honest/static graphs."""
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception:
        return None
    initializers = {item.name for item in inferred.graph.initializer}
    params = sum(int(np.prod(item.dims, dtype=np.int64)) for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.name in seen or value.name in initializers or value.name in {"input", "output"}:
            continue
        seen.add(value.name)
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return None
        elements = 1
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            elements *= int(dim.dim_value)
        try:
            itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        except Exception:
            return None
        memory += elements * itemsize
    return int(params + memory)


def structural_reason(model: onnx.ModelProto) -> str | None:
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        return f"checker:{type(exc).__name__}"
    if model.functions or model.graph.sparse_initializer:
        return "functions_or_sparse"
    domains = {item.domain for item in model.opset_import}
    if not domains <= {"", "ai.onnx"}:
        return "custom_domain"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED or "Sequence" in node.op_type:
            return f"banned:{node.op_type}"
        for attribute in node.attribute:
            if attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return "nested_graph"
    return None


def zip_paths() -> list[Path]:
    excluded_parts = {".git", ".venv", "node_modules", "lane_archive_zip_sweep"}
    return sorted(
        path for path in ROOT.rglob("*.zip")
        if not any(part in excluded_parts for part in path.parts)
        and path.resolve() != BASE.resolve()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=100, help="scan the N highest-cost tasks")
    parser.add_argument("--keep", type=int, default=12, help="retain this many static winners per task")
    parser.add_argument("--tasks", help="comma-separated explicit task IDs")
    parser.add_argument("--include-loose", action="store_true")
    parser.add_argument("--skip-zips", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    costs = load_costs()
    if args.tasks:
        targets = {int(item) for item in args.tasks.split(",") if item.strip()}
    else:
        targets = set(sorted(costs, key=costs.get, reverse=True)[: args.top])

    with zipfile.ZipFile(BASE) as archive:
        base_hashes = {task: digest(archive.read(f"task{task:03d}.onnx")) for task in targets}

    unique: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    stats: defaultdict[str, int] = defaultdict(int)
    errors: list[dict[str, str]] = []
    paths = [] if args.skip_zips else zip_paths()
    for index, path in enumerate(paths, 1):
        stats["zips_seen"] += 1
        try:
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    match = MEMBER.search(name)
                    if not match:
                        continue
                    task = int(match.group(1))
                    if task not in targets:
                        continue
                    stats["members_seen"] += 1
                    data = archive.read(name)
                    sha = digest(data)
                    if sha == base_hashes[task]:
                        stats["baseline_duplicates"] += 1
                        continue
                    entry = unique[task].setdefault(
                        sha,
                        {"data": data, "sha256": sha, "sources": []},
                    )
                    entry["sources"].append(f"{path.relative_to(ROOT)}::{name}")
        except Exception as exc:
            errors.append({"zip": str(path.relative_to(ROOT)), "error": repr(exc)})
        if index % 100 == 0:
            print(f"inventory {index}/{len(paths)} unique={sum(map(len, unique.values()))}", flush=True)

    if args.include_loose:
        excluded_parts = {".git", ".venv", "node_modules", args.out_dir.name}
        loose_paths = (
            path for path in ROOT.rglob("*.onnx")
            if not any(part in excluded_parts for part in path.parts)
        )
        for index, path in enumerate(loose_paths, 1):
            match = LOOSE.match(path.name)
            if not match:
                continue
            task = int(match.group(1))
            if task not in targets:
                continue
            stats["loose_seen"] += 1
            try:
                data = path.read_bytes()
                sha = digest(data)
                if sha == base_hashes[task]:
                    stats["baseline_duplicates"] += 1
                    continue
                entry = unique[task].setdefault(
                    sha,
                    {"data": data, "sha256": sha, "sources": []},
                )
                entry["sources"].append(str(path.relative_to(ROOT)))
            except Exception as exc:
                errors.append({"loose": str(path.relative_to(ROOT)), "error": repr(exc)})
            if index % 10000 == 0:
                print(f"loose inventory {index} unique={sum(map(len, unique.values()))}", flush=True)

    stats["unique_different"] = sum(map(len, unique.values()))
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("task*.onnx"):
        old.unlink()

    retained: dict[str, list[dict[str, object]]] = {}
    for task in sorted(targets, key=lambda item: costs[item], reverse=True):
        ranked: list[tuple[int, dict[str, object]]] = []
        for entry in unique.get(task, {}).values():
            data = entry["data"]
            try:
                model = onnx.load_model_from_string(data)
            except Exception:
                stats["parse_rejects"] += 1
                continue
            reason = structural_reason(model)
            if reason:
                stats["structure_rejects"] += 1
                continue
            candidate_cost = static_cost(model)
            if candidate_cost is None:
                stats["static_unknown"] += 1
                continue
            if candidate_cost >= costs[task]:
                stats["static_not_cheaper"] += 1
                continue
            ranked.append((candidate_cost, entry))

        ranked.sort(key=lambda item: (item[0], item[1]["sha256"]))
        task_rows: list[dict[str, object]] = []
        for rank, (candidate_cost, entry) in enumerate(ranked[: args.keep], 1):
            output = out_dir / f"task{task:03d}_r{rank:02d}_static{candidate_cost}.onnx"
            output.write_bytes(entry["data"])
            task_rows.append(
                {
                    "task": task,
                    "baseline_cost": costs[task],
                    "static_cost": candidate_cost,
                    "static_gain": score(candidate_cost) - score(costs[task]),
                    "sha256": entry["sha256"],
                    "path": str(output.relative_to(ROOT)),
                    "sources": entry["sources"][:20],
                    "source_count": len(entry["sources"]),
                }
            )
        if task_rows:
            retained[str(task)] = task_rows
            print(
                f"task{task:03d}: unique={len(unique.get(task, {}))} "
                f"retained={len(task_rows)} best_static={task_rows[0]['static_cost']} "
                f"base={costs[task]}",
                flush=True,
            )

    report = {
        "base": str(BASE.relative_to(ROOT)),
        "targets": sorted(targets),
        "stats": dict(stats),
        "zip_errors": errors,
        "retained": retained,
    }
    (out_dir / "inventory.json").write_text(json.dumps(report, indent=2))
    print(
        f"done zips={stats['zips_seen']} unique={stats['unique_different']} "
        f"tasks_with_leads={len(retained)} retained={sum(map(len, retained.values()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
