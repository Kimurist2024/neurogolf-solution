#!/usr/bin/env python3
"""Dedupe and screen every local B11 ONNX artifact against exact-zip costs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


BASE_COST = {264: 362, 281: 161, 300: 182, 358: 161, 376: 158, 387: 337, 392: 345}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def task_for_path(path: Path) -> int | None:
    match = re.match(r"^task[_-]?(\d{1,3})(?!\d)", path.name)
    if match:
        return int(match.group(1))
    matches: list[int] = []
    for part in path.parts:
        match = re.fullmatch(r"task[_-]?(\d{1,3})", part)
        if match:
            matches.append(int(match.group(1)))
    return matches[-1] if matches else None


def declared_memory_lower_bound(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    io_names = {item.name for item in list(inferred.graph.input) + list(inferred.graph.output)}
    total = 0
    for item in inferred.graph.value_info:
        if item.name in io_names:
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            raise ValueError(f"unshaped tensor {item.name}")
        dimensions: list[int] = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise ValueError(f"dynamic tensor {item.name}")
            dimensions.append(dimension.dim_value)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        total += math.prod(dimensions) * np.dtype(dtype).itemsize
    return total


def main() -> None:
    by_task: dict[int, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in ROOT.rglob("*.onnx"):
        task = task_for_path(path)
        if task not in BASE_COST:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_task[task][digest].append(path)

    result: dict[str, object] = {}
    for task in sorted(BASE_COST):
        baseline_digest = hashlib.sha256(
            (HERE / "baseline" / f"task{task:03d}.onnx").read_bytes()
        ).hexdigest()
        rows: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        for digest, paths in by_task[task].items():
            path = min(paths, key=lambda item: (len(str(item)), str(item)))
            try:
                model = onnx.load(path)
                params = scoring.calculate_params(model)
                if params is None:
                    raise ValueError("parameter count unavailable")
                lower_memory = declared_memory_lower_bound(model)
                lower_cost = params + lower_memory
                if lower_cost >= BASE_COST[task]:
                    continue
                ops = [node.op_type for node in model.graph.node]
                giant_einsum = any(
                    node.op_type == "Einsum" and len(node.input) >= 8
                    for node in model.graph.node
                )
                lookup = any(op in {"TfIdfVectorizer"} for op in ops)
                banned = sorted(
                    {
                        op
                        for op in ops
                        if op.upper() in BANNED or "Sequence" in op
                    }
                )
                nonstandard_domains = sorted(
                    {
                        item.domain
                        for item in model.opset_import
                        if item.domain not in {"", "ai.onnx"}
                    }
                )
                red_flags: list[str] = []
                if giant_einsum:
                    red_flags.append("giant_einsum")
                if lookup:
                    red_flags.append("lookup")
                if banned:
                    red_flags.append("banned_op")
                if nonstandard_domains:
                    red_flags.append("nonstandard_domain")
                if model.functions:
                    red_flags.append("functions")
                if model.graph.sparse_initializer:
                    red_flags.append("sparse_initializer")
                checker_full = True
                checker_error = None
                try:
                    onnx.checker.check_model(model, full_check=True)
                except Exception as exc:  # noqa: BLE001
                    checker_full = False
                    checker_error = f"{type(exc).__name__}: {exc}"
                row: dict[str, object] = {
                    "sha256": digest,
                    "is_exact_baseline": digest == baseline_digest,
                    "representative_path": str(path.relative_to(ROOT)),
                    "duplicate_paths": len(paths),
                    "nodes": len(model.graph.node),
                    "params": params,
                    "declared_memory_lower_bound": lower_memory,
                    "declared_cost_lower_bound": lower_cost,
                    "checker_full": checker_full,
                    "red_flags": red_flags,
                    "giant_einsum": giant_einsum,
                    "lookup": lookup,
                    "banned_ops": banned,
                    "nonstandard_domains": nonstandard_domains,
                }
                if checker_error:
                    row["checker_error"] = checker_error
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "sha256": digest,
                        "path": str(path.relative_to(ROOT)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        rows.sort(
            key=lambda item: (
                int(item["declared_cost_lower_bound"]),
                str(item["representative_path"]),
            )
        )
        result[str(task)] = {
            "baseline_cost": BASE_COST[task],
            "baseline_sha256": baseline_digest,
            "unique_local_models": len(by_task[task]),
            "models_with_declared_lower_bound_below_baseline": rows,
            "screen_errors": errors,
        }
        print(task, len(by_task[task]), len(rows), len(errors), flush=True)
    (HERE / "history_scan.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
