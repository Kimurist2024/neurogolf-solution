#!/usr/bin/env python3
"""Reprofile 8023.08 low-cost whites and apply semantics-preserving optimizers."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8023.08.zip"
AUTHORITY_SHA = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
TASKS = (62, 156, 238, 297, 324, 341, 398)
BASE = HERE / "base"
CANDIDATES = HERE / "optimizer_candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


PASSES = (
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_common_subexpression",
    "eliminate_nop_cast",
    "eliminate_nop_flatten",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_nop_transpose",
    "eliminate_nop_reshape",
    "eliminate_nop_with_unit",
    "eliminate_shape_gather",
    "eliminate_slice_after_shape",
    "eliminate_shape_op",
    "eliminate_consecutive_idempotent_ops",
    "fuse_consecutive_concats",
    "fuse_consecutive_slices",
    "fuse_consecutive_transposes",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_unsqueezes",
    "extract_constant_to_initializer",
)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"opt8023_{task:03d}_{label}_", dir="/tmp") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def static(model: onnx.ModelProto) -> dict[str, object]:
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception as exc:
        return {"full_check": False, "strict_data_prop": False,
                "error": f"{type(exc).__name__}: {exc}"}
    inputs = []
    outputs = []
    for value, target in ((item, inputs) for item in inferred.graph.input):
        target.append([int(dim.dim_value) if dim.HasField("dim_value") else 0
                       for dim in value.type.tensor_type.shape.dim])
    for value in inferred.graph.output:
        outputs.append([int(dim.dim_value) if dim.HasField("dim_value") else 0
                        for dim in value.type.tensor_type.shape.dim])
    nonstatic = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        if any(not dim.HasField("dim_value") or int(dim.dim_value) <= 0
               for dim in value.type.tensor_type.shape.dim):
            nonstatic.append(value.name)
    return {"full_check": True, "strict_data_prop": True, "inputs": inputs,
            "outputs": outputs, "nonstatic": nonstatic}


def optimize_fixed(model: onnx.ModelProto, passes: list[str], rounds: int = 1) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    for _ in range(rounds):
        before = result.SerializeToString()
        result = onnxoptimizer.optimize(result, passes)
        if result.SerializeToString() == before:
            break
    return result


def main() -> int:
    authority_blob = AUTHORITY.read_bytes()
    if sha256(authority_blob) != AUTHORITY_SHA:
        raise RuntimeError("8023.08 authority drift")
    BASE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY, "r") as archive:
        sources = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}

    result: dict[str, object] = {"authority": str(AUTHORITY.relative_to(ROOT)),
                                "authority_sha256": AUTHORITY_SHA, "tasks": {}}
    combo_passes = list(PASSES)
    for task in TASKS:
        blob = sources[task]
        base_path = BASE / f"task{task:03d}.onnx"
        base_path.write_bytes(blob)
        model = onnx.load_model_from_string(blob)
        normalized = copy.deepcopy(model)
        # Archived candidates can retain stale intermediate declarations even
        # when the runtime graph is correct.  Value-info is non-semantic; drop
        # it so strict inference reconstructs authoritative tensor shapes.
        del normalized.graph.value_info[:]
        base_profile = profile(model, task, "base")
        task_row: dict[str, object] = {
            "source_sha256": sha256(blob),
            "source_profile": base_profile,
            "source_static": static(model),
            "source_nodes": len(model.graph.node),
            "source_initializers": len(model.graph.initializer),
            "variants": [],
        }
        specs: list[tuple[str, list[str], int]] = [
            (name, [name], 1) for name in PASSES
        ] + [
            ("safe_combo", combo_passes, 1),
            ("safe_combo_fixed4", combo_passes, 4),
        ]
        seen = {sha256(blob)}
        for label, passes, rounds in specs:
            row: dict[str, object] = {"label": label, "passes": passes, "rounds": rounds}
            try:
                candidate = optimize_fixed(normalized, passes, rounds)
                candidate.producer_name = f"codex-8023-task{task:03d}-{label}"
                candidate_blob = candidate.SerializeToString()
                digest = sha256(candidate_blob)
                row["sha256"] = digest
                row["changed"] = candidate_blob != blob
                row["nodes"] = len(candidate.graph.node)
                row["initializers"] = len(candidate.graph.initializer)
                row["static"] = static(candidate)
                row["profile"] = profile(candidate, task, label)
                row["profile_valid"] = all(
                    int(row["profile"][key]) >= 0
                    for key in ("memory", "params", "cost")
                )
                structural = row["static"]
                row["strict_lower"] = bool(
                    row["profile_valid"]
                    and structural.get("full_check") is True
                    and structural.get("strict_data_prop") is True
                    and not structural.get("nonstatic")
                    and row["profile"]["cost"] < base_profile["cost"]
                )
                if row["strict_lower"] and digest not in seen:
                    path = CANDIDATES / f"task{task:03d}_{label}_cost{row['profile']['cost']}.onnx"
                    path.write_bytes(candidate_blob)
                    row["path"] = str(path.relative_to(ROOT))
                    seen.add(digest)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            task_row["variants"].append(row)
        result["tasks"][str(task)] = task_row
        winners = [row for row in task_row["variants"] if row.get("strict_lower")]
        print(json.dumps({"task": task, "base": base_profile,
                          "strict_lower": winners}, default=str), flush=True)
    (HERE / "optimizer_scan.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
