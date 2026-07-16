#!/usr/bin/env python3
"""Try conservative, input-independent ONNX optimizer passes on task285 SOUND.

The script writes only into this lane.  Each emitted model must pass the ONNX
checker and strict shape inference before it is measured by the official cost
implementation.  Semantic admission is deliberately left to audit_final.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_deep68/candidates/task285_true_dedup.onnx"
OUT = HERE / "candidates"

# These passes are graph identities; they do not use examples, coordinates,
# colors, or inferred task labels.  More aggressive algebraic rewrites are not
# considered here because integer overflow/rounding would need separate proof.
PASSES = [
    "eliminate_identity",
    "eliminate_deadend",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_nop_cast",
    "eliminate_nop_concat",
    "eliminate_nop_reshape",
    "eliminate_nop_transpose",
    "eliminate_nop_with_unit",
    "eliminate_consecutive_idempotent_ops",
    "fuse_consecutive_concats",
    "fuse_consecutive_slices",
    "fuse_consecutive_transposes",
    "fuse_consecutive_unsqueezes",
    "fuse_consecutive_reduce_unsqueeze",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def emit(model: onnx.ModelProto, name: str) -> dict[str, object]:
    path = OUT / f"{name}.onnx"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    onnx.save(model, path)
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        **measure(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = onnx.load(SOURCE)
    rows: list[dict[str, object]] = []
    base = emit(copy.deepcopy(source), "task285_sound_dedup_copy")
    base["passes"] = []
    rows.append(base)

    for optimizer_pass in PASSES:
        try:
            row = emit(onnxoptimizer.optimize(copy.deepcopy(source), [optimizer_pass]), f"task285_{optimizer_pass}")
            row["passes"] = [optimizer_pass]
            rows.append(row)
        except Exception as error:  # fail closed and retain the diagnostic
            rows.append({"name": optimizer_pass, "passes": [optimizer_pass], "error": repr(error)})

    combinations = {
        "task285_safe_eliminate": [
            "eliminate_identity",
            "eliminate_deadend",
            "eliminate_unused_initializer",
            "eliminate_duplicate_initializer",
            "eliminate_nop_cast",
            "eliminate_nop_concat",
            "eliminate_nop_reshape",
            "eliminate_nop_transpose",
        ],
        "task285_safe_all": PASSES,
    }
    for name, optimizer_passes in combinations.items():
        try:
            row = emit(onnxoptimizer.optimize(copy.deepcopy(source), optimizer_passes), name)
            row["passes"] = optimizer_passes
            rows.append(row)
        except Exception as error:
            rows.append({"name": name, "passes": optimizer_passes, "error": repr(error)})

    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        if "sha256" in row:
            unique.setdefault(str(row["sha256"]), row)
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "source_cost": measure(SOURCE),
        "rows": rows,
        "unique_models": sorted(unique.values(), key=lambda item: int(item["cost"])),
    }
    (HERE / "sweep_exact_passes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["unique_models"], indent=2))


if __name__ == "__main__":
    main()
