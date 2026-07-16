#!/usr/bin/env python3
"""Sweep conservative onnxoptimizer passes against an immutable baseline ZIP.

The sweep only discovers lower-cost candidates.  It deliberately does not
promote anything: checker, known examples, domain-fresh cases, bias audit and
final ZIP validation remain mandatory downstream gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxoptimizer
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.rank_dir import cost_of


PASSES = [
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_consecutive_idempotent_ops",
    "eliminate_if_with_const_cond",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_shape_gather",
    "eliminate_slice_after_shape",
    "eliminate_nop_transpose",
    "fuse_add_bias_into_conv",
    "fuse_bn_into_conv",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
    "fuse_pad_into_pool",
    "fuse_transpose_into_gemm",
    "fuse_concat_into_reshape",
    "eliminate_nop_reshape",
    "eliminate_nop_with_unit",
    "eliminate_common_subexpression",
    "fuse_qkv",
    "fuse_consecutive_unsqueezes",
    "eliminate_duplicate_initializer",
    "rewrite_where",
    "eliminate_identity",
    "fuse_consecutive_slices",
    "eliminate_shape_op",
    "eliminate_unused_initializer",
]


def validate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("functions/sparse initializers are forbidden")
    banned = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
    for node in model.graph.node:
        if node.op_type in banned or "Sequence" in node.op_type:
            raise ValueError(f"banned op {node.op_type}")


def measure(model: onnx.ModelProto, task: int) -> int:
    with tempfile.TemporaryDirectory(prefix=f"ngolf_opt_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        return int(cost_of(str(path))[2])


def parse_tasks(value: str) -> list[int]:
    tasks: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            tasks.update(range(start, end + 1))
        elif part.strip():
            tasks.add(int(part))
    return sorted(task for task in tasks if 1 <= task <= 400)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="1-400")
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--pass-only", choices=PASSES)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.base_costs.read_text())["costs"]
    ort.set_default_logger_severity(3)
    winners: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in parse_tasks(args.tasks):
            original = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            original_bytes = original.SerializeToString()
            cost_entry = costs.get(str(task))
            if cost_entry is None:
                try:
                    base_cost = measure(original, task)
                except Exception as exc:
                    failures.append({
                        "task": task,
                        "recipe": "baseline_cost",
                        "error": repr(exc),
                    })
                    continue
            else:
                base_cost = int(cost_entry["cost"] if isinstance(cost_entry, dict) else cost_entry)
            best: tuple[int, str, onnx.ModelProto] | None = None
            if args.pass_only:
                recipes = [(args.pass_only, [args.pass_only])]
            else:
                recipes = [("all_conservative", PASSES)]
            if not args.combined_only and not args.pass_only:
                recipes = [(name, [name]) for name in PASSES] + recipes
            for recipe_name, passes in recipes:
                try:
                    candidate = onnxoptimizer.optimize(original, passes)
                    if candidate.SerializeToString() == original_bytes:
                        continue
                    validate(candidate)
                    candidate_cost = measure(candidate, task)
                    if candidate_cost < base_cost and (best is None or candidate_cost < best[0]):
                        best = (candidate_cost, recipe_name, candidate)
                except Exception as exc:
                    failures.append({"task": task, "recipe": recipe_name, "error": repr(exc)})
            if best is None:
                continue
            candidate_cost, recipe_name, candidate = best
            path = args.out_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, path)
            item = {
                "task": task,
                "path": str(path),
                "baseline_cost": base_cost,
                "candidate_cost": candidate_cost,
                "projected_gain": math.log(base_cost / candidate_cost),
                "recipe": recipe_name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            winners.append(item)
            print(f"task{task:03d}: {base_cost}->{candidate_cost} {recipe_name}")

    payload = {
        "baseline": str(args.baseline),
        "tasks": args.tasks,
        "passes": PASSES,
        "winners": winners,
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
        "failures": failures,
    }
    manifest = args.out_dir / "manifest_pre_differential.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"], "failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
