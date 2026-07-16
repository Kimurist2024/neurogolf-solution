#!/usr/bin/env python3
"""Mechanical exact-pass scan for the assigned immutable authority payloads."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxoptimizer

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = HERE / "base"
OUT = HERE / "candidates" / "optimizer"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

PASS_SETS = {
    "matmul_add_gemm": ["fuse_matmul_add_bias_into_gemm"],
    "transpose_gemm": ["fuse_transpose_into_gemm"],
    "conv_add_bias": ["fuse_add_bias_into_conv"],
    "bn_conv": ["fuse_bn_into_conv"],
    "pad_conv": ["fuse_pad_into_conv"],
    "pad_pool": ["fuse_pad_into_pool"],
    "concat_reshape": ["fuse_concat_into_reshape"],
    "qkv": ["fuse_qkv"],
    "cse": ["eliminate_common_subexpression"],
    "idempotent": ["eliminate_consecutive_idempotent_ops"],
    "monotone_argmax": ["eliminate_nop_monotone_argmax"],
    "shape_gather": ["eliminate_shape_gather"],
    "slice_after_shape": ["eliminate_slice_after_shape"],
    "shape_op": ["eliminate_shape_op"],
    "consecutive_slices": ["fuse_consecutive_slices"],
    "reduce_unsqueeze": ["fuse_consecutive_reduce_unsqueeze"],
    "einsum_matmul": ["replace_einsum_with_matmul"],
    "slice_matmul": ["adjust_slice_and_matmul"],
    "rewrite_where": ["rewrite_where"],
    "adjust_add": ["adjust_add"],
    "cleanup_dce": ["eliminate_deadend"],
    "cleanup_initializer_aliases": ["eliminate_duplicate_initializer", "eliminate_unused_initializer"],
    "cleanup_noops": [
        "eliminate_identity", "eliminate_nop_cast", "eliminate_nop_concat",
        "eliminate_nop_dropout", "eliminate_nop_expand", "eliminate_nop_flatten",
        "eliminate_nop_pad", "eliminate_nop_reshape", "eliminate_nop_split",
        "eliminate_nop_transpose", "eliminate_nop_with_unit",
    ],
    "cleanup_all": [
        "eliminate_deadend", "eliminate_duplicate_initializer",
        "eliminate_unused_initializer", "eliminate_identity", "eliminate_nop_cast",
        "eliminate_nop_concat", "eliminate_nop_dropout", "eliminate_nop_expand",
        "eliminate_nop_flatten", "eliminate_nop_pad", "eliminate_nop_reshape",
        "eliminate_nop_split", "eliminate_nop_transpose", "eliminate_nop_with_unit",
        "eliminate_common_subexpression", "eliminate_consecutive_idempotent_ops",
    ],
}


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"tasks_scan_{task:03d}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(BASE.glob("task*.onnx")):
        task = int(path.stem.removeprefix("task"))
        raw = path.read_bytes()
        model = onnx.load_model_from_string(raw)
        baseline = profile(model, task)
        for label, passes in PASS_SETS.items():
            row = {"task": task, "pass_set": label, "baseline": baseline}
            try:
                candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
                row["changed"] = candidate.SerializeToString() != raw
                if row["changed"]:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    current = profile(candidate, task)
                    row["candidate"] = current
                    row["strict_lower"] = current["cost"] < baseline["cost"]
                    if row["strict_lower"]:
                        out = OUT / f"task{task:03d}_{label}.onnx"
                        onnx.save(candidate, out)
                        row["path"] = str(out.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    payload = {
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "rows": rows,
    }
    (HERE / "evidence" / "optimizer_scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "strict_lower": payload["strict_lower"]}, indent=2))


if __name__ == "__main__":
    main()
