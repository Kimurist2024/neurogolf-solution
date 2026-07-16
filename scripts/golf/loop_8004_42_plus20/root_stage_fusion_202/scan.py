#!/usr/bin/env python3
"""Exact optimizer rescan of the active 71407 candidates.

The immutable-authority archive was scanned previously, but several active
models are later exact rebuilds.  This scan asks whether those descendants
expose another exact fusion.  It only emits strict-lower diagnostics; promotion
still requires a separate semantic/runtime audit.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxoptimizer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
STAGE = ROOT / "others" / "71407"

PASS_SETS: dict[str, list[str]] = {
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
    "all_fusions": [
        "fuse_matmul_add_bias_into_gemm",
        "fuse_transpose_into_gemm",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_pad_into_conv",
        "fuse_pad_into_pool",
        "fuse_concat_into_reshape",
        "fuse_qkv",
        "eliminate_common_subexpression",
        "eliminate_deadend",
        "eliminate_unused_initializer",
        "eliminate_duplicate_initializer",
    ],
}


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int | str]:
    with tempfile.TemporaryDirectory(prefix=f"stagefusion_{task:03d}_{label}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        try:
            memory, params, cost = cost_of(str(path))
            return {"memory": int(memory), "params": int(params), "cost": int(cost)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in sorted(STAGE.glob("task*.onnx")):
        task = int(path.stem.removeprefix("task"))
        raw = path.read_bytes()
        model = onnx.load_model_from_string(raw)
        baseline = profile(model, task, "base")
        for label, passes in PASS_SETS.items():
            row = {"task": task, "pass_set": label, "baseline": baseline}
            try:
                candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
                encoded = candidate.SerializeToString()
                row["changed"] = encoded != raw
                if not row["changed"]:
                    rows.append(row)
                    continue
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                current = profile(candidate, task, label)
                row["candidate"] = current
                row["strict_lower"] = (
                    "cost" in baseline and "cost" in current
                    and current["cost"] < baseline["cost"]
                )
                if row["strict_lower"]:
                    out = CANDIDATES / f"task{task:03d}_{label}.onnx"
                    onnx.save(candidate, out)
                    row["path"] = str(out.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    strict = [r for r in rows if r.get("strict_lower")]
    payload = {
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_files": len(list(STAGE.glob("task*.onnx"))),
        "profiles": len(rows),
        "changed": sum(bool(r.get("changed")) for r in rows),
        "strict_lower_count": len(strict),
        "strict_lower": strict,
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "changed": payload["changed"], "strict": strict}, indent=2))


if __name__ == "__main__":
    main()
