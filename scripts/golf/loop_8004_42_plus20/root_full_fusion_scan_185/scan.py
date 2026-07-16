#!/usr/bin/env python3
"""Scan all 8009.46 members for exact ONNX graph fusions.

The pass families here were not covered by the prior staged-only optimizer
scan.  This phase is deliberately only a structural/cost pre-gate; any strict
winner is audited separately before it can enter the 71407 safe stage.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import onnx
import onnxoptimizer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"

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


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"fusion185_{task:03d}_{label}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        try:
            memory, params, cost = cost_of(str(path))
            return {"memory": int(memory), "params": int(params), "cost": int(cost)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".onnx")):
            task = int(Path(name).stem.removeprefix("task"))
            raw = archive.read(name)
            model = onnx.load_model_from_string(raw)
            baseline = profile(model, task, "base")
            for label, passes in PASS_SETS.items():
                row: dict[str, Any] = {
                    "task": task,
                    "pass_set": label,
                    "passes": passes,
                    "baseline": baseline,
                }
                try:
                    candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
                    encoded = candidate.SerializeToString()
                    row["changed"] = encoded != raw
                    if not row["changed"]:
                        rows.append(row)
                        continue
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(
                        candidate, strict_mode=True, data_prop=True
                    )
                    current = profile(candidate, task, label)
                    row["candidate"] = current
                    row["strict_lower"] = (
                        "cost" in baseline
                        and "cost" in current
                        and current["cost"] < baseline["cost"]
                    )
                    if row["strict_lower"]:
                        output = CANDIDATES / f"task{task:03d}_{label}.onnx"
                        onnx.save(candidate, output)
                        row["path"] = str(output.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    strict = [row for row in rows if row.get("strict_lower")]
    payload = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "pass_sets": PASS_SETS,
        "profiles": len(rows),
        "changed": sum(bool(row.get("changed")) for row in rows),
        "strict_lower_count": len(strict),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "strict_lower": strict}, indent=2))


if __name__ == "__main__":
    main()
