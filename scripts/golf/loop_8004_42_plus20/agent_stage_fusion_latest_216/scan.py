#!/usr/bin/env python3
"""Fresh exact-cleanup rescan of the current active 71407 descendants.

The first 21 profiles intentionally match root_stage_fusion_202.  Four
additional fixed-point profiles cover dead code, initializer aliases, no-op
nodes, and their combination.  This tool is read-only with respect to the
active stage: it only writes strict-lower diagnostic candidates beside this
script.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import onnxoptimizer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
STAGE = ROOT / "others" / "71407"

# Kept byte-for-byte equivalent in membership/order to root_stage_fusion_202.
BASE_PASS_SETS: dict[str, list[str]] = {
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

NOOP_PASSES = [
    "eliminate_identity",
    "eliminate_nop_cast",
    "eliminate_nop_concat",
    "eliminate_nop_dropout",
    "eliminate_nop_expand",
    "eliminate_nop_flatten",
    "eliminate_nop_pad",
    "eliminate_nop_reshape",
    "eliminate_nop_split",
    "eliminate_nop_transpose",
    "eliminate_nop_with_unit",
]

EXTRA_PASS_SETS: dict[str, list[str]] = {
    "cleanup_dce": ["eliminate_deadend"],
    "cleanup_initializer_aliases": [
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
    ],
    "cleanup_noops": NOOP_PASSES,
    "cleanup_all": [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
        *NOOP_PASSES,
        "eliminate_common_subexpression",
        "eliminate_consecutive_idempotent_ops",
    ],
}

PASS_SETS = {**BASE_PASS_SETS, **EXTRA_PASS_SETS}


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int | str]:
    with tempfile.TemporaryDirectory(prefix=f"stagefusion216_{task:03d}_{label}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        try:
            memory, params, cost = cost_of(str(path))
            return {"memory": int(memory), "params": int(params), "cost": int(cost)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def structural_scan(model: onnx.ModelProto) -> dict:
    """Count independently visible cleanup opportunities without mutating."""
    consumers: Counter[str] = Counter(
        name for node in model.graph.node for name in node.input if name
    )
    graph_outputs = {v.name for v in model.graph.output}
    init_by_digest: dict[tuple[int, tuple[int, ...], str], list[str]] = defaultdict(list)
    for init in model.graph.initializer:
        # Serialize only tensor payload/metadata, deliberately excluding name.
        clone = onnx.TensorProto()
        clone.CopyFrom(init)
        clone.name = ""
        digest = hashlib.sha256(clone.SerializeToString()).hexdigest()
        init_by_digest[(init.data_type, tuple(init.dims), digest)].append(init.name)
    duplicate_groups = [names for names in init_by_digest.values() if len(names) > 1]
    unused_initializers = [
        init.name for init in model.graph.initializer
        if consumers[init.name] == 0 and init.name not in graph_outputs
    ]
    dead_node_outputs = [
        out
        for node in model.graph.node
        for out in node.output
        if out and consumers[out] == 0 and out not in graph_outputs
    ]
    noop_op_types = Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type in {
            "Identity", "Dropout", "Flatten", "Expand", "Pad", "Reshape",
            "Split", "Transpose", "Cast", "Concat",
        }
    )
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "unused_initializers": unused_initializers,
        "duplicate_initializer_groups": duplicate_groups,
        "dead_node_outputs": dead_node_outputs,
        "possible_noop_op_types": dict(sorted(noop_op_types.items())),
    }


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    stage_paths = sorted(STAGE.glob("task*.onnx"))
    rows: list[dict] = []
    structures: list[dict] = []
    stage_snapshot: list[dict] = []
    for path in stage_paths:
        task = int(path.stem.removeprefix("task"))
        raw = path.read_bytes()
        stage_snapshot.append({
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        model = onnx.load_model_from_string(raw)
        baseline = profile(model, task, "base")
        structures.append({"task": task, **structural_scan(model)})
        for label, passes in PASS_SETS.items():
            row = {
                "task": task,
                "profile_class": "baseline_21" if label in BASE_PASS_SETS else "extra_cleanup",
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
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                current = profile(candidate, task, label)
                row["candidate"] = current
                row["strict_lower"] = (
                    "cost" in baseline
                    and "cost" in current
                    and current["cost"] < baseline["cost"]
                )
                if row["strict_lower"]:
                    out = CANDIDATES / f"task{task:03d}_{label}.onnx"
                    onnx.save(candidate, out)
                    row["path"] = str(out.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
                    row["transform_authority"] = {
                        "engine": "onnxoptimizer",
                        "version": getattr(onnxoptimizer, "__version__", "unknown"),
                        "fixed_point": True,
                        "passes": passes,
                    }
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    strict = [r for r in rows if r.get("strict_lower")]
    payload = {
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_files": len(stage_paths),
        "stage_snapshot": stage_snapshot,
        "baseline_pass_sets": len(BASE_PASS_SETS),
        "extra_cleanup_pass_sets": len(EXTRA_PASS_SETS),
        "profiles": len(rows),
        "baseline_profiles": sum(r["profile_class"] == "baseline_21" for r in rows),
        "cleanup_profiles": sum(r["profile_class"] == "extra_cleanup" for r in rows),
        "changed": sum(bool(r.get("changed")) for r in rows),
        "strict_lower_count": len(strict),
        "strict_lower": strict,
        "structural_scans": structures,
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "stage_files": payload["stage_files"],
        "profiles": payload["profiles"],
        "changed": payload["changed"],
        "strict_lower_count": len(strict),
        "strict_lower": strict,
    }, indent=2))


if __name__ == "__main__":
    main()
