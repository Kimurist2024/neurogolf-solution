#!/usr/bin/env python3
"""Fail-closed audit for the low44 eight-task expansion lane."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
OLD = ROOT / "submission_base_8004.50.zip"
TARGETS = (303, 98, 395, 167, 289, 38, 262, 269)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
RISK_OPS = {
    "CenterCropPad", "TfIdfVectorizer", "GatherND", "ScatterND",
    "ScatterElements", "ConstantOfShape", "Resize", "Shrink",
}
RULES = {
    303: "Add 2 to every cell whose row or column is entirely zero; otherwise retain the cell.",
    98: "Retain a cell iff its generator-defined clipped 3x3 neighborhood contains a zero; otherwise output zero.",
    395: "Remove row 3, compare every remaining cell with the corresponding row-3 cell, and output 2 iff both are zero.",
    167: "Emit a fixed 3x3 color-5 permutation pattern selected by the generator's distinct-character count modulo 5.",
    289: "Nearest-neighbor repeat every cell in both axes by the generator's distinct-character-count scale.",
    38: "Emit a 1x5 prefix of ones derived from adjacent horizontal 1,1 occurrences, followed by zeros.",
    262: "For every input row emit three copies of 3+(second-first)/5.",
    269: "Nearest-neighbor repeat every cell in both axes by the generator's distinct-character-count scale.",
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_private_exact15.audit_exact import trace_shapes  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def known(model: onnx.ModelProto, task: int, disabled: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    examples = scoring.load_examples(task)
    converted = []
    skipped = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            item = scoring.convert_to_numpy(example)
            if item is None:
                skipped += 1
            else:
                converted.append(item)
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {"right": 0, "wrong": 0, "errors": len(converted), "total": len(converted), "skipped": skipped, "session_error": f"{type(exc).__name__}: {exc}", "output_shapes": []}
    right = wrong = errors = near_margin = 0
    shapes: set[tuple[int, ...]] = set()
    for item in converted:
        try:
            raw = session.run(["output"], {"input": item["input"]})[0]
            shapes.add(tuple(int(x) for x in raw.shape))
            near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
            if np.array_equal(raw > 0, item["output"] > 0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {"right": right, "wrong": wrong, "errors": errors, "total": len(converted), "skipped": skipped, "near_margin": near_margin, "output_shapes": [list(x) for x in sorted(shapes)]}


def exact_scan(model: onnx.ModelProto) -> dict[str, object]:
    used = {name for node in model.graph.node for name in node.input}
    used.update(output.name for output in model.graph.output)
    unused = [item.name for item in model.graph.initializer if item.name not in used]

    def initializer_key(item: onnx.TensorProto) -> bytes:
        clone = copy.deepcopy(item)
        clone.name = ""
        return clone.SerializeToString(deterministic=True)

    duplicates = []
    for left in range(len(model.graph.initializer)):
        for right in range(left + 1, len(model.graph.initializer)):
            if initializer_key(model.graph.initializer[left]) == initializer_key(model.graph.initializer[right]):
                duplicates.append([model.graph.initializer[left].name, model.graph.initializer[right].name])
    identities = [index for index, node in enumerate(model.graph.node) if node.op_type == "Identity"]
    return {"unused_initializers": unused, "duplicate_initializers": duplicates, "identity_nodes": identities, "candidate_count": len(unused) + len(duplicates) + len(identities)}


def audit_task(task: int) -> dict[str, object]:
    path = HERE / "baselines" / f"task{task:03d}.onnx"
    model = onnx.load(path)
    memory, params, cost = cost_of(str(path))
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
    disable = known(model, task, True)
    default = known(model, task, False)
    try:
        trace = trace_shapes(model, task)
    except Exception as exc:
        trace = {"shape_cloak": True, "error": f"{type(exc).__name__}: {exc}"}
    ops = collections.Counter(node.op_type for node in model.graph.node)
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    declared = dims(model.graph.output[0])
    return {
        "task": task,
        "rule": RULES[task],
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path.read_bytes()),
        "file_bytes": path.stat().st_size,
        "actual_cost": {"memory": memory, "params": params, "cost": cost},
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "ops": dict(sorted(ops.items())),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum > 16,
        "risk_ops": sorted(set(ops) & RISK_OPS),
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_data_prop": strict,
        "strict_error": strict_error,
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type in BANNED or "Sequence" in node.op_type}),
        "conv_bias_findings": check_conv_bias(model),
        "declared_output_shape": declared,
        "runtime_shape_trace": trace,
        "known_disable_all": disable,
        "known_default": default,
        "exact_local_scan": exact_scan(model),
    }


def main() -> None:
    (HERE / "baselines").mkdir(parents=True, exist_ok=True)
    base_data = BASE.read_bytes()
    old_data = OLD.read_bytes()
    same = {}
    with zipfile.ZipFile(BASE) as latest, zipfile.ZipFile(OLD) as old:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = latest.read(member)
            (HERE / "baselines" / member).write_bytes(data)
            same[str(task)] = data == old.read(member)
    rows = {str(task): audit_task(task) for task in TARGETS}
    payload = {
        "baseline": BASE.name,
        "baseline_sha256": sha(base_data),
        "comparison_baseline": OLD.name,
        "comparison_sha256": sha(old_data),
        "targets": list(TARGETS),
        "same_payload_as_8004_50": same,
        "rows": rows,
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({task: {"cost": row["actual_cost"]["cost"], "ops": row["ops"], "known": [row["known_disable_all"]["right"], row["known_disable_all"]["total"]], "default": [row["known_default"]["right"], row["known_default"]["total"]], "shape_cloak": row["runtime_shape_trace"].get("shape_cloak"), "exact": row["exact_local_scan"]} for task, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
