#!/usr/bin/env python
"""S6 opportunity census: intermediate-tensor dtype shrink (FP16 / BOOL) for NeuroGolf 2026.

For every task model in artifacts/research_snapshot/:
  1. infer_shapes(strict_mode=True)
  2. collect intermediate tensors (node outputs + value_info, excluding 'input'/'output'),
     per-tensor elem_type and byte size; compute params (initializers + Constant values)
  3. per-task and aggregate dtype histograms of intermediate memory bytes
  4. FP16 scenario: FLOAT intermediates at 2 B/elem (instead of 4) plus an 18000-byte
     penalty when any node consumes the graph input directly (Cast(input)->fp16 of
     [1,10,30,30] x 2B). score = max(1, 25 - ln(max(1, params + memory)))
  5. census FLOAT intermediates consumed ONLY by comparison/logic ops (BOOL-able, 1 B/elem)
  6. write docs/research/s6-memory-census.md and .json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, shape_inference

REPO = Path("/Users/user/Downloads/projects/Kaggle/Neurogolf")
SNAPSHOT_DIR = REPO / "artifacts" / "research_snapshot"
REPORT_MD = REPO / "docs" / "research" / "s6-memory-census.md"
REPORT_JSON = REPO / "docs" / "research" / "s6-memory-census.json"

EXCLUDED_NAMES = {"input", "output"}
CAST_INPUT_PENALTY_BYTES = 18_000  # [1,10,30,30] * 2 B fp16 copy of the graph input
FLOAT_T = TensorProto.FLOAT

# Ops whose consumption of a FLOAT tensor could be served by a BOOL tensor instead.
LOGIC_OPS_ANY_INPUT = {
    "Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual",
    "And", "Or", "Not", "Xor",
}
WHERE_CONDITION_INDEX = 0


def dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN({elem_type})"


def dtype_itemsize(elem_type: int) -> int | None:
    try:
        return int(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize)
    except Exception:
        return None


def shape_elems(tensor_type) -> int | None:
    """Number of elements from a TypeProto.Tensor; None if any dim is symbolic/unset."""
    if not tensor_type.HasField("shape"):
        return None
    elems = 1
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            elems *= dim.dim_value
        else:
            return None  # symbolic or unknown
    return elems


def tensor_proto_elems(tp) -> int:
    elems = 1
    for d in tp.dims:
        elems *= d
    return elems  # dims=[] -> scalar -> 1


def count_params(graph) -> int:
    params = 0
    for init in graph.initializer:
        params += tensor_proto_elems(init)
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name in ("value", "sparse_value") and attr.HasField("t"):
                params += tensor_proto_elems(attr.t)
            elif attr.name == "value" and attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                params += tensor_proto_elems(attr.sparse_tensor.values)
            elif attr.name in ("value_float", "value_int", "value_string"):
                params += 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def analyze_task(path: Path) -> dict:
    model = onnx.load(str(path))
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph

    params = count_params(graph)

    # name -> ValueInfoProto type, from value_info, graph inputs and outputs
    type_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            type_map[vi.name] = vi.type.tensor_type

    # intermediate tensor names: node outputs + value_info entries, minus exclusions
    inter_names = []
    seen = set()
    for node in graph.node:
        for out in node.output:
            if out and out not in EXCLUDED_NAMES and out not in seen:
                seen.add(out)
                inter_names.append(out)
    for vi in graph.value_info:
        if vi.name and vi.name not in EXCLUDED_NAMES and vi.name not in seen:
            seen.add(vi.name)
            inter_names.append(vi.name)

    # consumer map: tensor name -> list of (op_type, input_index)
    consumers = defaultdict(list)
    input_consumed_directly = False
    for node in graph.node:
        for idx, inp in enumerate(node.input):
            if not inp:
                continue
            consumers[inp].append((node.op_type, idx))
            if inp == "input":
                input_consumed_directly = True

    dtype_bytes = Counter()       # dtype name -> bytes
    dtype_count = Counter()       # dtype name -> tensor count
    memory = 0
    float_bytes = 0
    unknown_tensors = []          # names with unresolvable shape/dtype
    boolable_tensors = 0
    boolable_float_bytes = 0      # current bytes (4 B/elem) of BOOL-able FLOAT tensors
    boolable_saving_bytes = 0     # bytes saved going 4 B -> 1 B

    for name in inter_names:
        tt = type_map.get(name)
        if tt is None:
            unknown_tensors.append(name)
            continue
        elem_type = tt.elem_type
        itemsize = dtype_itemsize(elem_type)
        elems = shape_elems(tt)
        if itemsize is None or elems is None:
            unknown_tensors.append(name)
            continue
        nbytes = elems * itemsize
        dname = dtype_name(elem_type)
        dtype_bytes[dname] += nbytes
        dtype_count[dname] += 1
        memory += nbytes
        if elem_type == FLOAT_T:
            float_bytes += nbytes
            cons = consumers.get(name, [])
            if cons and all(
                op in LOGIC_OPS_ANY_INPUT
                or (op == "Where" and idx == WHERE_CONDITION_INDEX)
                for op, idx in cons
            ):
                boolable_tensors += 1
                boolable_float_bytes += nbytes
                boolable_saving_bytes += elems * 3  # 4 B -> 1 B

    # FP16 scenario
    new_memory = (memory - float_bytes) + float_bytes // 2
    if input_consumed_directly:
        new_memory += CAST_INPUT_PENALTY_BYTES

    old_cost = params + memory
    new_cost = params + new_memory
    old_score = max(1.0, 25.0 - math.log(max(1, old_cost)))
    new_score = max(1.0, 25.0 - math.log(max(1, new_cost)))
    score_delta = new_score - old_score

    return {
        "task": path.stem,
        "params": params,
        "memory": memory,
        "float_bytes": float_bytes,
        "dtype_bytes": dict(dtype_bytes),
        "dtype_count": dict(dtype_count),
        "n_intermediates": len(inter_names),
        "n_unknown_tensors": len(unknown_tensors),
        "unknown_tensors": unknown_tensors[:10],
        "input_consumed_directly": input_consumed_directly,
        "fp16_new_memory": new_memory,
        "old_cost": old_cost,
        "new_cost": new_cost,
        "old_score": old_score,
        "new_score": new_score,
        "score_delta": score_delta,
        "boolable_tensors": boolable_tensors,
        "boolable_float_bytes": boolable_float_bytes,
        "boolable_saving_bytes": boolable_saving_bytes,
    }


def main() -> None:
    paths = sorted(SNAPSHOT_DIR.glob("task*.onnx"))
    results = []
    errors = []
    for p in paths:
        try:
            results.append(analyze_task(p))
        except Exception as exc:  # record, keep going
            errors.append({"task": p.stem, "error": f"{type(exc).__name__}: {exc}"})

    # aggregates
    total_memory = sum(r["memory"] for r in results)
    total_params = sum(r["params"] for r in results)
    total_float_bytes = sum(r["float_bytes"] for r in results)
    total_gain = sum(r["score_delta"] for r in results)
    total_pos_gain = sum(r["score_delta"] for r in results if r["score_delta"] > 0)
    agg_dtype_bytes = Counter()
    agg_dtype_count = Counter()
    for r in results:
        agg_dtype_bytes.update(r["dtype_bytes"])
        agg_dtype_count.update(r["dtype_count"])

    total_boolable_tensors = sum(r["boolable_tensors"] for r in results)
    total_boolable_bytes = sum(r["boolable_float_bytes"] for r in results)
    total_boolable_saving = sum(r["boolable_saving_bytes"] for r in results)
    tasks_with_boolable = sum(1 for r in results if r["boolable_tensors"] > 0)

    by_gain = sorted(results, key=lambda r: r["score_delta"], reverse=True)
    top20 = by_gain[:20]
    no_help = [r for r in results if r["score_delta"] <= 0.0]
    tiny_help = [r for r in results if 0.0 < r["score_delta"] < 0.01]
    n_unknown_total = sum(r["n_unknown_tensors"] for r in results)

    payload = {
        "meta": {
            "snapshot_dir": str(SNAPSHOT_DIR),
            "n_tasks": len(results),
            "n_errors": len(errors),
            "cast_input_penalty_bytes": CAST_INPUT_PENALTY_BYTES,
            "score_formula": "max(1, 25 - ln(max(1, memory + params)))",
        },
        "aggregate": {
            "total_memory_bytes": total_memory,
            "total_params": total_params,
            "total_float_intermediate_bytes": total_float_bytes,
            "dtype_bytes": dict(agg_dtype_bytes),
            "dtype_tensor_counts": dict(agg_dtype_count),
            "fp16_total_score_gain": total_gain,
            "fp16_total_score_gain_positive_only": total_pos_gain,
            "boolable_tensors": total_boolable_tensors,
            "boolable_current_bytes": total_boolable_bytes,
            "boolable_saving_bytes": total_boolable_saving,
            "tasks_with_boolable": tasks_with_boolable,
            "tasks_fp16_no_help": len(no_help),
            "tasks_fp16_tiny_help_lt_0p01": len(tiny_help),
            "unresolved_intermediate_tensors": n_unknown_total,
        },
        "top20_by_fp16_gain": [
            {k: r[k] for k in (
                "task", "params", "memory", "float_bytes", "fp16_new_memory",
                "old_score", "new_score", "score_delta", "boolable_saving_bytes")}
            for r in top20
        ],
        "fp16_no_help_tasks": [
            {k: r[k] for k in ("task", "params", "memory", "float_bytes", "score_delta")}
            for r in sorted(no_help, key=lambda r: r["score_delta"])
        ],
        "errors": errors,
        "per_task": results,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=1))

    # ---- markdown report ----
    lines = []
    a = lines.append
    a("# S6 Memory Census — Intermediate-Tensor Dtype Shrink (FP16 / BOOL)")
    a("")
    a(f"Snapshot: `{SNAPSHOT_DIR}` — {len(results)} tasks analyzed, {len(errors)} errors.")
    a("")
    a("Scorer: `score = max(1, 25 - ln(max(1, memory + params)))`; memory counts")
    a("intermediate tensors only (node outputs + value_info, excluding `input`/`output`),")
    a("shapes from `infer_shapes(strict_mode=True)`. FP16 scenario: every FLOAT")
    a("intermediate at 2 B/elem, plus an 18,000-byte penalty per task whose nodes")
    a("consume the graph input directly (one `Cast(input)->fp16` of [1,10,30,30]).")
    a("")
    a("## Aggregate (400 tasks)")
    a("")
    a(f"- Total current intermediate memory: **{total_memory:,} bytes**")
    a(f"- Total params (initializers + Constants): **{total_params:,}**")
    a(f"- Total FLOAT intermediate bytes: **{total_float_bytes:,}** "
      f"({(100.0 * total_float_bytes / total_memory if total_memory else 0):.1f}% of memory)")
    a(f"- Estimated total FP16 score gain: **{total_gain:+.3f} points** "
      f"(positive-deltas only: {total_pos_gain:+.3f})")
    a(f"- Tasks where FP16 does NOT help (delta <= 0): **{len(no_help)}**; "
      f"tiny help (<0.01): {len(tiny_help)}")
    a(f"- Unresolved intermediate tensors (no shape/dtype): {n_unknown_total}")
    a("")
    a("### Dtype histogram of intermediate memory (aggregate)")
    a("")
    a("| dtype | tensors | bytes | % of memory |")
    a("|---|---:|---:|---:|")
    for dname, b in agg_dtype_bytes.most_common():
        pct = 100.0 * b / total_memory if total_memory else 0.0
        a(f"| {dname} | {agg_dtype_count[dname]:,} | {b:,} | {pct:.2f}% |")
    a("")
    a("## BOOL-able census")
    a("")
    a("FLOAT intermediates consumed ONLY by comparison/logic ops")
    a("(Greater/Less/Equal/GreaterOrEqual/LessOrEqual/And/Or/Not/Xor, or Where as condition):")
    a("")
    a(f"- BOOL-able FLOAT tensors: **{total_boolable_tensors:,}** across "
      f"{tasks_with_boolable} tasks")
    a(f"- Current bytes held by those tensors (4 B/elem): **{total_boolable_bytes:,}**")
    a(f"- Bytes saved at 1 B/elem (BOOL): **{total_boolable_saving:,}**")
    a("")
    a("## Top 20 tasks by potential FP16 score gain")
    a("")
    a("| task | params | memory (B) | FLOAT (B) | fp16 memory (B) | old score | new score | delta |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in top20:
        a(f"| {r['task']} | {r['params']:,} | {r['memory']:,} | {r['float_bytes']:,} "
          f"| {r['fp16_new_memory']:,} | {r['old_score']:.3f} | {r['new_score']:.3f} "
          f"| {r['score_delta']:+.4f} |")
    a("")
    a("## Tasks where FP16 would NOT help (delta <= 0)")
    a("")
    if no_help:
        a("| task | params | memory (B) | FLOAT (B) | delta |")
        a("|---|---:|---:|---:|---:|")
        for r in sorted(no_help, key=lambda r: r["score_delta"]):
            a(f"| {r['task']} | {r['params']:,} | {r['memory']:,} "
              f"| {r['float_bytes']:,} | {r['score_delta']:+.4f} |")
        a("")
        a("Reasons: memory already small relative to params, FLOAT share of memory is")
        a("low (BOOL/INT64-dominated), or the 18,000-byte input-Cast penalty exceeds the")
        a("FLOAT savings.")
    else:
        a("(none)")
    if errors:
        a("")
        a("## Errors")
        a("")
        for e in errors:
            a(f"- {e['task']}: {e['error']}")
    a("")
    a("Machine-readable data: `docs/research/s6-memory-census.json` (per-task records")
    a("under `per_task`).")
    a("")
    REPORT_MD.write_text("\n".join(lines))

    # console summary
    print(f"tasks={len(results)} errors={len(errors)}")
    print(f"total_memory={total_memory:,} total_params={total_params:,}")
    print(f"total_float_bytes={total_float_bytes:,}")
    print(f"fp16_total_gain={total_gain:+.4f} (positive-only {total_pos_gain:+.4f})")
    print(f"boolable: tensors={total_boolable_tensors} bytes={total_boolable_bytes:,} "
          f"saving={total_boolable_saving:,} tasks={tasks_with_boolable}")
    print(f"no_help={len(no_help)} tiny_help<0.01={len(tiny_help)} unknown_tensors={n_unknown_total}")
    print("top5:")
    for r in top20[:5]:
        print(f"  {r['task']}: delta={r['score_delta']:+.4f} "
              f"(mem {r['memory']:,} -> {r['fp16_new_memory']:,}, params {r['params']:,})")


if __name__ == "__main__":
    main()
