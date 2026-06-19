"""S7 no-op census: quantify the opportunity of eliminating no-op nodes.

For every task model in artifacts/research_snapshot/ (read-only):

1. Run ``onnx.shape_inference.infer_shapes(strict_mode=True)`` to obtain
   dtype/shape for every node output.
2. Build a global op_type histogram (top 30 reported).
3. Detect no-op nodes whose removal (rewiring consumers to the node's data
   input) cannot change the computation:
     - Identity
     - Cast with input dtype == output dtype
     - Add/Sub with an all-zero constant operand (Sub: subtrahend only)
     - Mul/Div with an all-one constant operand (Div: divisor only)
     - Reshape/Squeeze/Unsqueeze/Flatten with input shape == output shape
     - Transpose with identity permutation
     - Pad with all-zero pads
     - Clip with absent / -inf / +inf bounds
     - Concat with a single input
4. Saving = byte size of the no-op's output tensor. When the output IS the
   graph output 'output' (which never counts toward memory), removal instead
   renames the producer's tensor to 'output', saving the producer's old
   output bytes — counted but flagged separately.
5. Memory/params/score deltas are computed by SIMULATING the rewired graph
   (chains of no-ops resolved; constants orphaned by the removal are also
   dropped, reducing params and — for Constant nodes — memory).

Static-shape memory only (the official scorer additionally maxes with the
ORT profiler trace; for these fixed-shape models the static value matches).

Outputs:
  docs/research/s7-noop-census.md
  docs/research/s7-noop-census.json
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

REPO = pathlib.Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO / "artifacts" / "research_snapshot"
OUT_MD = REPO / "docs" / "research" / "s7-noop-census.md"
OUT_JSON = REPO / "docs" / "research" / "s7-noop-census.json"

CATEGORIES = [
    "Identity",
    "Cast_same_dtype",
    "AddSub_zero",
    "MulDiv_one",
    "ShapeOp_same_shape",
    "Transpose_identity",
    "Pad_zero",
    "Clip_unbounded",
    "Concat_single",
]


def score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def tensor_bytes(vi_map: dict[str, Any], name: str) -> int | None:
    """Byte size of an intermediate tensor from inferred value info."""
    item = vi_map.get(name)
    if item is None or not item.type.HasField("tensor_type"):
        return None
    tt = item.type.tensor_type
    if not tt.HasField("shape"):
        return None
    n = 1
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        n *= dim.dim_value
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
    return n * np.dtype(np_dtype).itemsize


def tensor_shape(vi_map: dict[str, Any], inits: dict[str, Any], name: str):
    if name in inits:
        return tuple(inits[name].dims)
    item = vi_map.get(name)
    if item is None or not item.type.HasField("tensor_type"):
        return None
    tt = item.type.tensor_type
    if not tt.HasField("shape"):
        return None
    dims = []
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(dim.dim_value)
    return tuple(dims)


def tensor_dtype(vi_map: dict[str, Any], inits: dict[str, Any], name: str):
    if name in inits:
        return inits[name].data_type
    item = vi_map.get(name)
    if item is None or not item.type.HasField("tensor_type"):
        return None
    return item.type.tensor_type.elem_type


def constant_node_params(node: onnx.NodeProto) -> int:
    """Param count of a Constant node per the official scorer rules."""
    params = 0
    for attr in node.attribute:
        if attr.name == "value":
            params += math.prod(attr.t.dims)
        elif attr.name == "sparse_value":
            params += math.prod(attr.sparse_tensor.values.dims)
        elif attr.name == "value_floats":
            params += len(attr.floats)
        elif attr.name == "value_ints":
            params += len(attr.ints)
        elif attr.name == "value_strings":
            params += len(attr.strings)
    return params


def calculate_params(graph: onnx.GraphProto) -> int:
    params = 0
    for init in graph.initializer:
        params += math.prod(init.dims)
    for node in graph.node:
        if node.op_type == "Constant":
            params += constant_node_params(node)
    return params


def build_const_values(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """name -> ndarray for initializers and Constant node outputs."""
    consts: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        try:
            consts[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            try:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
                elif attr.name == "value_floats":
                    consts[node.output[0]] = np.array(list(attr.floats), np.float32)
                elif attr.name == "value_ints":
                    consts[node.output[0]] = np.array(list(attr.ints), np.int64)
                elif attr.name == "value_float":
                    consts[node.output[0]] = np.array(attr.f, np.float32)
                elif attr.name == "value_int":
                    consts[node.output[0]] = np.array(attr.i, np.int64)
            except Exception:
                pass
    return consts


def detect_noops(
    graph: onnx.GraphProto,
    vi_map: dict[str, Any],
    inits: dict[str, Any],
    consts: dict[str, np.ndarray],
    opset: int,
) -> list[tuple[int, str, str]]:
    """Return [(node_index, category, surviving_input_name)]."""
    found: list[tuple[int, str, str]] = []
    for idx, node in enumerate(graph.node):
        op = node.op_type
        cat: str | None = None
        keep: str | None = None

        if op == "Identity":
            cat, keep = "Identity", node.input[0]

        elif op == "Cast":
            din = tensor_dtype(vi_map, inits, node.input[0])
            dout = tensor_dtype(vi_map, inits, node.output[0])
            if din is not None and din == dout:
                cat, keep = "Cast_same_dtype", node.input[0]

        elif op in ("Add", "Sub", "Mul", "Div") and len(node.input) == 2:
            a, b = node.input[0], node.input[1]
            out_shape = tensor_shape(vi_map, inits, node.output[0])

            def is_noop_operand(const_name: str, live_name: str, want) -> bool:
                v = consts.get(const_name)
                if v is None or v.size == 0:
                    return False
                if not bool(np.all(v == want)):
                    return False
                live_shape = tensor_shape(vi_map, inits, live_name)
                return (
                    out_shape is not None
                    and live_shape is not None
                    and live_shape == out_shape
                )

            if op in ("Add", "Sub"):
                if is_noop_operand(b, a, 0):
                    cat, keep = "AddSub_zero", a
                elif op == "Add" and is_noop_operand(a, b, 0):
                    cat, keep = "AddSub_zero", b
            else:  # Mul / Div
                if is_noop_operand(b, a, 1):
                    cat, keep = "MulDiv_one", a
                elif op == "Mul" and is_noop_operand(a, b, 1):
                    cat, keep = "MulDiv_one", b

        elif op in ("Reshape", "Squeeze", "Unsqueeze", "Flatten"):
            sin = tensor_shape(vi_map, inits, node.input[0])
            sout = tensor_shape(vi_map, inits, node.output[0])
            if sin is not None and sout is not None and sin == sout:
                cat, keep = "ShapeOp_same_shape", node.input[0]

        elif op == "Transpose":
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            sin = tensor_shape(vi_map, inits, node.input[0])
            if perm is not None:
                if perm == list(range(len(perm))):
                    cat, keep = "Transpose_identity", node.input[0]
            elif sin is not None and len(sin) <= 1:
                # absent perm reverses dims: identity only for rank <= 1
                cat, keep = "Transpose_identity", node.input[0]

        elif op == "Pad":
            pads = None
            for attr in node.attribute:
                if attr.name in ("pads", "paddings"):
                    pads = list(attr.ints)
            if pads is None and len(node.input) >= 2 and node.input[1]:
                v = consts.get(node.input[1])
                if v is not None:
                    pads = [int(x) for x in np.asarray(v).flatten()]
            if pads is not None and all(p == 0 for p in pads):
                cat, keep = "Pad_zero", node.input[0]

        elif op == "Clip":
            lo = hi = None
            has_attr_bounds = False
            for attr in node.attribute:
                if attr.name == "min":
                    lo, has_attr_bounds = attr.f, True
                elif attr.name == "max":
                    hi, has_attr_bounds = attr.f, True
            if opset >= 11 and not has_attr_bounds:
                if len(node.input) >= 2 and node.input[1]:
                    v = consts.get(node.input[1])
                    lo = float(np.asarray(v).flatten()[0]) if v is not None else np.nan
                if len(node.input) >= 3 and node.input[2]:
                    v = consts.get(node.input[2])
                    hi = float(np.asarray(v).flatten()[0]) if v is not None else np.nan
            lo_ok = lo is None or lo == float("-inf")
            hi_ok = hi is None or hi == float("inf")
            if lo_ok and hi_ok:
                cat, keep = "Clip_unbounded", node.input[0]

        elif op == "Concat" and len(node.input) == 1:
            cat, keep = "Concat_single", node.input[0]

        if cat is not None and keep is not None and node.output and node.output[0]:
            found.append((idx, cat, keep))
    return found


def analyze_task(path: pathlib.Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    graph_raw = model.graph
    opset = max(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        default=9,
    )
    op_hist = Counter(n.op_type for n in graph_raw.node)

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    vi_map = {
        t.name: t
        for t in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    inits = {init.name: init for init in graph.initializer}
    consts = build_const_values(graph)

    # --- memory before (static, mirrors scorer minus profiler max) ----------
    intermediates: dict[str, int] = {}
    for node in graph.node:
        for out in node.output:
            if out and out not in ("input", "output"):
                b = tensor_bytes(vi_map, out)
                if b is None:
                    raise ValueError(f"unresolvable shape for tensor {out!r}")
                intermediates[out] = b
    for t in graph.value_info:
        if t.name not in ("input", "output") and t.name not in intermediates:
            b = tensor_bytes(vi_map, t.name)
            if b is not None:
                intermediates[t.name] = b
    memory_before = sum(intermediates.values())
    params_before = calculate_params(graph)

    # --- detect no-ops -------------------------------------------------------
    noops = detect_noops(graph, vi_map, inits, consts, opset)
    noop_idx = {i for i, _, _ in noops}
    replacement = {graph.node[i].output[0]: keep for i, _, keep in noops}

    def resolve(name: str) -> str:
        seen = set()
        while name in replacement and name not in seen:
            seen.add(name)
            name = replacement[name]
        return name

    # tensor renamed to 'output' if a no-op chain produces the graph output
    producer = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producer[out] = i
    renamed_to_output: str | None = None
    cur = "output"
    while cur in producer and producer[cur] in noop_idx:
        cur = replacement[cur]
    if cur != "output":
        renamed_to_output = cur  # may be 'input'/initializer -> Identity kept, free

    # --- per-noop attribution -----------------------------------------------
    by_cat_count: Counter = Counter()
    by_cat_bytes: Counter = Counter()
    output_flagged = 0
    output_flagged_bytes = 0
    for i, cat, keep in noops:
        out_name = graph.node[i].output[0]
        by_cat_count[cat] += 1
        if out_name == "output":
            src = resolve(keep)
            b = intermediates.get(src, 0)  # 0 if source is input/initializer
            output_flagged += 1
            output_flagged_bytes += b
            by_cat_bytes[cat] += b
        else:
            by_cat_bytes[cat] += intermediates.get(out_name, 0)

    # --- simulate rewired graph ----------------------------------------------
    remaining = [i for i in range(len(graph.node)) if i not in noop_idx]
    # consumers after rewiring
    consumer_count: Counter = Counter()
    for i in remaining:
        for inp in graph.node[i].input:
            if inp:
                consumer_count[resolve(inp)] += 1
    if renamed_to_output is not None:
        consumer_count[renamed_to_output] += 1  # must still be produced
    else:
        consumer_count["output"] += 1

    # constants orphaned by the removal (single pass: direct orphans only)
    params_after = params_before
    orphan_inits = 0
    orphan_const_nodes = 0
    dead_const_idx: set[int] = set()
    for name, init in inits.items():
        if consumer_count[name] == 0:
            referenced_before = any(
                name in n.input for n in graph.node
            )
            if referenced_before:  # only count if the no-op removal orphaned it
                only_noop_consumers = all(
                    (idx in noop_idx)
                    for idx, n in enumerate(graph.node)
                    if name in n.input
                )
                if only_noop_consumers:
                    params_after -= math.prod(init.dims)
                    orphan_inits += 1
    for i in remaining:
        node = graph.node[i]
        if node.op_type == "Constant" and node.output:
            if consumer_count[node.output[0]] == 0:
                consumed_by_noop = any(
                    node.output[0] in graph.node[j].input for j in noop_idx
                )
                if consumed_by_noop:
                    dead_const_idx.add(i)
                    params_after -= constant_node_params(node)
                    orphan_const_nodes += 1

    final_nodes = [i for i in remaining if i not in dead_const_idx]
    memory_after = 0
    for i in final_nodes:
        for out in graph.node[i].output:
            if not out or out in ("input", "output"):
                continue
            if out == renamed_to_output:
                continue  # becomes 'output' -> free
            memory_after += intermediates.get(out, 0)

    cost_before = memory_before + params_before
    cost_after = memory_after + params_after
    return {
        "task": path.stem,
        "n_nodes": len(graph_raw.node),
        "op_hist": dict(op_hist),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "params_before": params_before,
        "params_after": params_after,
        "noop_count": len(noops),
        "noops_by_category": dict(by_cat_count),
        "bytes_by_category": dict(by_cat_bytes),
        "removable_bytes": memory_before - memory_after,
        "removable_params": params_before - params_after,
        "orphaned_initializers": orphan_inits,
        "orphaned_constant_nodes": orphan_const_nodes,
        "output_rewire_flagged": output_flagged,
        "output_rewire_flagged_bytes": output_flagged_bytes,
        "score_before": score(cost_before),
        "score_after": score(cost_after),
        "score_delta": score(cost_after) - score(cost_before),
        "noop_detail": [
            {
                "node_index": i,
                "op_type": graph.node[i].op_type,
                "category": cat,
                "output": graph.node[i].output[0],
                "rewire_to": keep,
                "is_graph_output": graph.node[i].output[0] == "output",
            }
            for i, cat, keep in noops
        ],
    }


def main() -> None:
    paths = sorted(SNAPSHOT_DIR.glob("task*.onnx"))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    global_hist: Counter = Counter()

    for p in paths:
        try:
            r = analyze_task(p)
            global_hist.update(r["op_hist"])
            results.append(r)
        except Exception as exc:  # record, don't abort the census
            errors.append({"task": p.stem, "error": f"{type(exc).__name__}: {exc}"})

    total_by_cat: Counter = Counter()
    bytes_by_cat: Counter = Counter()
    for r in results:
        total_by_cat.update(r["noops_by_category"])
        bytes_by_cat.update(r["bytes_by_category"])

    total_noops = sum(total_by_cat.values())
    total_bytes = sum(r["removable_bytes"] for r in results)
    total_params_removed = sum(r["removable_params"] for r in results)
    total_score_delta = sum(r["score_delta"] for r in results)
    tasks_with_noops = sum(1 for r in results if r["noop_count"] > 0)
    flagged = sum(r["output_rewire_flagged"] for r in results)
    flagged_bytes = sum(r["output_rewire_flagged_bytes"] for r in results)
    top_tasks = sorted(results, key=lambda r: -r["score_delta"])[:20]

    summary = {
        "n_tasks": len(results),
        "n_errors": len(errors),
        "tasks_with_noops": tasks_with_noops,
        "total_noop_count": total_noops,
        "noops_by_category": dict(total_by_cat),
        "attributed_bytes_by_category": dict(bytes_by_cat),
        "total_removable_bytes": total_bytes,
        "total_removable_params": total_params_removed,
        "output_rewire_flagged": flagged,
        "output_rewire_flagged_bytes": flagged_bytes,
        "estimated_total_score_delta": total_score_delta,
        "top30_op_types": global_hist.most_common(30),
        "top5_tasks": [
            {
                "task": r["task"],
                "score_delta": r["score_delta"],
                "removable_bytes": r["removable_bytes"],
                "noop_count": r["noop_count"],
            }
            for r in top_tasks[:5]
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"summary": summary, "errors": errors, "tasks": results},
            indent=2,
        )
    )

    # ---------------- markdown report ----------------------------------------
    lines: list[str] = []
    lines.append("# S7 No-op Node Elimination Census")
    lines.append("")
    lines.append(
        f"Snapshot: `artifacts/research_snapshot/` — {len(paths)} models, "
        f"{len(results)} analyzed, {len(errors)} errors."
    )
    lines.append("")
    lines.append("Scoring: `score = max(1, 25 - ln(max(1, memory + params)))`. ")
    lines.append(
        "Memory here is the STATIC inferred-shape memory (the official scorer "
        "additionally maxes with the ORT profiler trace; shapes are fixed for "
        "these models so the static value is the operative one)."
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Total no-op nodes detected: **{total_noops}** "
                 f"across **{tasks_with_noops}/{len(results)}** tasks")
    lines.append(f"- Total removable intermediate bytes: **{total_bytes:,}**")
    lines.append(f"- Total removable params (orphaned constants): "
                 f"**{total_params_removed:,}**")
    lines.append(f"- Estimated total score gain: **{total_score_delta:+.4f}** "
                 f"(sum over tasks)")
    lines.append(f"- Graph-output rewire cases (flagged): {flagged} "
                 f"({flagged_bytes:,} bytes)")
    lines.append("")
    lines.append("## No-ops by category")
    lines.append("")
    lines.append("| Category | Count | Attributed bytes |")
    lines.append("|---|---:|---:|")
    for cat in CATEGORIES:
        lines.append(
            f"| {cat} | {total_by_cat.get(cat, 0)} | "
            f"{bytes_by_cat.get(cat, 0):,} |"
        )
    lines.append("")
    lines.append(
        "Attributed bytes = each no-op's own output tensor (producer tensor "
        "for graph-output cases). The authoritative total above comes from "
        "full graph simulation (chains resolved, orphaned constants dropped), "
        "so the per-category column can differ slightly from the total."
    )
    lines.append("")
    lines.append("## Global op_type histogram (top 30, all 400 models)")
    lines.append("")
    lines.append("| op_type | count |")
    lines.append("|---|---:|")
    for op, c in global_hist.most_common(30):
        lines.append(f"| {op} | {c} |")
    lines.append("")
    lines.append("## Top 20 tasks by estimated score gain")
    lines.append("")
    lines.append(
        "| Task | No-ops | Removable bytes | Removable params | "
        "Score before | Score after | Delta |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in top_tasks:
        lines.append(
            f"| {r['task']} | {r['noop_count']} | {r['removable_bytes']:,} | "
            f"{r['removable_params']:,} | {r['score_before']:.4f} | "
            f"{r['score_after']:.4f} | {r['score_delta']:+.4f} |"
        )
    lines.append("")
    lines.append("## Per-task results (tasks with at least one no-op)")
    lines.append("")
    lines.append(
        "| Task | No-ops | Categories | Removable bytes | "
        "Removable params | Score delta |"
    )
    lines.append("|---|---:|---|---:|---:|---:|")
    for r in results:
        if r["noop_count"] == 0:
            continue
        cats = ", ".join(
            f"{k}:{v}" for k, v in sorted(r["noops_by_category"].items())
        )
        lines.append(
            f"| {r['task']} | {r['noop_count']} | {cats} | "
            f"{r['removable_bytes']:,} | {r['removable_params']:,} | "
            f"{r['score_delta']:+.4f} |"
        )
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- {e['task']}: {e['error']}")
        lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Add/Sub-zero and Mul/Div-one removals are sign-exact for the final "
        "`> 0.0` threshold but can flip the sign of float zeros "
        "(`-0.0 + 0.0 = +0.0`) feeding downstream ops; any actual transform "
        "must be verified with `outputs_bit_identical` / gold verification."
    )
    lines.append(
        "- Broadcast safety enforced: zero/one-operand eliminations require "
        "the surviving input's shape to equal the output shape."
    )
    lines.append(
        "- Orphaned-constant cleanup is one pass (constants whose only "
        "consumers were removed no-ops); deeper dead-code elimination is a "
        "separate opportunity."
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))

    # ---------------- console summary -----------------------------------------
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
