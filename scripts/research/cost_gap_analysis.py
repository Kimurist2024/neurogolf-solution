#!/usr/bin/env python
"""Cost-gap analysis vs the ~7700-point NeuroGolf leaders.

1. Per-task cost/score distribution from artifacts/reports/run-012.json
   (chosen candidate): cost deciles + score histogram.
2. Per-model census of artifacts/optimized_pre_merge/ (READ-ONLY): node count,
   intermediate-tensor count, memory vs params split (shape-inference based,
   reusing scripts/research/s6_memory_census.py helpers).
3. Leader math: score=max(1,25-ln(cost)) -> leader avg cost; cost-bucket counts;
   scenario table where every task with cost>T is reduced to T for
   T in {100, 900, 2000} (full set and top-50-only variants).
4. Priority queue: 50 highest-cost tasks (id, cost, nodes, op histogram);
   for the top 10, programmatic eyeball of inputs/neurogolf-2026/taskNNN.json
   train pairs -> transformation-family guess.

Writes docs/research/cost-gap-analysis.md and .json.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import onnx
from onnx import shape_inference

REPO = Path("/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf")
RUN_REPORT = REPO / "artifacts" / "reports" / "run-012.json"
MODEL_DIR = REPO / "artifacts" / "optimized_pre_merge"  # read-only snapshot
INPUT_DIR = REPO / "inputs" / "neurogolf-2026"
OUT_MD = REPO / "docs" / "research" / "cost-gap-analysis.md"
OUT_JSON = REPO / "docs" / "research" / "cost-gap-analysis.json"

LEADER_TOTAL = 7700.0
N_TASKS = 400
SCENARIO_TARGETS = (100, 900, 2000)
BUCKET_EDGES = (100, 314, 1000, 5000, 20000, 50000)
TOP_N = 50
EYEBALL_N = 10

# ---- reuse s6_memory_census helpers (count_params, shape_elems, dtype_itemsize)
_spec = importlib.util.spec_from_file_location(
    "s6_memory_census", REPO / "scripts" / "research" / "s6_memory_census.py"
)
s6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s6)

EXCLUDED_NAMES = {"input", "output"}


def score_of(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


# ---------------------------------------------------------------- run-012 load
def load_run() -> list[dict]:
    data = json.loads(RUN_REPORT.read_text())
    rows = []
    for t in data["tasks"]:
        chosen = t.get("chosen") or t.get("baseline")
        rows.append(
            {
                "task": int(t["task"]),
                "cost": int(chosen["cost"]),
                "score": float(chosen["score"]),
                "memory": int(chosen.get("memory", 0)),
                "params": int(chosen.get("params", 0)),
                "source": chosen.get("source", ""),
            }
        )
    rows.sort(key=lambda r: r["task"])
    return rows


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ------------------------------------------------------------ model census
def census_model(path: Path) -> dict:
    model = onnx.load(str(path))
    graph = model.graph
    n_nodes = len(graph.node)
    op_hist = Counter(n.op_type for n in graph.node)
    params = s6.count_params(graph)

    inter_names = []
    seen = set()
    for node in graph.node:
        for out in node.output:
            if out and out not in EXCLUDED_NAMES and out not in seen:
                seen.add(out)
                inter_names.append(out)

    memory = None
    n_unknown = 0
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        type_map = {}
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(
            inferred.graph.output
        ):
            if vi.type.HasField("tensor_type"):
                type_map[vi.name] = vi.type.tensor_type
        memory = 0
        for name in inter_names:
            tt = type_map.get(name)
            if tt is None:
                n_unknown += 1
                continue
            itemsize = s6.dtype_itemsize(tt.elem_type)
            elems = s6.shape_elems(tt)
            if itemsize is None or elems is None:
                n_unknown += 1
                continue
            memory += elems * itemsize
    except Exception:
        memory = None

    return {
        "n_nodes": n_nodes,
        "n_intermediates": len(inter_names),
        "params": params,
        "memory_inferred": memory,
        "n_unknown_tensors": n_unknown,
        "op_hist": dict(op_hist.most_common()),
    }


# ------------------------------------------------------- task family eyeball
def grid_shape(g: list[list[int]]) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def is_subgrid(small: list[list[int]], big: list[list[int]]) -> bool:
    sh, sw = grid_shape(small)
    bh, bw = grid_shape(big)
    if sh > bh or sw > bw:
        return False
    for r0 in range(bh - sh + 1):
        for c0 in range(bw - sw + 1):
            if all(big[r0 + r][c0 + c] == small[r][c] for r in range(sh) for c in range(sw)):
                return True
    return False


def classify_task(task_id: int) -> dict:
    path = INPUT_DIR / f"task{task_id:03d}.json"
    data = json.loads(path.read_text())
    pairs = data.get("train", []) + data.get("test", [])
    info = {
        "task": task_id,
        "n_train": len(data.get("train", [])),
        "n_arc_gen": len(data.get("arc-gen", [])),
    }
    in_shapes = [grid_shape(p["input"]) for p in pairs]
    out_shapes = [grid_shape(p["output"]) for p in pairs]
    in_colors = sorted({c for p in pairs for row in p["input"] for c in row})
    out_colors = sorted({c for p in pairs for row in p["output"] for c in row})
    info["in_shapes"] = in_shapes
    info["out_shapes"] = out_shapes
    info["in_colors"] = in_colors
    info["out_colors"] = out_colors

    same_size = all(i == o for i, o in zip(in_shapes, out_shapes))
    info["same_size"] = same_size

    family = "object-level/other"
    detail = ""
    if same_size:
        # pointwise color-map check (across train+test, then verify on arc-gen sample)
        cmap: dict[int, int] = {}
        pointwise = True
        check_pairs = pairs + data.get("arc-gen", [])[:50]
        for p in check_pairs:
            if grid_shape(p["input"]) != grid_shape(p["output"]):
                pointwise = False
                break
            for ri, ro in zip(p["input"], p["output"]):
                for a, b in zip(ri, ro):
                    if cmap.setdefault(a, b) != b:
                        pointwise = False
                        break
                if not pointwise:
                    break
            if not pointwise:
                break
        if pointwise:
            identity = all(cmap.get(k) == k for k in cmap)
            family = "same-size pointwise recolor" + (" (identity)" if identity else "")
            detail = f"color map {cmap}"
        else:
            family = "same-size cell-wise (non-pointwise: local/structural)"
            detail = "output shape == input shape but cell value depends on context"
    else:
        ratios = set()
        ok = True
        for (ih, iw), (oh, ow) in zip(in_shapes, out_shapes):
            if oh % ih == 0 and ow % iw == 0 and (oh > ih or ow > iw):
                ratios.add((oh // ih, ow // iw))
            else:
                ok = False
                break
        if ok and len(ratios) == 1:
            family = "upscale/tiling"
            detail = f"output = input x {ratios.pop()}"
        else:
            ratios = set()
            ok = True
            for (ih, iw), (oh, ow) in zip(in_shapes, out_shapes):
                if oh and ow and ih % oh == 0 and iw % ow == 0 and (ih > oh or iw > ow):
                    ratios.add((ih // oh, iw // ow))
                else:
                    ok = False
                    break
            if ok and len(ratios) == 1:
                family = "downscale/block-reduce"
                detail = f"input = output x {ratios.pop()}"
            elif len(set(out_shapes)) == 1:
                family = "fixed-size output"
                detail = f"all outputs {out_shapes[0]}"
            elif all(is_subgrid(p["output"], p["input"]) for p in pairs):
                family = "crop/extract subgrid"
            else:
                family = "size-changing object-level/other"
    info["family"] = family
    info["detail"] = detail
    return info


# -------------------------------------------------------------------- main
def main() -> None:
    rows = load_run()
    assert len(rows) == N_TASKS, f"expected {N_TASKS} tasks, got {len(rows)}"
    total_score = sum(r["score"] for r in rows)
    costs = sorted(r["cost"] for r in rows)

    deciles = {f"p{int(q*100)}": percentile(costs, q) for q in [i / 10 for i in range(11)]}
    score_hist = Counter(int(r["score"]) for r in rows)  # floor bins

    # leader math
    leader_avg_score = LEADER_TOTAL / N_TASKS
    leader_avg_cost = math.exp(25.0 - leader_avg_score)
    our_avg_cost = sum(costs) / len(costs)

    # cost buckets
    buckets = {f"<= {e}": sum(1 for c in costs if c <= e) for e in BUCKET_EDGES}
    buckets["> 50000"] = sum(1 for c in costs if c > 50000)

    # scenarios
    by_cost_desc = sorted(rows, key=lambda r: r["cost"], reverse=True)
    top50_ids = {r["task"] for r in by_cost_desc[:TOP_N]}
    scenarios = []
    for target in SCENARIO_TARGETS:
        affected = [r for r in rows if r["cost"] > target]
        new_total = sum(
            score_of(target) if r["cost"] > target else r["score"] for r in rows
        )
        top50_total = sum(
            score_of(target) if (r["task"] in top50_ids and r["cost"] > target) else r["score"]
            for r in rows
        )
        scenarios.append(
            {
                "target_cost": target,
                "score_at_target": score_of(target),
                "n_tasks_affected": len(affected),
                "total_score_all_affected": new_total,
                "gain_all": new_total - total_score,
                "total_score_top50_only": top50_total,
                "gain_top50_only": top50_total - total_score,
            }
        )

    # model census for top-50 (and totals over all 400 for context)
    print("censusing top-50 models...")
    top50 = []
    for r in by_cost_desc[:TOP_N]:
        path = MODEL_DIR / f"task{r['task']:03d}.onnx"
        c = census_model(path) if path.exists() else {"error": "missing"}
        top50.append({**r, **c})

    print("censusing all 400 models (nodes/params/memory)...")
    full_census = {}
    for r in rows:
        path = MODEL_DIR / f"task{r['task']:03d}.onnx"
        if path.exists():
            full_census[r["task"]] = census_model(path)
    agg_ops = Counter()
    for c in full_census.values():
        agg_ops.update(c["op_hist"])
    total_nodes = sum(c["n_nodes"] for c in full_census.values())
    total_mem = sum(c["memory_inferred"] or 0 for c in full_census.values())
    total_params = sum(c["params"] for c in full_census.values())

    # eyeball top-10
    print("classifying top-10 tasks...")
    eyeball = []
    for r in by_cost_desc[:EYEBALL_N]:
        try:
            eyeball.append(classify_task(r["task"]))
        except Exception as exc:
            eyeball.append({"task": r["task"], "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "meta": {
            "run_report": str(RUN_REPORT),
            "model_dir": str(MODEL_DIR),
            "score_formula": "max(1, 25 - ln(cost)); cost = memory + params",
            "leader_total_assumed": LEADER_TOTAL,
        },
        "current": {
            "total_score": total_score,
            "avg_score": total_score / N_TASKS,
            "avg_cost": our_avg_cost,
            "median_cost": percentile(costs, 0.5),
            "cost_deciles": deciles,
            "score_histogram_floor_bins": dict(sorted(score_hist.items())),
            "cost_buckets": buckets,
        },
        "leader_math": {
            "leader_avg_score": leader_avg_score,
            "leader_implied_avg_cost": leader_avg_cost,
            "gap_points": LEADER_TOTAL - total_score,
        },
        "scenarios": scenarios,
        "census_aggregate": {
            "total_nodes": total_nodes,
            "total_intermediate_memory": total_mem,
            "total_params": total_params,
            "op_histogram": dict(agg_ops.most_common()),
        },
        "priority_queue_top50": top50,
        "eyeball_top10": eyeball,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1))

    # ------------------------------ markdown
    L = []
    a = L.append
    a("# Cost-Gap Analysis vs 7700-Point Leaders (run-012)")
    a("")
    a(f"- Our total: **{total_score:.2f}** (avg {total_score/N_TASKS:.3f}/task, "
      f"avg cost {our_avg_cost:,.0f}, median cost {percentile(costs,0.5):,.0f})")
    a(f"- Leaders: **{LEADER_TOTAL:.0f}** -> avg {leader_avg_score:.3f}/task -> "
      f"implied avg cost **e^{25-leader_avg_score:.2f} = {leader_avg_cost:,.0f}**")
    a(f"- Gap: **{LEADER_TOTAL - total_score:.2f} points**")
    a("")
    a("## Cost deciles")
    a("")
    a("| " + " | ".join(deciles.keys()) + " |")
    a("|" + "---:|" * len(deciles))
    a("| " + " | ".join(f"{v:,.0f}" for v in deciles.values()) + " |")
    a("")
    a("## Score histogram (floor bins)")
    a("")
    a("| score bin | tasks |")
    a("|---:|---:|")
    for k in sorted(score_hist):
        a(f"| [{k},{k+1}) | {score_hist[k]} |")
    a("")
    a("## Cost buckets")
    a("")
    a("| bucket | tasks |")
    a("|---|---:|")
    for k, v in buckets.items():
        a(f"| {k} | {v} |")
    a("")
    a("## Scenario table — reduce every task with cost > T to exactly T")
    a("")
    a("| target T | score@T | tasks affected | total (all affected) | gain (all) "
      "| total (top-50 only) | gain (top-50 only) |")
    a("|---:|---:|---:|---:|---:|---:|---:|")
    for s in scenarios:
        a(f"| {s['target_cost']:,} | {s['score_at_target']:.3f} | {s['n_tasks_affected']} "
          f"| {s['total_score_all_affected']:.2f} | {s['gain_all']:+.2f} "
          f"| {s['total_score_top50_only']:.2f} | {s['gain_top50_only']:+.2f} |")
    a("")
    a("## Whole-set census (optimized_pre_merge, 400 models)")
    a("")
    a(f"- Total nodes: {total_nodes:,}; total intermediate memory {total_mem:,} B; "
      f"total params {total_params:,}")
    a("- Top ops: " + ", ".join(f"{k} x{v}" for k, v in agg_ops.most_common(15)))
    a("")
    a("## Priority queue — 50 highest-cost tasks")
    a("")
    a("| task | cost | score | memory | params | nodes | interm. | top ops |")
    a("|---:|---:|---:|---:|---:|---:|---:|---|")
    for t in top50:
        ops = ", ".join(f"{k}x{v}" for k, v in list(t.get("op_hist", {}).items())[:6])
        a(f"| {t['task']} | {t['cost']:,} | {t['score']:.2f} | {t['memory']:,} "
          f"| {t['params']:,} | {t.get('n_nodes','?')} | {t.get('n_intermediates','?')} | {ops} |")
    a("")
    a("## Eyeball of top-10 priority tasks (train+test pairs)")
    a("")
    a("| task | cost | family | same-size | in shapes (sample) | out shapes (sample) "
      "| colors in/out | detail |")
    a("|---:|---:|---|---|---|---|---|---|")
    cost_by_task = {r["task"]: r["cost"] for r in rows}
    for e in eyeball:
        if "error" in e:
            a(f"| {e['task']} | {cost_by_task.get(e['task'],0):,} | ERROR {e['error']} | | | | | |")
            continue
        ins = ", ".join(f"{h}x{w}" for h, w in e["in_shapes"][:4])
        outs = ", ".join(f"{h}x{w}" for h, w in e["out_shapes"][:4])
        a(f"| {e['task']} | {cost_by_task[e['task']]:,} | **{e['family']}** | {e['same_size']} "
          f"| {ins} | {outs} | {len(e['in_colors'])}/{len(e['out_colors'])} | {e['detail'][:80]} |")
    a("")
    a("Machine-readable data: `docs/research/cost-gap-analysis.json`.")
    a("")
    OUT_MD.write_text("\n".join(L))

    # console summary
    print(f"\ntotal_score={total_score:.2f} avg_cost={our_avg_cost:,.0f} "
          f"median_cost={percentile(costs,0.5):,.0f}")
    print("buckets:", buckets)
    print("scenarios:")
    for s in scenarios:
        print(f"  T={s['target_cost']}: all->{s['total_score_all_affected']:.2f} "
              f"({s['gain_all']:+.2f}), top50->{s['total_score_top50_only']:.2f} "
              f"({s['gain_top50_only']:+.2f}), affected={s['n_tasks_affected']}")
    print("top10 priority:")
    for t in top50[:10]:
        fam = next((e.get("family", "?") for e in eyeball if e["task"] == t["task"]), "?")
        print(f"  task{t['task']:03d}: cost={t['cost']:,} nodes={t.get('n_nodes')} fam={fam}")


if __name__ == "__main__":
    main()
