#!/usr/bin/env python3
"""Pinned 8012.15 cost>2500 scan using the audited three-worker engine.

This lane deliberately reuses the already reviewed scanner from lane409 and
only changes the immutable cost scope, exclusion catalogue, output directory,
and fresh seeds.  The protected root submission, score ledger, and others/
remain read-only.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import time
import math
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENGINE = ROOT / "scripts/golf/restart8012_cost1001_2500_3w_409/scan.py"
_ENGINE_MODULE = None
_ORIGINAL_WORKER_MAIN = None


def worker_main_skip_empty(args):
    """Avoid generating 4,000 fresh cases when structural candidates are zero."""
    if _ENGINE_MODULE is None or _ORIGINAL_WORKER_MAIN is None:
        raise RuntimeError("worker engine is not initialized")
    started = time.monotonic()
    tasks = [int(task) for task in args["tasks"]]
    nonempty = [task for task in tasks if args["candidates"][task]]
    empty = [task for task in tasks if not args["candidates"][task]]
    if nonempty:
        child_args = {
            **args,
            "tasks": nonempty,
            "costs": {task: args["costs"][task] for task in nonempty},
            "candidates": {task: args["candidates"][task] for task in nonempty},
            "authority_members": {
                task: args["authority_members"][task] for task in nonempty
            },
        }
        result = _ORIGINAL_WORKER_MAIN(child_args)
    else:
        result = {
            "worker_index": int(args["worker_index"]),
            "pid": os.getpid(),
            "tasks": [],
            "candidate_load": 0,
            "results": [],
            "finalists": [],
            "elapsed_seconds": 0.0,
        }
    for task in empty:
        data = args["authority_members"][task]
        model = onnx.load_model_from_string(data)
        params = int(_ENGINE_MODULE.parameter_count(model))
        cost = int(args["costs"][task])
        result["results"].append(
            {
                "task": task,
                "ledger_cost": cost,
                "authority": {
                    "member": f"task{task:03d}.onnx",
                    "sha256": _ENGINE_MODULE.sha256(data),
                    "bytes": len(data),
                    "profile": {
                        "memory": cost - params,
                        "params": params,
                        "cost": cost,
                        "correct": None,
                        "method": "pinned_ledger_plus_exact_parameter_count; no candidates",
                    },
                    "ledger_cost_matches_actual": True,
                },
                "known_counts": {},
                "screen_candidate_count": 0,
                "eligible_fresh_count": 0,
                "screen": [],
                "fresh_audits": [],
                "finalist": None,
                "empty_candidate_short_circuit": True,
                "elapsed_seconds": 0.0,
            }
        )
        print(
            {
                "worker": int(args["worker_index"]),
                "pid": os.getpid(),
                "task_done": task,
                "screened": 0,
                "eligible": 0,
                "winner": None,
                "short_circuit": "no structural candidates",
            },
            flush=True,
        )
    result["tasks"] = tasks
    result["candidate_load"] = sum(
        len(args["candidates"][task]) for task in tasks
    )
    result["results"].sort(key=lambda row: int(row["task"]))
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def load_engine():
    name = "restart8012_cost2501_plus_engine_411"
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(ENGINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    global _ENGINE_MODULE, _ORIGINAL_WORKER_MAIN
    engine = load_engine()
    _ENGINE_MODULE = engine
    engine.HERE = HERE
    engine.RANGE_MIN = 2_501
    engine.RANGE_MAX = 10**12
    engine.FRESH_PER_SEED = 2_000
    engine.MAX_FRESH_CANDIDATES_PER_TASK = 5

    # The final 51-task operational list plus every task explicitly documented
    # elsewhere in private_zero_tasks.md as black/unsafe/contaminating.  This
    # intentionally excludes task018 and task191 even though they are absent
    # from the final compact list, because both have explicit black evidence.
    engine.PRIVATE_ZERO_OR_UNSOUND = {
        9, 15, 18, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102,
        112, 133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178,
        182, 185, 187, 191, 192, 196, 198, 202, 204, 205, 208, 209,
        216, 219, 222, 233, 246, 255, 264, 277, 285, 286, 302, 319,
        325, 333, 343, 344, 346, 361, 365, 366, 372, 377, 379, 391,
        393, 396,
    }
    engine.LATEST_LB_BLACK = {70, 134, 202, 343}

    # Fail closed before any graph execution.  Some historical task076 models
    # are deliberately pathological and can spend unbounded time in a known
    # screen even though their declared output is the scalar cloak
    # [1,1,1,1].  Canonical I/O is a mandatory admission condition, so record
    # and remove those artifacts before the three runtime workers start.
    original_discover = engine.discover_candidates

    def discover_canonical_first(eligible, authority_members):
        candidates, meta = original_discover(eligible, authority_members)
        rejected = []

        def dims(value):
            return [
                int(dim.dim_value) if dim.HasField("dim_value") else None
                for dim in value.type.tensor_type.shape.dim
            ]

        for task, rows in candidates.items():
            kept = []
            for row in rows:
                reasons = []
                static_cost_floor = None
                try:
                    model = onnx.load_model_from_string(row["data"])
                    input_dims = dims(model.graph.input[0]) if len(model.graph.input) == 1 else None
                    output_dims = dims(model.graph.output[0]) if len(model.graph.output) == 1 else None
                except Exception as exc:  # fail closed
                    input_dims = output_dims = None
                    error = f"{type(exc).__name__}: {exc}"
                    reasons.append("load_error")
                else:
                    error = None
                    if input_dims != [1, 10, 30, 30] or output_dims != [1, 10, 30, 30]:
                        reasons.append("noncanonical_io")
                    try:
                        onnx.checker.check_model(model, full_check=True)
                    except Exception as exc:
                        reasons.append("full_checker")
                        error = f"{type(exc).__name__}: {exc}"
                    try:
                        inferred = onnx.shape_inference.infer_shapes(
                            model, strict_mode=True, data_prop=True
                        )
                    except Exception as exc:
                        reasons.append("strict_shape_data_prop")
                        error = f"{type(exc).__name__}: {exc}"
                    else:
                        values = {
                            value.name: value
                            for value in [
                                *inferred.graph.input,
                                *inferred.graph.value_info,
                                *inferred.graph.output,
                            ]
                        }
                        missing = []
                        nonstatic = []
                        graph_outputs = {value.name for value in inferred.graph.output}
                        for node in inferred.graph.node:
                            for name in node.output:
                                if not name or name in graph_outputs:
                                    continue
                                value = values.get(name)
                                if value is None:
                                    missing.append(name)
                                    continue
                                shape = dims(value)
                                if not shape or any(dim is None or dim <= 0 for dim in shape):
                                    nonstatic.append({"name": name, "shape": shape})
                        if missing:
                            reasons.append("missing_node_output_shape")
                        if nonstatic:
                            reasons.append("nonstatic_node_output_shape")
                        if not missing and not nonstatic:
                            # The official scorer sums every inferred node-output
                            # tensor and may only increase a tensor's charge from
                            # the runtime profile.  Therefore this inferred sum
                            # plus parameter count is a safe cost lower bound.
                            seen_outputs = set()
                            static_memory = 0
                            for node in inferred.graph.node:
                                for name in node.output:
                                    if (
                                        not name
                                        or name in graph_outputs
                                        or name in seen_outputs
                                    ):
                                        continue
                                    seen_outputs.add(name)
                                    value = values[name]
                                    shape = dims(value)
                                    dtype = onnx.helper.tensor_dtype_to_np_dtype(
                                        value.type.tensor_type.elem_type
                                    )
                                    static_memory += math.prod(shape) * dtype.itemsize
                            static_cost_floor = int(row["params"]) + static_memory
                            if static_cost_floor >= int(eligible[int(task)]):
                                reasons.append("inferred_static_cost_not_lower")
                if reasons:
                    rejected.append(
                        {
                            "task": int(task),
                            "sha256": row["sha256"],
                            "sources": row["sources"],
                            "input_shape": input_dims,
                            "output_shape": output_dims,
                            "error": error,
                            "inferred_static_cost_floor": static_cost_floor,
                            "authority_cost": int(eligible[int(task)]),
                            "reasons": sorted(set(reasons)),
                            "reason": "mandatory_static_structure_preexecution",
                        }
                    )
                else:
                    kept.append(row)
            candidates[task] = kept
        meta["preexecution_noncanonical_rejections"] = rejected
        meta["preexecution_noncanonical_rejection_count"] = len(rejected)
        meta["screen_candidates_after_canonical_gate"] = sum(
            len(rows) for rows in candidates.values()
        )
        return candidates, meta

    engine.discover_candidates = discover_canonical_first

    # The imported engine explicitly asks for spawn.  Because it is loaded by
    # file path under a lane-local module name, use fork so all three children
    # inherit that reviewed module without an import-path ambiguity.
    original_get_context = mp.get_context
    engine.mp.get_context = lambda _method=None: original_get_context("fork")
    _ORIGINAL_WORKER_MAIN = engine.worker_main
    engine.worker_main = worker_main_skip_empty
    engine.main()


if __name__ == "__main__":
    main()
