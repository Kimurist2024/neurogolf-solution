#!/usr/bin/env python3
"""Find strictly cheaper, behavior-preserving shaves of a baseline ZIP.

Only graph rewrites that are exact by construction are attempted:

* remove nodes unreachable from graph outputs;
* remove unused initializers;
* merge byte-identical non-I/O initializers; and
* run the repository's exact no-op eliminator.

This script never mutates the baseline ZIP or root submission artifacts.  It
writes one ONNX per strict cost winner plus a JSON manifest under ``--out-dir``.
Behavioral differential testing remains a separate mandatory adoption gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.rank_dir import cost_of
from scripts.lib import optimizations


def _param_count(tensor: onnx.TensorProto) -> int:
    return math.prod(tensor.dims) if tensor.dims else 1


def prune_dead(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    """Remove the pure ONNX subgraph that cannot reach a graph output."""
    result = copy.deepcopy(model)
    graph = result.graph
    needed = {value.name for value in graph.output}
    live_indices: set[int] = set()
    for index in range(len(graph.node) - 1, -1, -1):
        node = graph.node[index]
        if any(name and name in needed for name in node.output):
            live_indices.add(index)
            needed.update(name for name in node.input if name)

    dead_nodes = [
        node for index, node in enumerate(graph.node) if index not in live_indices
    ]
    live_nodes = [
        node for index, node in enumerate(graph.node) if index in live_indices
    ]
    del graph.node[:]
    graph.node.extend(live_nodes)

    protected = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {name for node in graph.node for name in node.input if name}
    )
    dead_initializers = [
        tensor for tensor in graph.initializer if tensor.name not in protected
    ]
    kept_initializers = [
        tensor for tensor in graph.initializer if tensor.name in protected
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    live_names = (
        protected
        | {name for node in graph.node for name in node.output if name}
    )
    kept_value_info = [vi for vi in graph.value_info if vi.name in live_names]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)

    return result, {
        "dead_nodes": len(dead_nodes),
        "dead_node_ops": [node.op_type for node in dead_nodes],
        "dead_initializers": len(dead_initializers),
        "dead_initializer_params": sum(
            _param_count(tensor) for tensor in dead_initializers
        ),
    }


def dedup_initializers(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, int]]:
    """Merge exactly equal initializers without touching graph I/O names."""
    result = copy.deepcopy(model)
    graph = result.graph
    io_names = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
    )
    groups: dict[tuple[str, tuple[int, ...], bytes], list[onnx.TensorProto]] = {}
    for tensor in graph.initializer:
        if tensor.name in io_names:
            continue
        array = numpy_helper.to_array(tensor)
        key = (array.dtype.str, tuple(array.shape), array.tobytes())
        groups.setdefault(key, []).append(tensor)

    replacements: dict[str, str] = {}
    saved_params = 0
    for tensors in groups.values():
        if len(tensors) < 2:
            continue
        canonical = min(tensors, key=lambda tensor: (len(tensor.name), tensor.name))
        for tensor in tensors:
            if tensor.name == canonical.name:
                continue
            replacements[tensor.name] = canonical.name
            saved_params += _param_count(tensor)

    if replacements:
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name in replacements:
                    node.input[index] = replacements[name]
        kept = [
            tensor for tensor in graph.initializer
            if tensor.name not in replacements
        ]
        del graph.initializer[:]
        graph.initializer.extend(kept)

    return result, {
        "deduplicated_initializers": len(replacements),
        "deduplicated_params": saved_params,
    }


def validate_structure(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if model.functions:
        raise ValueError("model-local functions are not allowed")
    if model.graph.sparse_initializer:
        raise ValueError("sparse initializers are not allowed")
    banned = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
    for node in model.graph.node:
        if node.op_type in banned or "Sequence" in node.op_type:
            raise ValueError(f"banned op: {node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError(f"nested graph: {node.name or node.op_type}")


def optimize(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    current, dead_stats = prune_dead(model)
    current, dedup_stats = dedup_initializers(current)
    current, noop_stats = optimizations.g3_remove_noops(current)
    current, final_dead_stats = prune_dead(current)
    validate_structure(current)
    return current, {
        **dead_stats,
        **dedup_stats,
        "noop": noop_stats,
        "post_noop_dead": final_dead_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_costs = json.loads(args.base_costs.read_text())["costs"]
    winners: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            try:
                original_bytes = archive.read(member)
                original = onnx.load_model_from_string(original_bytes)
                candidate, stats = optimize(original)
                candidate_bytes = candidate.SerializeToString()
                if candidate_bytes == original_bytes:
                    continue

                with tempfile.TemporaryDirectory(prefix=f"exact_shave_{task:03d}_") as tmp:
                    path = Path(tmp) / member
                    onnx.save(candidate, path)
                    memory, params, candidate_cost = cost_of(str(path))
                baseline_cost = int(base_costs[str(task)])
                if candidate_cost < 0 or candidate_cost >= baseline_cost:
                    continue

                output = args.out_dir / member
                onnx.save(candidate, output)
                gain = math.log(baseline_cost / candidate_cost)
                winners.append({
                    "task": task,
                    "path": str(output),
                    "baseline_cost": baseline_cost,
                    "candidate_cost": candidate_cost,
                    "candidate_memory": memory,
                    "candidate_params": params,
                    "projected_gain": gain,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "stats": stats,
                })
                print(
                    f"task{task:03d}: {baseline_cost}->{candidate_cost} "
                    f"gain={gain:.9f}"
                )
            except Exception as exc:  # fail closed per task
                failures.append({"task": task, "error": repr(exc)})

    payload = {
        "baseline": str(args.baseline),
        "winners": winners,
        "winner_count": len(winners),
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
        "failures": failures,
    }
    manifest = args.out_dir / "manifest_pre_differential.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "winner_count": payload["winner_count"],
        "projected_gain": payload["projected_gain"],
        "failure_count": len(failures),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
