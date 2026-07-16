#!/usr/bin/env python3
"""Same-static-type/shape node bypass sweep for all cost-400..500 tasks."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent / "bypass"
ROOT = HERE.parents[3]


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MAIN = import_path(
    "restart8018_91_bypass_main",
    ROOT / "scripts/golf/restart8018_91_lane_high/worker.py",
)
BASE = MAIN.BASE
BAND = (
    (102, 493), (156, 483), (374, 481), (25, 472), (250, 468),
    (270, 465), (62, 442), (324, 434), (275, 428), (308, 427),
    (8, 421), (184, 421), (333, 421), (268, 420), (112, 418),
    (134, 417), (377, 409), (354, 403),
)
ELIGIBLE = tuple(task for task, _cost in BAND)
COSTS = dict(BAND)
BASE.HERE = HERE
BASE.BAND = BAND
BASE.ELIGIBLE = ELIGIBLE
BASE.COSTS = COSTS
BASE.PRIVATE_ZERO_OR_UNSOUND = set()
BASE.EXPLICIT_LATEST_LB_BLACK = set()
BASE.CHANGED_FROM_8011_05 = set(ELIGIBLE)


def signature(value: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]] | None:
    if not value.type.HasField("tensor_type"):
        return None
    tensor = value.type.tensor_type
    dims = tensor.shape.dim
    if any(not dim.HasField("dim_value") or int(dim.dim_value) <= 0 for dim in dims):
        return None
    return int(tensor.elem_type), tuple(int(dim.dim_value) for dim in dims)


def variants(task: int, source: onnx.ModelProto):
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(source), strict_mode=True)
    typed = {
        value.name: value
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.output)
            + list(inferred.graph.value_info)
        )
    }
    for init in inferred.graph.initializer:
        typed[init.name] = onnx.helper.make_tensor_value_info(
            init.name, init.data_type, list(init.dims)
        )
    graph_outputs = {value.name for value in source.graph.output}
    for index, node in enumerate(source.graph.node):
        outputs = [name for name in node.output if name]
        if len(outputs) != 1 or outputs[0] in graph_outputs:
            continue
        output = outputs[0]
        out_sig = signature(typed[output]) if output in typed else None
        if out_sig is None:
            continue
        for slot, input_name in enumerate(node.input):
            if not input_name or input_name == output or input_name not in typed:
                continue
            if signature(typed[input_name]) != out_sig:
                continue
            model = copy.deepcopy(source)
            del model.graph.node[index]
            for consumer in model.graph.node:
                for pos, value in enumerate(consumer.input):
                    if value == output:
                        consumer.input[pos] = input_name
            kept_vi = [value for value in model.graph.value_info if value.name != output]
            del model.graph.value_info[:]
            model.graph.value_info.extend(kept_vi)
            used = {value for n in model.graph.node for value in n.input if value}
            kept_init = [item for item in model.graph.initializer if item.name in used]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept_init)
            data = model.SerializeToString()
            yield data, {
                "name": f"task{task:03d}_bypass_{index}_{slot}",
                "family": "same_static_signature_bypass",
                "detail": f"remove {node.op_type}[{index}] and route input {slot}",
                "node_index": index,
                "node_type": node.op_type,
                "input_slot": slot,
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    worker = BASE.Worker(args.worker)
    worker.reprofile_authority()
    generated = 0
    with zipfile.ZipFile(BASE.AUTHORITY) as archive:
        for task in worker.tasks:
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            for data, meta in variants(task, model):
                generated += 1
                worker.consider(task, data, meta)
    finalists = worker.full_audit()
    strict = []
    accepted = []
    for row in finalists:
        path = ROOT / row["saved_path"]
        gate = MAIN.strict_gate(path, int(row["task"]), int(row["authority_cost"]))
        strict.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)
    payload = {
        "worker": args.worker,
        "assigned_tasks": list(worker.tasks),
        "generated": generated,
        "task_rows": [worker.task_rows[task] for task in worker.tasks],
        "counters": dict(worker.counters),
        "pre_strict_finalists": finalists,
        "strict_gates": strict,
        "finalists": accepted,
    }
    (HERE / f"worker_{args.worker}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "worker": args.worker,
        "tasks": list(worker.tasks),
        "generated": generated,
        "strict_winners": [
            {"task": row["task"], "cost": row["candidate_cost"],
             "gain": row["score_gain"], "path": row["saved_path"]}
            for row in accepted
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
