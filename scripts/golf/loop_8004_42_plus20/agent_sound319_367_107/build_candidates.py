#!/usr/bin/env python3
"""Build exact-shave and truthful-control candidates for tasks 319/367."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8008.14.zip"
AUTHORITY_SHA256 = "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6"
COSTS = {319: 1003, 367: 2179}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

scan_path = ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py"
spec = importlib.util.spec_from_file_location("lane107_exact", scan_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load exact transform library")
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_candidate(task: int, label: str, model: onnx.ModelProto, rows: list[dict[str, Any]]) -> None:
    data = model.SerializeToString()
    sha = digest(data)
    path = HERE / "candidates" / f"task{task:03d}_{label}_{sha[:12]}.onnx"
    path.write_bytes(data)
    rows.append({
        "task": task,
        "label": label,
        "path": relative(path),
        "sha256": sha,
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    })


def runtime_shapes(task: int, model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    output_names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in output_names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                output_names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sanitized = scoring.sanitize_model(traced)
    if sanitized is None:
        raise RuntimeError("sanitize failed while tracing")
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    example = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    if example is None:
        raise RuntimeError("missing benchmark")
    values = session.run(
        [item.name for item in session.get_outputs()],
        {session.get_inputs()[0].name: example["input"]},
    )
    return {name: list(np.asarray(value).shape) for name, value in zip(output_names, values)}


def set_shape(value: onnx.ValueInfoProto, dims: list[int]) -> None:
    del value.type.tensor_type.shape.dim[:]
    for size in dims:
        value.type.tensor_type.shape.dim.add().dim_value = int(size)


def honest_metadata(task: int, base: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(base)
    shapes = runtime_shapes(task, model)
    changed = []
    for value in list(model.graph.value_info) + list(model.graph.output):
        actual = shapes.get(value.name)
        if actual is None:
            continue
        before = [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        if before != actual:
            set_shape(value, actual)
            changed.append({"tensor": value.name, "before": before, "after": actual})
    return model, {"changed_shapes": changed, "changed_count": len(changed)}


def fold_scalar_shape_chain(base: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    """Fold Size/Add nodes whose inputs are compile-time initializers."""
    model = copy.deepcopy(base)
    arrays: dict[str, np.ndarray] = {}
    for item in model.graph.initializer:
        try:
            arrays[item.name] = numpy_helper.to_array(item)
        except Exception:
            pass
    nodes = list(model.graph.node)
    remove: set[int] = set()
    actions = []
    for index, node in enumerate(nodes):
        result = None
        if node.op_type == "Size" and len(node.input) == 1 and node.input[0] in arrays:
            result = np.asarray(arrays[node.input[0]].size, dtype=np.int64)
        elif node.op_type == "Add" and len(node.input) == 2 and all(name in arrays for name in node.input):
            result = np.add(arrays[node.input[0]], arrays[node.input[1]])
        if result is None or len(node.output) != 1:
            continue
        name = node.output[0]
        result = np.ascontiguousarray(result)
        arrays[name] = result
        model.graph.initializer.append(numpy_helper.from_array(result, name))
        for value in model.graph.value_info:
            if value.name == name:
                set_shape(value, list(result.shape))
        remove.add(index)
        actions.append({
            "node_index": index,
            "op": node.op_type,
            "output": name,
            "value": result.tolist(),
            "proof": "all inputs are compile-time constants",
        })
    if remove:
        del model.graph.node[:]
        model.graph.node.extend(node for index, node in enumerate(nodes) if index not in remove)
    return model, actions


def main() -> int:
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority hash mismatch")
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in COSTS}

    rows: list[dict[str, Any]] = []
    build_notes: dict[str, Any] = {}
    authorities: dict[int, onnx.ModelProto] = {}
    for task, data in payloads.items():
        model = onnx.load_model_from_string(data)
        authorities[task] = model
        write_candidate(task, "authority", model, rows)
        honest, note = honest_metadata(task, model)
        build_notes[f"task{task}_honest_metadata"] = note
        write_candidate(task, "honest_metadata", honest, rows)

        for kind in ("cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb", "combined"):
            transformed, actions = exact.transform(model, kind)
            if actions["semantic_action_count"] or actions["metadata_action_count"]:
                write_candidate(task, f"exact_{kind}", transformed, rows)
                build_notes[f"task{task}_exact_{kind}"] = actions

    scalar_folded, scalar_actions = fold_scalar_shape_chain(authorities[367])
    build_notes["task367_scalar_shape_fold"] = scalar_actions
    if scalar_actions:
        write_candidate(367, "exact_scalar_shape_fold", scalar_folded, rows)
        try:
            honest_fold, note = honest_metadata(367, scalar_folded)
            build_notes["task367_scalar_shape_fold_honest_metadata"] = note
            write_candidate(367, "exact_scalar_shape_fold_honest_metadata", honest_fold, rows)
        except Exception as exc:  # noqa: BLE001
            build_notes["task367_scalar_shape_fold_honest_metadata_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    truthful_source = (
        ROOT
        / "scripts/golf/loop_8004_42_plus20/agent_rebuild_mid5/"
        / "candidate_task367_truthful_rowmask.onnx"
    )
    truthful = onnx.load(truthful_source)
    write_candidate(367, "truthful_true_rule_control", truthful, rows)
    for kind in ("cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb", "combined"):
        transformed, actions = exact.transform(truthful, kind)
        if actions["semantic_action_count"] or actions["metadata_action_count"]:
            write_candidate(367, f"truthful_exact_{kind}", transformed, rows)
            build_notes[f"task367_truthful_exact_{kind}"] = actions
    truth_fold, truth_actions = fold_scalar_shape_chain(truthful)
    build_notes["task367_truthful_control_scalar_fold"] = truth_actions
    if truth_actions:
        write_candidate(367, "truthful_true_rule_control_scalar_fold", truth_fold, rows)

    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((row["task"], row["sha256"]), row)
    report = {
        "authority_zip": "submission_base_8008.14.zip",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "authority_costs": COSTS,
        "candidate_count_before_dedupe": len(rows),
        "candidate_count": len(unique),
        "candidates": list(unique.values()),
        "build_notes": build_notes,
    }
    (HERE / "audit/build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "by_task": {
            str(task): sum(row["task"] == task for row in unique.values()) for task in COSTS
        },
        "labels": [row["label"] for row in unique.values()],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
