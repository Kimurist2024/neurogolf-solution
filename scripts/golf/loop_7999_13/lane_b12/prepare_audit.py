#!/usr/bin/env python3
"""Freeze and independently audit exact-base and B12 archive candidates."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


ZIP = ROOT / "submission_base_7999.13.zip"
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400"
TASKS = {
    254: {"hash": "a61f2674", "file": "task254_r01_static42.onnx", "archive_static": 42},
    267: {"hash": "aabf363d", "file": "task267_r01_static30.onnx", "archive_static": 30},
    322: {"hash": "d037b0a7", "file": "task322_r01_static19.onnx", "archive_static": 19},
    323: {"hash": "d06dbe63", "file": "task323_r01_static104.onnx", "archive_static": 104},
    372: {"hash": "e98196ab", "file": "task372_r01_static12.onnx", "archive_static": 12},
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TfIdfVectorizer", "Hardmax"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(item: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in item.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("")
    return result


def attr_value(attribute: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.reshape(-1).tolist(),
        }
    return value


def initializer_rows(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        preview: list[object] = []
        for value in array.reshape(-1)[:16].tolist():
            if isinstance(value, float) and not math.isfinite(value):
                preview.append("NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity"))
            else:
                preview.append(value)
        rows.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True,
                "nonfinite_elements": int(np.count_nonzero(~np.isfinite(array)))
                if array.dtype.kind in "fc"
                else 0,
                "preview": preview,
            }
        )
    return rows


def node_rows(model: onnx.ModelProto) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "op": node.op_type,
            "domain": node.domain,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {item.name: attr_value(item) for item in node.attribute},
            "input_multiplicity": dict(collections.Counter(node.input)),
        }
        for index, node in enumerate(model.graph.node)
    ]


def static_cost(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    io_names = {item.name for item in list(inferred.graph.input) + list(inferred.graph.output)}
    initializer_names = {item.name for item in inferred.graph.initializer}
    tensor_map = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    node_outputs = {name for node in inferred.graph.node for name in node.output if name}
    memory = 0
    for name in sorted(node_outputs):
        if name in io_names or name in initializer_names:
            continue
        item = tensor_map[name]
        shape = dims(item)
        if any(not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError(f"nonstatic tensor {name}: {shape}")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(item.type.tensor_type.elem_type)
        memory += math.prod(shape) * np.dtype(dtype).itemsize
    params = scoring.calculate_params(inferred)
    assert params is not None
    return {"memory": memory, "params": params, "cost": memory + params}


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    custom_domains = sorted(
        {item.domain for item in inferred.opset_import if item.domain not in {"", "ai.onnx"}}
    )
    banned = sorted(
        {
            node.op_type
            for node in inferred.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    nested = [
        {"node": node.op_type, "attribute": attr.name}
        for node in inferred.graph.node
        for attr in node.attribute
        if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
    ]
    nonstatic = []
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not item.type.HasField("tensor_type") or any(
            not dimension.HasField("dim_value") or dimension.dim_value <= 0
            for dimension in item.type.tensor_type.shape.dim
        ):
            nonstatic.append(item.name)
    giant = [
        {"node_index": index, "inputs": len(node.input)}
        for index, node in enumerate(inferred.graph.node)
        if node.op_type == "Einsum" and len(node.input) >= 8
    ]
    lookup = [node.op_type for node in inferred.graph.node if node.op_type in LOOKUP]
    nonfinite = [row["name"] for row in initializer_rows(inferred) if not row["finite"]]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "standard_domains": not custom_domains,
        "custom_domains": custom_domains,
        "functions": len(inferred.functions),
        "sparse_initializers": len(inferred.graph.sparse_initializer),
        "nested_graph_attributes": nested,
        "banned_or_sequence_ops": banned,
        "dynamic_or_nonpositive_tensors": nonstatic,
        "conv_bias_findings": check_conv_bias(inferred),
        "giant_einsum_nodes": giant,
        "lookup_ops": lookup,
        "nonfinite_initializers": nonfinite,
        "op_histogram": dict(collections.Counter(node.op_type for node in inferred.graph.node)),
        "node_count": len(inferred.graph.node),
        "value_info_count": len(inferred.graph.value_info),
        "static_cost": static_cost(inferred),
        "initializers": initializer_rows(inferred),
        "nodes": node_rows(inferred),
    }


def runtime_shapes(task: int, path: Path) -> dict[str, object]:
    original = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(original, strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    declared = {
        item.name: dims(item)
        for item in list(original.graph.value_info) + list(original.graph.output)
    }
    traced = copy.deepcopy(original)
    original_outputs = {item.name for item in traced.graph.output}
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert benchmark is not None
    values = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    tensors: list[dict[str, object]] = []
    mismatches = []
    for name, value in zip(names, values, strict=True):
        array = np.asarray(value)
        inferred_shape = dims(typed[name])
        actual_shape = list(array.shape)
        row = {
            "name": name,
            "graph_output": name in original_outputs,
            "declared_shape": declared.get(name),
            "inferred_shape": inferred_shape,
            "actual_shape": actual_shape,
            "dtype": str(array.dtype),
            "bytes": int(array.nbytes),
            "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True,
            "nonfinite_elements": int(np.count_nonzero(~np.isfinite(array)))
            if array.dtype.kind in "fc"
            else 0,
        }
        tensors.append(row)
        if inferred_shape != actual_shape or (
            name in declared and declared[name] != actual_shape
        ):
            mismatches.append(row)
    return {"all_node_outputs": tensors, "shape_mismatches": mismatches}


def make_session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(task: int, path: Path, disable_all: bool) -> dict[str, object]:
    session = make_session(path, disable_all)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result: dict[str, object] = {}
    total_nonfinite_cases = total_nonfinite_elements = 0
    min_positive = math.inf
    max_abs_finite = 0.0
    for subset in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            assert benchmark is not None
            try:
                raw = np.asarray(session.run([output_name], {input_name: benchmark["input"]})[0])
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            decoded = raw > 0.0
            if np.array_equal(decoded, benchmark["output"] > 0.0):
                right += 1
            else:
                wrong += 1
            if raw.dtype.kind in "fc":
                nonfinite = ~np.isfinite(raw)
                total_nonfinite_cases += int(nonfinite.any())
                total_nonfinite_elements += int(np.count_nonzero(nonfinite))
                finite = raw[np.isfinite(raw)]
                if finite.size:
                    max_abs_finite = max(max_abs_finite, float(np.abs(finite).max()))
                    positive = finite[finite > 0.0]
                    if positive.size:
                        min_positive = min(min_positive, float(positive.min()))
        result[subset] = {"right": right, "wrong": wrong, "errors": errors}
    result["total"] = {
        key: sum(int(result[subset][key]) for subset in ("train", "test", "arc-gen"))
        for key in ("right", "wrong", "errors")
    }
    result["raw_numeric"] = {
        "nonfinite_cases": total_nonfinite_cases,
        "nonfinite_elements": total_nonfinite_elements,
        "minimum_positive_finite": None if min_positive == math.inf else min_positive,
        "maximum_absolute_finite": max_abs_finite,
    }
    return result


def graph_diff(base: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, object]:
    base_init = {row["name"]: row for row in initializer_rows(base)}
    cand_init = {row["name"]: row for row in initializer_rows(candidate)}
    shared = sorted(set(base_init) & set(cand_init))
    return {
        "node_count": {"baseline": len(base.graph.node), "candidate": len(candidate.graph.node)},
        "initializer_names_removed": sorted(set(base_init) - set(cand_init)),
        "initializer_names_added": sorted(set(cand_init) - set(base_init)),
        "shared_initializer_changes": {
            name: {"baseline": base_init[name], "candidate": cand_init[name]}
            for name in shared
            if base_init[name]["sha256"] != cand_init[name]["sha256"]
            or base_init[name]["shape"] != cand_init[name]["shape"]
        },
        "baseline_nodes": node_rows(base),
        "candidate_nodes": node_rows(candidate),
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    baseline_dir = HERE / "baseline"
    candidate_dir = HERE / "candidates"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        for task, info in TASKS.items():
            (baseline_dir / f"task{task:03d}.onnx").write_bytes(
                archive.read(f"task{task:03d}.onnx")
            )
            shutil.copyfile(ARCHIVE / info["file"], candidate_dir / info["file"])
    archive_inventory = json.loads((ARCHIVE / "inventory.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "baseline_zip": str(ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": sha256(ZIP),
        "tasks": {},
    }
    for task, info in TASKS.items():
        baseline_path = baseline_dir / f"task{task:03d}.onnx"
        candidate_path = candidate_dir / info["file"]
        baseline = onnx.load(baseline_path)
        candidate = onnx.load(candidate_path)
        with_path = HERE / "score_work" / f"task{task:03d}"
        with_path.mkdir(parents=True, exist_ok=True)
        baseline_profile = scoring.score_and_verify(
            copy.deepcopy(baseline), task, str(with_path / "baseline"),
            label=f"base{task}", require_correct=True,
        )
        candidate_profile = scoring.score_and_verify(
            copy.deepcopy(candidate), task, str(with_path / "candidate"),
            label=f"cand{task}", require_correct=True,
        )
        margin_ok, margin_min = scoring.model_margin_stable(copy.deepcopy(candidate), task)
        lineage = next(
            item
            for item in archive_inventory["retained"][str(task)]
            if item["sha256"] == sha256(candidate_path)
        )
        row = {
            "task": task,
            "generator_hash": info["hash"],
            "lineage": lineage,
            "baseline_structure": structure(baseline_path),
            "candidate_structure": structure(candidate_path),
            "baseline_actual_profile": baseline_profile,
            "candidate_actual_profile": candidate_profile,
            "candidate_margin": {"stable": bool(margin_ok), "minimum": margin_min},
            "baseline_runtime_shapes": runtime_shapes(task, baseline_path),
            "candidate_runtime_shapes": runtime_shapes(task, candidate_path),
            "known_disable_all": known(task, candidate_path, True),
            "known_default": known(task, candidate_path, False),
            "graph_diff": graph_diff(baseline, candidate),
        }
        report["tasks"][str(task)] = row
        (HERE / "structural_audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            task,
            baseline_profile["cost"],
            candidate_profile["cost"],
            row["candidate_structure"]["giant_einsum_nodes"],
            row["candidate_structure"]["conv_bias_findings"],
            flush=True,
        )


if __name__ == "__main__":
    main()
