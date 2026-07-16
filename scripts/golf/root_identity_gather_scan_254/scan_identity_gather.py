#!/usr/bin/env python3
"""Scan all 400 authority members for formally identity Gather-family nodes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
RESULT = HERE / "scan_result.json"
CANDIDATES = HERE / "candidates"
FRESH_COUNT = 5000
FRESH_SEEDS = (254_000_001, 254_000_002)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "identity_gather_screen_helpers",
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "identity_gather_trace_helpers",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    result: dict[str, tuple[int, ...]] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                break
            dims.append(int(dim.dim_value))
        else:
            result[value.name] = tuple(dims)
    for item in inferred.graph.initializer:
        if item.dims and all(dim > 0 for dim in item.dims):
            result[item.name] = tuple(int(dim) for dim in item.dims)
    return result


def attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {attr.name: int(attr.i) for attr in node.attribute if attr.type == attr.INT}


def normalize_indices(values: np.ndarray, dimension: int) -> np.ndarray | None:
    normalized = values.astype(np.int64, copy=True)
    normalized[normalized < 0] += dimension
    if np.any(normalized < 0) or np.any(normalized >= dimension):
        return None
    return normalized


def standard_gather_identity(
    data_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    indices: np.ndarray,
    axis: int,
) -> tuple[bool, str]:
    rank = len(data_shape)
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        return False, "axis_out_of_range"
    if output_shape != data_shape:
        return False, "output_shape_differs"
    if indices.ndim != 1 or indices.shape[0] != data_shape[axis]:
        return False, "indices_not_complete_axis_vector"
    normalized = normalize_indices(indices, data_shape[axis])
    if normalized is None:
        return False, "indices_out_of_range"
    identity = np.arange(data_shape[axis], dtype=np.int64)
    if not np.array_equal(normalized, identity):
        return False, "normalized_indices_not_identity_order"
    return True, "exhaustive_axis_identity"


def gather_elements_identity(
    data_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    indices: np.ndarray,
    axis: int,
) -> tuple[bool, str]:
    rank = len(data_shape)
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        return False, "axis_out_of_range"
    if output_shape != data_shape or tuple(indices.shape) != data_shape:
        return False, "shape_not_full_identity_domain"
    normalized = normalize_indices(indices, data_shape[axis])
    if normalized is None:
        return False, "indices_out_of_range"
    coordinate = np.indices(data_shape, sparse=False)[axis]
    if not np.array_equal(normalized, coordinate):
        return False, "exhaustive_coordinate_map_not_identity"
    return True, "exhaustive_element_coordinate_identity"


def gather_nd_identity(
    data_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    indices: np.ndarray,
    batch_dims: int,
) -> tuple[bool, str]:
    if output_shape != data_shape or indices.ndim < 1:
        return False, "output_shape_differs_or_indices_scalar"
    rank = len(data_shape)
    q = indices.ndim
    k = int(indices.shape[-1])
    if batch_dims < 0 or batch_dims >= q or batch_dims + k > rank:
        return False, "invalid_batch_or_tuple_rank"
    expected = tuple(indices.shape[:-1]) + tuple(data_shape[batch_dims + k :])
    if expected != output_shape:
        return False, "gathernd_shape_formula_differs"

    # Exhaust the complete static output coordinate domain.  For every output
    # element, compute the source coordinate prescribed by GatherND and require
    # exact equality to the output coordinate.  This is a formal finite proof,
    # not randomized tensor testing.
    prefix_rank = q - 1
    for out_coord in np.ndindex(output_shape):
        prefix = out_coord[:prefix_rank]
        vector = np.asarray(indices[prefix], dtype=np.int64).copy()
        for component in range(k):
            dim = data_shape[batch_dims + component]
            if vector[component] < 0:
                vector[component] += dim
            if vector[component] < 0 or vector[component] >= dim:
                return False, "indices_out_of_range"
        data_coord = (
            tuple(prefix[:batch_dims])
            + tuple(int(value) for value in vector)
            + tuple(out_coord[prefix_rank:])
        )
        if data_coord != out_coord:
            return False, "exhaustive_coordinate_map_not_identity"
    return True, "exhaustive_gathernd_coordinate_identity"


def prove_node(
    node: onnx.NodeProto,
    shapes: dict[str, tuple[int, ...]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "op": node.op_type,
        "output": node.output[0] if node.output else "",
        "data": node.input[0] if node.input else "",
        "indices": node.input[1] if len(node.input) > 1 else "",
        "identity": False,
    }
    if len(node.input) < 2 or not node.output:
        row["reason"] = "missing_required_input_or_output"
        return row
    data_shape = shapes.get(node.input[0])
    output_shape = shapes.get(node.output[0])
    indices = arrays.get(node.input[1])
    if data_shape is None or output_shape is None:
        row["reason"] = "nonstatic_data_or_output_shape"
        return row
    if indices is None:
        row["reason"] = "indices_not_initializer"
        return row
    if indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        row["reason"] = "indices_not_int32_or_int64"
        return row
    attrs = attributes(node)
    if node.op_type == "Gather":
        ok, reason = standard_gather_identity(
            data_shape, output_shape, indices, attrs.get("axis", 0)
        )
    elif node.op_type == "GatherElements":
        ok, reason = gather_elements_identity(
            data_shape, output_shape, indices, attrs.get("axis", 0)
        )
    elif node.op_type == "GatherND":
        ok, reason = gather_nd_identity(
            data_shape, output_shape, indices, attrs.get("batch_dims", 0)
        )
    else:
        ok, reason = False, "not_gather_family"
    row.update(
        {
            "identity": ok,
            "reason": reason,
            "data_shape": list(data_shape),
            "output_shape": list(output_shape),
            "indices_shape": list(indices.shape),
            "indices_params": int(indices.size),
            "attributes": attrs,
        }
    )
    return row


def discover(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    shapes = tensor_shapes(model)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                uses[name].append(index)
    rows = []
    proven_indices: dict[int, dict[str, Any]] = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Gather", "GatherElements", "GatherND"}:
            continue
        row = prove_node(node, shapes, arrays)
        row["node_index"] = index
        rows.append(row)
        if row["identity"]:
            proven_indices[index] = row

    removable_initializers = []
    replace_indices = []
    for name, user_indices in uses.items():
        if name not in arrays:
            continue
        if user_indices and all(
            index in proven_indices
            and len(model.graph.node[index].input) > 1
            and model.graph.node[index].input[1] == name
            for index in user_indices
        ):
            removable_initializers.append(name)
            replace_indices.extend(user_indices)
    replace_indices = sorted(set(replace_indices))
    return {
        "task": task,
        "authority_sha256": sha256(data),
        "gather_family_count": len(rows),
        "proofs": rows,
        "static_identity_count": len(proven_indices),
        "removable_initializers": sorted(removable_initializers),
        "replace_node_indices": replace_indices,
        "removable_params": sum(int(arrays[name].size) for name in removable_initializers),
        "candidate_possible": bool(replace_indices and removable_initializers),
    }


def build_candidate(data: bytes, discovery: dict[str, Any]) -> bytes:
    model = onnx.load_model_from_string(data)
    replace = set(int(index) for index in discovery["replace_node_indices"])
    remove_initializers = set(discovery["removable_initializers"])
    for index, node in enumerate(model.graph.node):
        if index not in replace:
            continue
        old_op = node.op_type
        data_name = node.input[0]
        del node.input[:]
        node.input.append(data_name)
        node.op_type = "Identity"
        node.domain = ""
        del node.attribute[:]
        node.doc_string = f"exact replacement of identity {old_op}"
    kept = [item for item in model.graph.initializer if item.name not in remove_initializers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model.SerializeToString()


def score(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"identity_gather_{task:03d}_{label}_") as wd:
        return scoring.score_and_verify(
            model, task, wd, label=label, require_correct=False
        )


def strict_audit(data: bytes) -> dict[str, Any]:
    return SCREEN.structural_audit(data)


def runtime_trace(task: int, data: bytes) -> dict[str, Any]:
    return TRACE.runtime_shape_trace(task, onnx.load_model_from_string(data))


def make_session(data: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    return SCREEN.make_session(data, disabled, threads)


def known_raw_four(task: int, authority: bytes, candidate: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    result: dict[str, Any] = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            stats = {
                "total": 0,
                "candidate_right": 0,
                "raw_equal": 0,
                "threshold_equal": 0,
                "authority_errors": 0,
                "candidate_errors": 0,
            }
            try:
                base_session = make_session(authority, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["authority_session_error"] = f"{type(exc).__name__}: {exc}"
                result[key] = stats
                continue
            try:
                cand_session = make_session(candidate, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["candidate_session_error"] = f"{type(exc).__name__}: {exc}"
                result[key] = stats
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    stats["total"] += 1
                    try:
                        base = base_session.run(
                            None,
                            {base_session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["authority_errors"] += 1
                        continue
                    try:
                        cand = cand_session.run(
                            None,
                            {cand_session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["candidate_errors"] += 1
                        continue
                    want = benchmark["output"] > 0
                    stats["candidate_right"] += int(np.array_equal(cand > 0, want))
                    stats["raw_equal"] += int(np.array_equal(cand, base))
                    stats["threshold_equal"] += int(
                        np.array_equal(cand > 0, base > 0)
                    )
            result[key] = stats
    return result


def known_gate(report: dict[str, Any]) -> bool:
    return len(report) == 4 and all(
        row.get("total", 0) > 0
        and row.get("candidate_right") == row.get("total")
        and row.get("raw_equal") == row.get("total")
        and row.get("threshold_equal") == row.get("total")
        and row.get("authority_errors") == 0
        and row.get("candidate_errors") == 0
        and not row.get("authority_session_error")
        and not row.get("candidate_session_error")
        for row in report.values()
    )


def fresh_four(task: int, candidate: bytes) -> dict[str, Any]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    configs = (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    )
    sessions = {
        name: make_session(candidate, disabled, threads)
        for disabled, threads, name in configs
    }
    runs = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {
            name: {"right": 0, "wrong": 0, "errors": 0}
            for _, _, name in configs
        }
        valid = attempts = generation_errors = conversion_skips = 0
        while valid < FRESH_COUNT:
            attempts += 1
            try:
                benchmark = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if benchmark is None:
                conversion_skips += 1
                continue
            valid += 1
            want = benchmark["output"] > 0
            for _, _, name in configs:
                try:
                    session = sessions[name]
                    raw = session.run(
                        None, {session.get_inputs()[0].name: benchmark["input"]}
                    )[0]
                    if np.array_equal(raw > 0, want):
                        stats[name]["right"] += 1
                    else:
                        stats[name]["wrong"] += 1
                except Exception:  # noqa: BLE001
                    stats[name]["errors"] += 1
        runs.append(
            {
                "seed": seed,
                "valid": valid,
                "attempts": attempts,
                "generation_errors": generation_errors,
                "conversion_skips": conversion_skips,
                "configs": stats,
            }
        )
    return {"count_per_seed": FRESH_COUNT, "seeds": list(FRESH_SEEDS), "runs": runs}


def main() -> None:
    ort.set_default_logger_severity(4)
    authority_archive = AUTHORITY.read_bytes()
    if sha256(authority_archive) != AUTHORITY_SHA256:
        raise RuntimeError("authority archive changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)

    discoveries = []
    payloads: dict[int, bytes] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            payloads[task] = data
            row = discover(task, data)
            if row["gather_family_count"] or row["candidate_possible"]:
                discoveries.append(row)

    candidate_rows = []
    for discovery in discoveries:
        if not discovery["candidate_possible"]:
            continue
        task = int(discovery["task"])
        authority = payloads[task]
        candidate = build_candidate(authority, discovery)
        candidate_path = CANDIDATES / f"task{task:03d}_identity_gather.onnx"
        onnx.save_model(onnx.load_model_from_string(candidate), candidate_path)
        candidate = candidate_path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "path": str(candidate_path.relative_to(ROOT)),
            "authority_sha256": sha256(authority),
            "candidate_sha256": sha256(candidate),
            "discovery": discovery,
            "authority_score": score(task, authority, "authority"),
            "candidate_score": score(task, candidate, "candidate"),
            "strict": strict_audit(candidate),
        }
        try:
            trace = runtime_trace(task, candidate)
            row["runtime_shape"] = trace
            row["truthful"] = not trace["declared_actual_mismatches"]
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_error"] = f"{type(exc).__name__}: {exc}"
            row["truthful"] = False
        authority_score = row["authority_score"]
        candidate_score = row["candidate_score"]
        row["strictly_lower"] = bool(
            authority_score
            and candidate_score
            and candidate_score["cost"] < authority_score["cost"]
        )
        if not row["strict"]["pass"]:
            row["decision"] = "REJECT_STRUCTURE_SCHEMA_UB"
        elif not candidate_score or not candidate_score.get("correct"):
            row["decision"] = "REJECT_OFFICIAL_KNOWN_OR_SCORE"
        elif not row["truthful"]:
            row["decision"] = "REJECT_RUNTIME_SHAPE"
        elif not row["strictly_lower"]:
            row["decision"] = "REJECT_NOT_STRICTLY_LOWER"
        else:
            known = known_raw_four(task, authority, candidate)
            row["known_raw_four"] = known
            row["known_raw_four_pass"] = known_gate(known)
            if not row["known_raw_four_pass"]:
                row["decision"] = "REJECT_KNOWN_RAW_OR_RUNTIME"
            else:
                fresh = fresh_four(task, candidate)
                row["fresh"] = fresh
                fresh_pass = all(
                    config["right"] / FRESH_COUNT >= 0.90
                    and config["errors"] == 0
                    for run in fresh["runs"]
                    for config in run["configs"].values()
                )
                row["fresh_pass"] = fresh_pass
                row["decision"] = "ACCEPT" if fresh_pass else "REJECT_FRESH_POLICY90"
        candidate_rows.append(row)
        print(
            f"task{task:03d} static_identity={discovery['static_identity_count']} "
            f"params=-{discovery['removable_params']} "
            f"cost={row['authority_score']}->{row['candidate_score']} "
            f"decision={row['decision']}",
            flush=True,
        )

    accepted = [row for row in candidate_rows if row["decision"] == "ACCEPT"]
    report = {
        "authority_archive": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "tasks_with_gather_family": len(discoveries),
        "gather_family_nodes": sum(row["gather_family_count"] for row in discoveries),
        "static_identity_nodes": sum(
            row["static_identity_count"] for row in discoveries
        ),
        "truthful_identity_survivors": sum(
            bool(row["strict"]["pass"])
            and bool(row.get("candidate_score") and row["candidate_score"].get("correct"))
            and bool(row.get("truthful"))
            for row in candidate_rows
        ),
        "candidate_tasks": len(candidate_rows),
        "discoveries": discoveries,
        "candidates": candidate_rows,
        "accepted": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["candidate_sha256"],
                "cost": row["candidate_score"]["cost"],
            }
            for row in accepted
        ],
        "decision": "ACCEPT" if accepted else "NO_SAFE_IDENTITY_GATHER_WINNER",
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks_with_gather_family": report["tasks_with_gather_family"],
                "gather_family_nodes": report["gather_family_nodes"],
                "static_identity_nodes": report["static_identity_nodes"],
                "truthful_identity_survivors": report["truthful_identity_survivors"],
                "candidate_tasks": report["candidate_tasks"],
                "accepted": len(accepted),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
