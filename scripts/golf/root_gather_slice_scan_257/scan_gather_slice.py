#!/usr/bin/env python3
"""All-400 exact Gather(const arithmetic indices) -> Slice scan."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
RESULT = HERE / "scan_result.json"
CANDIDATES = HERE / "candidates"
FRESH_COUNT = 1000
FRESH_SEEDS = (257_000_001, 257_000_002)

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
    "gather_slice_screen_helpers",
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "gather_slice_trace_helpers",
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


def node_axis(node: onnx.NodeProto) -> int:
    return next((int(attr.i) for attr in node.attribute if attr.name == "axis"), 0)


def slice_end(axis_length: int, last: int, step: int) -> int:
    if step > 0:
        return last + 1
    if last > 0:
        return last - 1
    # ONNX normalizes a negative end by adding the dimension, then clamps to
    # -1 for a negative step.  -(N+1) therefore denotes the exclusive boundary
    # immediately before element zero without relying on INT64_MIN sentinels.
    return -axis_length - 1


def prove_gather(
    node: onnx.NodeProto,
    index: int,
    shapes: dict[str, tuple[int, ...]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "node_index": index,
        "output": node.output[0] if node.output else "",
        "data": node.input[0] if node.input else "",
        "indices": node.input[1] if len(node.input) > 1 else "",
        "convertible": False,
    }
    if node.op_type != "Gather" or len(node.input) < 2 or not node.output:
        row["reason"] = "not_standard_gather_or_missing_input"
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
    if indices.ndim != 1 or indices.size == 0:
        row["reason"] = "indices_not_nonempty_1d"
        return row
    rank = len(data_shape)
    axis = node_axis(node)
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        row["reason"] = "axis_out_of_range"
        return row
    dimension = data_shape[axis]
    normalized = indices.astype(np.int64, copy=True)
    normalized[normalized < 0] += dimension
    if np.any(normalized < 0) or np.any(normalized >= dimension):
        row["reason"] = "normalized_index_out_of_range"
        return row
    if normalized.size == 1:
        step = 1
    else:
        differences = np.diff(normalized)
        step = int(differences[0])
        if step == 0 or not np.all(differences == step):
            row["reason"] = "normalized_indices_not_unique_arithmetic_sequence"
            return row
    expected_output = data_shape[:axis] + (int(normalized.size),) + data_shape[axis + 1 :]
    if output_shape != expected_output:
        row["reason"] = "declared_output_shape_not_gather_formula"
        return row
    start = int(normalized[0])
    end = slice_end(dimension, int(normalized[-1]), step)
    reproduced = np.arange(dimension, dtype=np.int64)[slice(start, end, step)]
    if not np.array_equal(reproduced, normalized):
        row["reason"] = "slice_boundary_self_check_failed"
        return row
    row.update(
        {
            "convertible": True,
            "reason": "exhaustive_normalized_arithmetic_sequence",
            "data_shape": list(data_shape),
            "output_shape": list(output_shape),
            "indices_dtype": str(indices.dtype),
            "indices_shape": list(indices.shape),
            "indices_params": int(indices.size),
            "normalized_indices": normalized.tolist(),
            "slice": {"start": start, "end": end, "axis": axis, "step": step},
        }
    )
    return row


def singleton_i64_values(model: onnx.ModelProto, excluded: set[str]) -> dict[int, str]:
    values: dict[int, str] = {}
    for item in model.graph.initializer:
        if item.name in excluded or item.data_type != TensorProto.INT64 or list(item.dims) != [1]:
            continue
        array = np.asarray(numpy_helper.to_array(item), dtype=np.int64)
        values.setdefault(int(array[0]), item.name)
    return values


def choose_groups(
    model: onnx.ModelProto,
    groups: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Choose the exact group subset with maximal parameter saving."""

    if not groups:
        return {"selected": [], "param_saving": 0, "added_values": []}
    if len(groups) > 20:
        # This branch was not reached by the pinned all-400 authority.  Keep a
        # fail-closed deterministic fallback rather than exponential work.
        subsets = [tuple(range(len(groups)))]
    else:
        subsets = itertools.chain.from_iterable(
            itertools.combinations(range(len(groups)), count)
            for count in range(1, len(groups) + 1)
        )
    best = {"selected": [], "param_saving": 0, "added_values": []}
    for subset in subsets:
        removed = {groups[index]["initializer"] for index in subset}
        existing = singleton_i64_values(model, removed)
        needed = {
            int(value)
            for index in subset
            for proof in groups[index]["proofs"]
            for value in proof["slice"].values()
        }
        added = sorted(value for value in needed if value not in existing)
        saved = sum(int(arrays[name].size) for name in removed) - len(added)
        if saved > best["param_saving"] or (
            saved == best["param_saving"] and len(subset) > len(best["selected"])
        ):
            best = {
                "selected": list(subset),
                "param_saving": int(saved),
                "added_values": added,
            }
    return best


def discover(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    shapes = tensor_shapes(model)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                uses[name].append((node_index, input_index))
    proofs = []
    convertible_by_node = {}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Gather":
            continue
        proof = prove_gather(node, node_index, shapes, arrays)
        proofs.append(proof)
        if proof["convertible"]:
            convertible_by_node[node_index] = proof

    groups = []
    for name, array in arrays.items():
        user_slots = uses.get(name, [])
        if not user_slots:
            continue
        if all(
            input_index == 1
            and node_index in convertible_by_node
            and model.graph.node[node_index].input[1] == name
            for node_index, input_index in user_slots
        ):
            groups.append(
                {
                    "initializer": name,
                    "params": int(array.size),
                    "node_indices": [node_index for node_index, _ in user_slots],
                    "proofs": [convertible_by_node[node_index] for node_index, _ in user_slots],
                }
            )
    choice = choose_groups(model, groups, arrays)
    selected_groups = [groups[index] for index in choice["selected"]]
    return {
        "task": task,
        "authority_sha256": sha256(data),
        "gather_count": len(proofs),
        "convertible_node_count": sum(proof["convertible"] for proof in proofs),
        "proofs": proofs,
        "all_use_convertible_groups": groups,
        "selected_groups": selected_groups,
        "projected_param_saving": choice["param_saving"],
        "added_constant_values": choice["added_values"],
        "candidate_possible": bool(selected_groups and choice["param_saving"] > 0),
    }


def build_candidate(data: bytes, discovery: dict[str, Any]) -> bytes:
    model = onnx.load_model_from_string(data)
    selected_groups = discovery["selected_groups"]
    remove = {group["initializer"] for group in selected_groups}
    remaining = [item for item in model.graph.initializer if item.name not in remove]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining)
    constant_names = singleton_i64_values(model, set())
    for value in discovery["added_constant_values"]:
        name = f"gather_slice_i64_{'m' if value < 0 else 'p'}{abs(int(value))}"
        suffix = 0
        existing_names = {item.name for item in model.graph.initializer}
        while name in existing_names:
            suffix += 1
            name = f"gather_slice_i64_{'m' if value < 0 else 'p'}{abs(int(value))}_{suffix}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name)
        )
        constant_names[int(value)] = name

    proof_by_node = {
        int(proof["node_index"]): proof
        for group in selected_groups
        for proof in group["proofs"]
    }
    for node_index, proof in proof_by_node.items():
        node = model.graph.node[node_index]
        spec = proof["slice"]
        data_name = node.input[0]
        del node.input[:]
        node.input.extend(
            [
                data_name,
                constant_names[int(spec["start"])],
                constant_names[int(spec["end"])],
                constant_names[int(spec["axis"])],
                constant_names[int(spec["step"])],
            ]
        )
        node.op_type = "Slice"
        node.domain = ""
        del node.attribute[:]
        node.doc_string = "exact arithmetic-index Gather replacement"
    return model.SerializeToString()


def score(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"gather_slice_{task:03d}_{label}_") as wd:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, wd, label=label, require_correct=False
        )


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
                base_session = SCREEN.make_session(authority, disabled, threads)
                cand_session = SCREEN.make_session(candidate, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["session_error"] = f"{type(exc).__name__}: {exc}"
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
                            None, {base_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["authority_errors"] += 1
                        continue
                    try:
                        cand = cand_session.run(
                            None, {cand_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["candidate_errors"] += 1
                        continue
                    stats["candidate_right"] += int(
                        np.array_equal(cand > 0, benchmark["output"] > 0)
                    )
                    stats["raw_equal"] += int(np.array_equal(cand, base))
                    stats["threshold_equal"] += int(np.array_equal(cand > 0, base > 0))
            result[key] = stats
    return result


def known_pass(report: dict[str, Any]) -> bool:
    return len(report) == 4 and all(
        row.get("total", 0) > 0
        and row.get("candidate_right") == row.get("total")
        and row.get("raw_equal") == row.get("total")
        and row.get("threshold_equal") == row.get("total")
        and row.get("authority_errors") == 0
        and row.get("candidate_errors") == 0
        and not row.get("session_error")
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
        name: SCREEN.make_session(candidate, disabled, threads)
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
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority archive changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    discoveries = []
    payloads = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            payloads[task] = data
            row = discover(task, data)
            if row["gather_count"]:
                discoveries.append(row)

    candidate_rows = []
    for discovery in discoveries:
        if not discovery["candidate_possible"]:
            continue
        task = int(discovery["task"])
        authority = payloads[task]
        candidate = build_candidate(authority, discovery)
        path = CANDIDATES / f"task{task:03d}_gather_slice.onnx"
        onnx.save_model(onnx.load_model_from_string(candidate), path)
        candidate = path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "authority_sha256": sha256(authority),
            "candidate_sha256": sha256(candidate),
            "discovery": discovery,
            "authority_score": score(task, authority, "authority"),
            "candidate_score": score(task, candidate, "candidate"),
            "strict": SCREEN.structural_audit(candidate),
        }
        try:
            trace = TRACE.runtime_shape_trace(task, onnx.load_model_from_string(candidate))
            row["runtime_shape"] = trace
            row["truthful"] = not trace["declared_actual_mismatches"]
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_error"] = f"{type(exc).__name__}: {exc}"
            row["truthful"] = False
        base_score = row["authority_score"]
        cand_score = row["candidate_score"]
        row["strictly_lower"] = bool(
            base_score and cand_score and cand_score["cost"] < base_score["cost"]
        )
        if not row["strictly_lower"]:
            row["decision"] = "REJECT_NOT_STRICTLY_LOWER_OR_UNSCORABLE"
        elif not row["strict"]["pass"]:
            row["decision"] = "REJECT_STRUCTURE_SCHEMA_UB"
        elif not cand_score.get("correct"):
            row["decision"] = "REJECT_OFFICIAL_KNOWN"
        else:
            known = known_raw_four(task, authority, candidate)
            row["known_raw_four"] = known
            row["known_raw_four_pass"] = known_pass(known)
            if not row["truthful"]:
                row["decision"] = "REJECT_RUNTIME_SHAPE"
            elif not row["known_raw_four_pass"]:
                row["decision"] = "REJECT_KNOWN_RAW_OR_RUNTIME"
            else:
                fresh = fresh_four(task, candidate)
                row["fresh"] = fresh
                row["fresh_pass"] = all(
                    config["right"] == FRESH_COUNT
                    and config["wrong"] == 0
                    and config["errors"] == 0
                    for run in fresh["runs"]
                    for config in run["configs"].values()
                )
                row["decision"] = "ACCEPT" if row["fresh_pass"] else "REJECT_FRESH"
        candidate_rows.append(row)
        print(
            f"task{task:03d} projected_params=-{discovery['projected_param_saving']} "
            f"cost={base_score}->{cand_score} decision={row['decision']}",
            flush=True,
        )

    accepted = [row for row in candidate_rows if row["decision"] == "ACCEPT"]
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "tasks_with_gather": len(discoveries),
        "gather_nodes": sum(row["gather_count"] for row in discoveries),
        "convertible_nodes": sum(row["convertible_node_count"] for row in discoveries),
        "candidate_tasks": len(candidate_rows),
        "discoveries": discoveries,
        "candidates": candidate_rows,
        "accepted": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["candidate_sha256"],
                "authority_cost": row["authority_score"]["cost"],
                "candidate_cost": row["candidate_score"]["cost"],
            }
            for row in accepted
        ],
        "decision": "ACCEPT" if accepted else "NO_SAFE_GATHER_SLICE_WINNER",
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks_with_gather": report["tasks_with_gather"],
                "gather_nodes": report["gather_nodes"],
                "convertible_nodes": report["convertible_nodes"],
                "candidate_tasks": report["candidate_tasks"],
                "accepted": len(accepted),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
