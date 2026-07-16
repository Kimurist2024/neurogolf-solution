#!/usr/bin/env python3
"""Lane 130: exact mem/param shaving for task182, task365, and task374."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (182, 365, 374)
HASHES = {182: "776ffc46", 365: "e50d258f", 374: "ea32f347"}
MEMBER_SHA256 = {
    182: "625b31492d9135295229c67ca0322000a2ff351e81e627bb882b89dde6bfda97",
    365: "85d63fa65d51d5aa065c5966725d092813c21b0e7a92453bf745d096371e3214",
    374: "93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a",
}
BASE_COSTS = {182: 949, 365: 1355, 374: 481}
KINDS = (
    "cleanup",
    "dedupe",
    "noops",
    "cse",
    "optional",
    "fold",
    "absorb",
    "combined",
    "normalize",
    "normalized_combined",
)
REFERENCE_SEEDS = (130_000_000, 130_100_000)
REFERENCE_FRESH_PER_SEED = 1500
CANDIDATE_FRESH_PER_SEED = 1500

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


BASE = load_module(
    "lane130_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_high077_361_129/run_lane.py",
)
EXACT = BASE.EXACT
AUDIT = BASE.AUDIT
RANK = BASE.RANK


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def normalize(grid: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid]


def components(
    grid: list[list[int]],
    predicate: Callable[[int], bool],
) -> list[set[tuple[int, int]]]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    result: list[set[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or not predicate(grid[row][col]):
                continue
            seen.add((row, col))
            queue = [(row, col)]
            component = {(row, col)}
            for rr, cc in queue:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    point = rr + dr, cc + dc
                    if not (0 <= point[0] < height and 0 <= point[1] < width):
                        continue
                    if point in seen or not predicate(grid[point[0]][point[1]]):
                        continue
                    seen.add(point)
                    component.add(point)
                    queue.append(point)
            result.append(component)
    return result


def solve_182(grid: list[list[int]]) -> list[list[int]]:
    """Recolor every external copy matching the framed source sprite."""
    height, width = len(grid), len(grid[0])
    frames = []
    for top in range(height - 6):
        for left in range(width - 6):
            border = (
                [(top, left + offset) for offset in range(7)]
                + [(top + 6, left + offset) for offset in range(7)]
                + [(top + offset, left) for offset in range(1, 6)]
                + [(top + offset, left + 6) for offset in range(1, 6)]
            )
            if all(grid[row][col] == 5 for row, col in border):
                frames.append((top, left))
    if len(frames) != 1:
        raise ValueError(f"expected one complete frame, got {len(frames)}")
    frame_row, frame_col = frames[0]
    sprites = components(grid, lambda value: value not in (0, 5))
    source = next(
        component
        for component in sprites
        if all(
            frame_row < row < frame_row + 6 and frame_col < col < frame_col + 6
            for row, col in component
        )
    )

    def signature(component: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        min_row = min(row for row, _ in component)
        min_col = min(col for _, col in component)
        return frozenset((row - min_row, col - min_col) for row, col in component)

    source_signature = signature(source)
    source_row, source_col = next(iter(source))
    source_color = grid[source_row][source_col]
    output = copy.deepcopy(grid)
    for component in sprites:
        if signature(component) != source_signature:
            continue
        for row, col in component:
            output[row][col] = source_color
    return output


def solve_365(grid: list[list[int]]) -> list[list[int]]:
    """Extract the filled rectangle with the greatest number of red cells."""
    objects = components(grid, lambda value: value != 0)
    target = max(
        objects,
        key=lambda component: sum(grid[row][col] == 2 for row, col in component),
    )
    rows = [row for row, _ in target]
    cols = [col for _, col in target]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def solve_374(grid: list[list[int]]) -> list[list[int]]:
    """Color shortest/middle/longest gray lines as 2/4/1."""
    lines = sorted(components(grid, lambda value: value == 5), key=len)
    if len(lines) != 3 or not (len(lines[0]) < len(lines[1]) < len(lines[2])):
        raise ValueError("task374 requires three distinct line lengths")
    output = copy.deepcopy(grid)
    for line, color in zip(lines, (2, 4, 1)):
        for row, col in line:
            output[row][col] = color
    return output


SOLVERS = {182: solve_182, 365: solve_365, 374: solve_374}


def verify_references() -> dict[str, Any]:
    result: dict[str, Any] = {
        "seeds": list(REFERENCE_SEEDS),
        "fresh_per_seed": REFERENCE_FRESH_PER_SEED,
        "tasks": {},
    }
    for task in TASKS:
        solver = SOLVERS[task]
        examples = scoring.load_examples(task)
        known = examples["train"] + examples["test"] + examples["arc-gen"]
        known_right = sum(
            solver(normalize(example["input"])) == normalize(example["output"])
            for example in known
        )
        generator = importlib.import_module(f"task_{HASHES[task]}")
        fresh = []
        for seed_base in REFERENCE_SEEDS:
            seed = seed_base + task
            random.seed(seed)
            right = errors = 0
            first_failure = None
            for index in range(REFERENCE_FRESH_PER_SEED):
                try:
                    example = generator.generate()
                    ok = (
                        solver(normalize(example["input"]))
                        == normalize(example["output"])
                    )
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    ok = False
                    if first_failure is None:
                        first_failure = {
                            "index": index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                right += int(ok)
                if not ok and first_failure is None:
                    first_failure = {"index": index}
            fresh.append(
                {
                    "seed": seed,
                    "right": right,
                    "total": REFERENCE_FRESH_PER_SEED,
                    "errors": errors,
                    "first_failure": first_failure,
                }
            )
        passed = known_right == len(known) and all(
            row["right"] == REFERENCE_FRESH_PER_SEED and row["errors"] == 0
            for row in fresh
        )
        result["tasks"][str(task)] = {
            "hash": HASHES[task],
            "known": {"right": known_right, "total": len(known)},
            "fresh": fresh,
            "pass": passed,
        }
        print(f"REF task{task:03d} known={known_right}/{len(known)}", flush=True)
    return result


def initializer_duplicates(model: onnx.ModelProto) -> list[list[str]]:
    groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = {}
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        key = (initializer.data_type, tuple(initializer.dims), array.tobytes())
        groups.setdefault(key, []).append(initializer.name)
    return [names for names in groups.values() if len(names) > 1]


def selu_type_audit(model: onnx.ModelProto) -> dict[str, Any]:
    opset = max(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        default=18,
    )
    schema = onnx.defs.get_schema("Selu", max_inclusive_version=opset)
    allowed = sorted(
        {
            type_name
            for constraint in schema.type_constraints
            for type_name in constraint.allowed_type_strs
        }
    )
    mul = next(node for node in model.graph.node if node.output == ["cnt6"])
    input_name = mul.input[0]
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    tensors = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    elem_type = tensors[input_name].type.tensor_type.elem_type
    actual = f"tensor({onnx.TensorProto.DataType.Name(elem_type).lower()})"
    return {
        "opset": opset,
        "target_node": "cnt6",
        "target_input": input_name,
        "target_input_type": actual,
        "selu_allowed_types": allowed,
        "u8_admitted": actual in allowed,
        "decision": "reject_type_constraint" if actual not in allowed else "eligible",
    }


def fold_fixed_shape(
    model: onnx.ModelProto,
    node_output: str,
    expected_input_shape: tuple[int, ...],
    expected_shape_value: tuple[int, ...],
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    node = next(node for node in candidate.graph.node if node.output == [node_output])
    if node.op_type != "Shape":
        raise ValueError(f"{node_output} is not produced by Shape")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    tensor_map = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    input_value = tensor_map[node.input[0]]
    actual_input_shape = tuple(
        int(dim.dim_value) for dim in input_value.type.tensor_type.shape.dim
    )
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"fixed-shape proof drift for {node.input[0]}: "
            f"{actual_input_shape} != {expected_input_shape}"
        )
    start = next((attribute.i for attribute in node.attribute if attribute.name == "start"), 0)
    end = next(
        (attribute.i for attribute in node.attribute if attribute.name == "end"),
        len(actual_input_shape),
    )
    actual_value = actual_input_shape[start:end]
    if actual_value != expected_shape_value:
        raise ValueError(f"Shape value drift: {actual_value} != {expected_shape_value}")
    candidate.graph.node.remove(node)
    for value in list(candidate.graph.value_info):
        if value.name == node_output:
            candidate.graph.value_info.remove(value)
    candidate.graph.initializer.append(
        numpy_helper.from_array(np.asarray(actual_value, dtype=np.int64), node_output)
    )
    return candidate, {
        "fixed_shape_folds": [
            {
                "op": "Shape",
                "input": node.input[0],
                "output": node_output,
                "input_shape": list(actual_input_shape),
                "start": start,
                "end": end,
                "constant_value": list(actual_value),
                "proof": "model_contract_static_shape_all_input_values",
            }
        ],
        "semantic_action_count": 1,
        "metadata_action_count": 1,
    }


def castlike_i32_to_cast(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    node = next(node for node in candidate.graph.node if node.output == ["grid_idx"])
    if node.op_type != "CastLike" or list(node.input) != ["color_fake", "i32dummy"]:
        raise ValueError("task374 grid_idx CastLike contract drift")
    node.op_type = "Cast"
    del node.input[1:]
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", onnx.TensorProto.INT32)])
    consumers = [
        other
        for other in candidate.graph.node
        if "i32dummy" in other.input
    ]
    if consumers:
        raise ValueError("i32dummy still has consumers")
    for initializer in list(candidate.graph.initializer):
        if initializer.name == "i32dummy":
            candidate.graph.initializer.remove(initializer)
    return candidate, {
        "attributeized_casts": [
            {
                "output": "grid_idx",
                "from": "CastLike(color_fake,i32dummy)",
                "to": "Cast(color_fake,to=INT32)",
                "removed_initializer": "i32dummy",
                "proof": "CastLike_second_input_supplies_only_INT32_dtype",
            }
        ],
        "semantic_action_count": 1,
        "metadata_action_count": 0,
    }


def merge_actions(*actions: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    semantic = metadata = 0
    for action in actions:
        semantic += int(action.get("semantic_action_count", 0))
        metadata += int(action.get("metadata_action_count", 0))
        for key, value in action.items():
            if key in ("semantic_action_count", "metadata_action_count"):
                continue
            if isinstance(value, list):
                result.setdefault(key, []).extend(value)
            else:
                result[key] = value
    result["semantic_action_count"] = semantic
    result["metadata_action_count"] = metadata
    return result


def exact_proofs(actions: dict[str, Any]) -> list[str]:
    return sorted(
        {
            item["proof"]
            for value in actions.values()
            if isinstance(value, list)
            for item in value
            if isinstance(item, dict) and "proof" in item
        }
    )


def make_sessions(data: bytes) -> tuple[dict[str, ort.InferenceSession], dict[str, str]]:
    sessions: dict[str, ort.InferenceSession] = {}
    errors: dict[str, str] = {}
    for mode, disable_all in (("disable_all", True), ("default", False)):
        try:
            sessions[mode] = BASE.make_session(data, disable_all)
        except Exception as exc:  # noqa: BLE001
            errors[mode] = f"{type(exc).__name__}: {exc}"
    return sessions, errors


def candidate_equivalence_and_fresh(
    task: int,
    base_data: bytes,
    candidate_data: bytes,
) -> dict[str, Any]:
    base_sessions, base_errors = make_sessions(base_data)
    cand_sessions, cand_errors = make_sessions(candidate_data)
    result: dict[str, Any] = {
        "fresh_per_seed_per_mode": CANDIDATE_FRESH_PER_SEED,
        "base_session_errors": base_errors,
        "candidate_session_errors": cand_errors,
        "known": {},
        "fresh": [],
        "pass": False,
    }
    if base_errors or cand_errors:
        result["not_run_reason"] = "both models must load in both ORT modes"
        return result

    examples = scoring.load_examples(task)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    for mode in ("disable_all", "default"):
        stats = {"total": len(known), "bitwise_equal": 0, "candidate_correct": 0, "errors": 0}
        for example in known:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                base_raw = np.asarray(
                    base_sessions[mode].run(
                        [base_sessions[mode].get_outputs()[0].name],
                        {base_sessions[mode].get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
                candidate_raw = np.asarray(
                    cand_sessions[mode].run(
                        [cand_sessions[mode].get_outputs()[0].name],
                        {cand_sessions[mode].get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
                stats["bitwise_equal"] += int(
                    base_raw.dtype == candidate_raw.dtype
                    and base_raw.shape == candidate_raw.shape
                    and base_raw.tobytes() == candidate_raw.tobytes()
                )
                stats["candidate_correct"] += int(
                    np.array_equal(candidate_raw > 0, benchmark["output"] > 0)
                )
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
        result["known"][mode] = stats

    generator = importlib.import_module(f"task_{HASHES[task]}")
    for seed in (130_200_000 + task, 130_300_000 + task):
        random.seed(seed)
        modes = {
            mode: {
                "bitwise_equal": 0,
                "candidate_correct": 0,
                "wrong": 0,
                "errors": 0,
                "nonfinite": 0,
                "near_positive": 0,
                "min_positive": None,
            }
            for mode in ("disable_all", "default")
        }
        valid = 0
        while valid < CANDIDATE_FRESH_PER_SEED:
            benchmark = scoring.convert_to_numpy(generator.generate())
            if benchmark is None:
                continue
            valid += 1
            for mode in ("disable_all", "default"):
                stats = modes[mode]
                try:
                    base_raw = np.asarray(
                        base_sessions[mode].run(
                            [base_sessions[mode].get_outputs()[0].name],
                            {base_sessions[mode].get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    candidate_raw = np.asarray(
                        cand_sessions[mode].run(
                            [cand_sessions[mode].get_outputs()[0].name],
                            {cand_sessions[mode].get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    stats["bitwise_equal"] += int(
                        base_raw.dtype == candidate_raw.dtype
                        and base_raw.shape == candidate_raw.shape
                        and base_raw.tobytes() == candidate_raw.tobytes()
                    )
                    correct = np.array_equal(candidate_raw > 0, benchmark["output"] > 0)
                    stats["candidate_correct"] += int(correct)
                    stats["wrong"] += int(not correct)
                    stats["nonfinite"] += int(not np.isfinite(candidate_raw).all())
                    positives = candidate_raw[candidate_raw > 0]
                    if positives.size:
                        minimum = float(positives.min())
                        old = stats["min_positive"]
                        stats["min_positive"] = minimum if old is None else min(old, minimum)
                        stats["near_positive"] += int(
                            np.count_nonzero((candidate_raw > 0) & (candidate_raw < 0.25))
                        )
                except Exception:  # noqa: BLE001
                    stats["errors"] += 1
        result["fresh"].append({"seed": seed, "valid": valid, "modes": modes})

    known_pass = all(
        stats["bitwise_equal"] == stats["total"]
        and stats["candidate_correct"] == stats["total"]
        and stats["errors"] == 0
        for stats in result["known"].values()
    )
    fresh_pass = all(
        stats["bitwise_equal"] == CANDIDATE_FRESH_PER_SEED
        and stats["candidate_correct"] == CANDIDATE_FRESH_PER_SEED
        and stats["wrong"] == 0
        and stats["errors"] == 0
        and stats["nonfinite"] == 0
        and stats["near_positive"] == 0
        for run in result["fresh"]
        for stats in run["modes"].values()
    )
    result["pass"] = known_pass and fresh_pass
    return result


def mismatch_summary(audit: dict[str, Any]) -> dict[str, Any]:
    trace = audit.get("runtime_shape_trace") or {}
    mismatches = trace.get("declared_actual_mismatches", [])
    return {
        "error": trace.get("error"),
        "count": len(mismatches),
        "single_example_intermediate_bytes": trace.get("single_example_intermediate_bytes"),
    }


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    result = BASE.structure(model)
    result["duplicate_initializer_groups"] = initializer_duplicates(model)
    return result


def cost(path: Path) -> dict[str, int]:
    memory, params, total = RANK.cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(total)}


def variants(task: int, base: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    result = []
    for kind in KINDS:
        candidate, actions = EXACT.transform(base, kind)
        result.append((kind, candidate, actions))
    if task == 365:
        candidate, actions = fold_fixed_shape(
            base, "__hc_tlflat_shape", (64,), (64,)
        )
        result.append(("fixed_shape_fold", candidate, actions))
    if task == 374:
        shape_model, shape_actions = fold_fixed_shape(
            base, "__sp_shape", (1, 10, 30, 30), (1,)
        )
        cast_model, cast_actions = castlike_i32_to_cast(base)
        combined, combined_cast = castlike_i32_to_cast(shape_model)
        result.extend(
            [
                ("fixed_shape_fold", shape_model, shape_actions),
                ("cast_i32_attribute", cast_model, cast_actions),
                (
                    "fixed_shape_fold_cast_i32",
                    combined,
                    merge_actions(shape_actions, combined_cast),
                ),
            ]
        )
    return result


def main() -> int:
    for name in ("baseline", "candidates", "candidate", "audit"):
        (HERE / name).mkdir(parents=True, exist_ok=True)

    archive_before = sha256(AUTHORITY.read_bytes())
    if archive_before != ARCHIVE_SHA256:
        raise RuntimeError(f"authority drift before lane: {archive_before}")
    with zipfile.ZipFile(AUTHORITY) as archive:
        payloads = {
            task: archive.read(f"task{task:03d}.onnx")
            for task in TASKS
        }
    for task, data in payloads.items():
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} member drift")
        (HERE / "baseline" / f"task{task:03d}.onnx").write_bytes(data)

    reference_path = HERE / "audit/reference_audit.json"
    if reference_path.exists():
        references = json.loads(reference_path.read_text())
    else:
        references = verify_references()
        reference_path.write_text(json.dumps(references, indent=2) + "\n")

    type_constraints = {
        "task182_u8_mul_to_selu": selu_type_audit(
            onnx.load(HERE / "baseline/task182.onnx")
        )
    }
    (HERE / "audit/type_constraints.json").write_text(
        json.dumps(type_constraints, indent=2) + "\n"
    )

    baselines: dict[str, Any] = {}
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        measured = cost(path)
        if measured["cost"] != BASE_COSTS[task]:
            raise RuntimeError(f"task{task:03d} cost drift: {measured}")
        audited = AUDIT.audit(f"baseline_task{task:03d}", task, path)
        baseline_dual_fresh = candidate_equivalence_and_fresh(
            task, path.read_bytes(), path.read_bytes()
        )
        baselines[str(task)] = {
            "task": task,
            "path": relative(path),
            "sha256": sha256(path.read_bytes()),
            "file_bytes": path.stat().st_size,
            "cost": measured,
            "structure": structure(onnx.load(path)),
            "full_audit": audited,
            "runtime_shape": mismatch_summary(audited),
            "dual_ort_known_fresh": baseline_dual_fresh,
        }
        print(
            f"BASE task{task:03d} cost={measured['cost']} "
            f"mismatches={baselines[str(task)]['runtime_shape']['count']}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for task in TASKS:
        base_path = HERE / "baseline" / f"task{task:03d}.onnx"
        base_model = onnx.load(base_path)
        base_data = base_path.read_bytes()
        baseline_mismatches = baselines[str(task)]["runtime_shape"]["count"]
        known_total = references["tasks"][str(task)]["known"]["total"]
        for kind, candidate, actions in variants(task, base_model):
            data = candidate.SerializeToString()
            digest = sha256(data)
            if digest == MEMBER_SHA256[task] or (task, digest) in seen:
                continue
            seen.add((task, digest))
            path = HERE / "candidates" / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
            path.write_bytes(data)
            proof = exact_proofs(actions)
            custom_exact = all(
                item in {
                    "model_contract_static_shape_all_input_values",
                    "CastLike_second_input_supplies_only_INT32_dtype",
                }
                for item in proof
            ) and bool(proof)
            generic_exact = (
                bool(proof)
                and actions.get("metadata_action_count", 0) == 0
                and all(
                    item
                    in {
                        "identity",
                        "all_inputs_constant_numpy_equivalent",
                        "initializer_byte_identity",
                        "schema_optional_unused_output",
                        "pure_node_same_op_attributes_inputs",
                        "neutral_add_zero",
                        "neutral_mul_one",
                        "same_dtype_cast",
                        "same_shape_reshape",
                        "factor_constants_numpy_equivalent",
                    }
                    for item in proof
                )
            )
            theorem_exact = custom_exact or generic_exact
            row: dict[str, Any] = {
                "task": task,
                "kind": kind,
                "path": relative(path),
                "sha256": digest,
                "authority_sha256": MEMBER_SHA256[task],
                "authority_cost": BASE_COSTS[task],
                "actions": actions,
                "all_input_exact_proofs": proof,
                "theorem_exact": theorem_exact,
                "structure": structure(candidate),
            }
            if (
                row["structure"].get("checker_full") is not True
                or row["structure"].get("strict_shape_data_prop") is not True
            ):
                row["stage"] = "REJECT_CHECKER_OR_STRICT_SHAPE"
                rows.append(row)
                continue
            try:
                measured = cost(path)
            except Exception as exc:  # noqa: BLE001
                row["cost_error"] = f"{type(exc).__name__}: {exc}"
                row["stage"] = "REJECT_UNSCORABLE"
                rows.append(row)
                continue
            row["cost"] = measured
            if measured["cost"] < 0:
                row["stage"] = "REJECT_UNSCORABLE"
                rows.append(row)
                continue
            if measured["cost"] >= BASE_COSTS[task]:
                row["stage"] = "REJECT_NOT_STRICTLY_LOWER"
                rows.append(row)
                continue
            if not theorem_exact:
                row["stage"] = "REJECT_NO_ALL_INPUT_EQUIVALENCE_THEOREM"
                rows.append(row)
                continue

            audited = AUDIT.audit(f"candidate_task{task:03d}_{kind}", task, path)
            row["full_audit"] = audited
            row["runtime_shape"] = mismatch_summary(audited)
            official = audited.get("official_like_score") or {}
            disable = audited.get("known_disable_all") or {}
            default = audited.get("known_default") or {}
            structural_ok = (
                audited.get("full_check") is True
                and audited.get("strict_shape_data_prop") is True
                and not audited.get("banned_ops")
                and not audited.get("nonstandard_domains")
                and not audited.get("function_count", 0)
                and not audited.get("sparse_initializer_count", 0)
            )
            known_ok = (
                BASE.total_perfect(disable, known_total)
                and BASE.total_perfect(default, known_total)
            )
            official_ok = (
                official.get("correct") is True
                and official.get("cost") == measured["cost"]
            )
            runtime_ok = (
                row["runtime_shape"]["error"] is None
                and row["runtime_shape"]["count"] <= baseline_mismatches
            )
            row["gate_summary"] = {
                "structural_ok": structural_ok,
                "known_both_ort": known_ok,
                "official_correct_cost": official_ok,
                "no_new_shape_cloak_or_runtime_error": runtime_ok,
            }
            if not all(row["gate_summary"].values()):
                row["stage"] = "REJECT_SOUND_GATES"
                rows.append(row)
                continue

            equivalence = candidate_equivalence_and_fresh(
                task, base_data, data
            )
            row["bitwise_and_fresh"] = equivalence
            if not equivalence["pass"]:
                row["stage"] = "REJECT_BITWISE_OR_FRESH"
                rows.append(row)
                continue

            gain = math.log(BASE_COSTS[task] / measured["cost"])
            destination = HERE / "candidate" / f"task{task:03d}.onnx"
            destination.write_bytes(data)
            row["stage"] = "ADMIT"
            row["projected_gain"] = gain
            row["admitted_path"] = relative(destination)
            winner = {
                "task": task,
                "old_cost": BASE_COSTS[task],
                "new_cost": measured["cost"],
                "gain": gain,
                "sha256": digest,
                "path": relative(destination),
                "all_input_exact_proofs": proof,
                "known_per_ort": known_total,
                "fresh_per_seed_per_ort": CANDIDATE_FRESH_PER_SEED,
                "fresh_seeds": [130_200_000 + task, 130_300_000 + task],
            }
            winners.append(winner)
            rows.append(row)
        print(
            f"SCAN task{task:03d} variants={sum(row['task'] == task for row in rows)}",
            flush=True,
        )

    stage_counts = dict(Counter(row["stage"] for row in rows))
    archive_after = sha256(AUTHORITY.read_bytes())
    if archive_after != archive_before:
        raise RuntimeError(f"authority changed during lane: {archive_after}")

    result = {
        "lane": 130,
        "authority": {
            "score": 8009.46,
            "archive": relative(AUTHORITY),
            "sha256_before": archive_before,
            "sha256_after": archive_after,
            "member_sha256": MEMBER_SHA256,
            "costs": BASE_COSTS,
        },
        "references": references,
        "type_constraints": type_constraints,
        "baselines": baselines,
        "rows": rows,
        "variant_count": len(rows),
        "stage_counts": stage_counts,
        "winner_count": len(winners),
        "winners": winners,
    }
    (HERE / "audit/results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "authority_archive_sha256": archive_after,
        "authority_member_sha256": MEMBER_SHA256,
        "authority_costs": BASE_COSTS,
        "winner_count": len(winners),
        "winners": winners,
        "root_files_modified": [],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "variant_count": len(rows),
                "stage_counts": stage_counts,
                "winner_count": len(winners),
                "winners": winners,
                "reference_pass": all(
                    value["pass"] for value in references["tasks"].values()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
