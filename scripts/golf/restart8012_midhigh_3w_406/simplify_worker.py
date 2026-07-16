#!/usr/bin/env python3
"""Apply exact/current-graph initializer, Einsum, Gather and lookup shaves."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
OUT = HERE / "simplify_candidates"
EVIDENCE = HERE / "simplify_evidence.json"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

import sys
sys.path.insert(0, str(ROOT))
from scripts.lib import scoring  # noqa: E402


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = import_path("restart8012_exact_shave", ROOT / "scripts/golf/loop_7999_13/exact_shave.py")
SCALAR = import_path("restart8012_scalar_shave", ROOT / "scripts/golf/loop_7999_13/scalar_constant_shave.py")
EINSUM_UNIT = import_path("restart8012_einsum_unit", ROOT / "scripts/golf/root_einsum_unit_factor_scan_256/scan.py")
GATHER_SLICE = import_path("restart8012_gather_slice", ROOT / "scripts/golf/root_gather_slice_scan_257/scan_gather_slice.py")


def structural(model: onnx.ModelProto) -> tuple[bool, str | None]:
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        if model.functions or model.graph.sparse_initializer:
            raise ValueError("functions or sparse initializers")
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"nonfinite initializer {init.name}")
        for node in model.graph.node:
            if node.op_type in BANNED or "Sequence" in node.op_type:
                raise ValueError(f"banned op {node.op_type}")
            for attr in node.attribute:
                if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                    raise ValueError("nested graph")
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def make_session(model: onnx.ModelProto, optimization: ort.GraphOptimizationLevel, threads: int) -> ort.InferenceSession | None:
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            return None
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None


def cases(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    return [
        converted
        for example in examples["train"] + examples["test"] + examples["arc-gen"]
        if (converted := scoring.convert_to_numpy(example)) is not None
    ]


def evaluate(runtime: ort.InferenceSession | None, items: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    right = wrong = errors = nonfinite = smallpositive = shape_errors = 0
    for item in items:
        try:
            if runtime is None:
                raise RuntimeError("no runtime")
            raw = runtime.run(["output"], {"input": item["input"]})[0]
            if raw.shape != item["output"].shape:
                shape_errors += 1
                wrong += 1
                continue
            if not np.isfinite(raw).all():
                nonfinite += 1
            if np.any((raw > 0.0) & (raw < 0.25)):
                smallpositive += 1
            if np.array_equal(raw > 0.0, item["output"] > 0.0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    total = len(items)
    return {
        "total": total, "right": right, "wrong": wrong, "errors": errors,
        "nonfinite": nonfinite, "smallpositive": smallpositive,
        "shape_errors": shape_errors, "accuracy": right / total if total else 0.0,
    }


def clean(row: dict[str, Any]) -> bool:
    return not any(row[key] for key in ("errors", "nonfinite", "smallpositive", "shape_errors"))


def scalar_variant(original: onnx.ModelProto) -> tuple[onnx.ModelProto | None, dict[str, Any]]:
    uses: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in original.graph.node:
        for position, name in enumerate(node.input):
            if name:
                uses[name].append((node, position))
    candidate = copy.deepcopy(original)
    changes = []
    for index, initializer in enumerate(original.graph.initializer):
        reduced = SCALAR.broadcast_reduction(numpy_helper.to_array(initializer))
        consumers = uses.get(initializer.name, [])
        if reduced is None or not consumers:
            continue
        if not all(SCALAR.safe_use(node, position) for node, position in consumers):
            continue
        before = list(initializer.dims)
        candidate.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(np.ascontiguousarray(reduced), initializer.name)
        )
        changes.append({
            "name": initializer.name, "before": before, "after": list(reduced.shape),
            "saved_params": int(math.prod(before or [1]) - reduced.size),
        })
    return (candidate if changes else None), {"changes": changes}


def einsum_unit_variants(original: onnx.ModelProto) -> list[tuple[onnx.ModelProto, dict[str, Any]]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in original.graph.initializer
    }
    occurrence = EINSUM_UNIT.uses(original)
    result = []
    for name, array in arrays.items():
        if not array.size or not np.isfinite(array).all() or not np.all(array == 1):
            continue
        positions = occurrence.get(name, [])
        if not positions:
            continue
        plans = {}
        for node_index in sorted({node_index for node_index, _ in positions}):
            plan = EINSUM_UNIT.rewrite_node_for_initializer(
                original.graph.node[node_index], name, array
            )
            if plan is None:
                break
            plans[node_index] = plan
        else:
            candidate = copy.deepcopy(original)
            for node_index, (new_inputs, new_equation) in plans.items():
                node = candidate.graph.node[node_index]
                del node.input[:]
                node.input.extend(new_inputs)
                EINSUM_UNIT.set_equation(node, new_equation)
            keep = [item for item in candidate.graph.initializer if item.name != name]
            del candidate.graph.initializer[:]
            candidate.graph.initializer.extend(keep)
            result.append((candidate, {
                "initializer": name, "shape": list(array.shape),
                "saved_params": int(array.size), "uses": positions,
            }))
    return result


def gather_slice_variant(task: int, data: bytes) -> tuple[onnx.ModelProto | None, dict[str, Any]]:
    try:
        discovery = GATHER_SLICE.discover(task, data)
        if not discovery.get("candidate_possible"):
            return None, discovery
        built = GATHER_SLICE.build_candidate(data, discovery)
        return onnx.load_model_from_string(built), discovery
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def lookup_variants(original: onnx.ModelProto) -> list[tuple[onnx.ModelProto, dict[str, Any]]]:
    """Replace a Gather table by another same-index Gather plus an exact scalar."""
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in original.graph.initializer}
    groups: dict[tuple[str, int], list[tuple[int, onnx.NodeProto]]] = defaultdict(list)
    for index, node in enumerate(original.graph.node):
        if node.op_type != "Gather" or len(node.input) < 2 or node.input[0] not in arrays or not node.output:
            continue
        axis = next((int(attr.i) for attr in node.attribute if attr.name == "axis"), 0)
        groups[(node.input[1], axis)].append((index, node))
    result = []
    for (indices, axis), rows in groups.items():
        for target_index, target in rows:
            target_array = arrays[target.input[0]]
            for base_index, base in rows:
                if base_index >= target_index:
                    continue  # preserve topological order
                base_array = arrays[base.input[0]]
                if target_array.shape != base_array.shape or target_array.dtype != base_array.dtype:
                    continue
                difference = target_array - base_array
                scalar = difference.reshape(-1)[0]
                if not np.array_equal(difference, np.full(difference.shape, scalar, dtype=difference.dtype)):
                    continue
                candidate = copy.deepcopy(original)
                scalar_name = f"lookup_scalar_{target_index}_{base_index}"
                candidate.graph.initializer.append(
                    numpy_helper.from_array(np.asarray(scalar, dtype=target_array.dtype), scalar_name)
                )
                old_table = target.input[0]
                replacement = helper.make_node(
                    "Add", [base.output[0], scalar_name], [target.output[0]],
                    name=f"lookup_relation_{target_index}_{base_index}",
                )
                candidate.graph.node[target_index].CopyFrom(replacement)
                if not any(old_table in node.input for node in candidate.graph.node):
                    keep = [item for item in candidate.graph.initializer if item.name != old_table]
                    del candidate.graph.initializer[:]
                    candidate.graph.initializer.extend(keep)
                result.append((candidate, {
                    "indices": indices, "axis": axis, "target_table": old_table,
                    "base_table": base.input[0], "scalar": scalar.item(),
                    "estimated_saved_params": int(target_array.size - 1),
                }))
    return result


def optimize(candidate: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return EXACT.optimize(candidate)[0]
    except Exception:
        return candidate


def main() -> int:
    started = time.monotonic()
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    authority = json.loads((HERE / "authority.json").read_text())
    scope = {int(row["task"]): int(row["cost"]) for row in authority["scope"]}
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "authority": authority["authority"], "authority_sha256": AUTHORITY_SHA256,
        "scope_count": len(scope), "results": [], "finalists": [],
        "families": ["exact_noop_dead_dedupe", "broadcast_initializer_shave",
                     "einsum_unit_operand", "gather_arithmetic_to_slice",
                     "same_index_lookup_relation", "convtranspose_initializer_audit"],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(scope, key=lambda value: (-scope[value], value)):
            data = archive.read(f"task{task:03d}.onnx")
            original = onnx.load_model_from_string(data)
            variants: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
            try:
                exact, stats = EXACT.optimize(original)
                variants.append(("exact_noop_dead_dedupe", exact, stats))
            except Exception as exc:  # noqa: BLE001
                variants.append(("exact_noop_dead_dedupe_error", original, {"error": f"{type(exc).__name__}: {exc}"}))
            scalar, scalar_meta = scalar_variant(original)
            if scalar is not None:
                variants.append(("broadcast_initializer_shave", optimize(scalar), scalar_meta))
            for candidate, meta in einsum_unit_variants(original):
                variants.append(("einsum_unit_operand", optimize(candidate), meta))
            candidate, meta = gather_slice_variant(task, data)
            if candidate is not None:
                variants.append(("gather_arithmetic_to_slice", optimize(candidate), meta))
            for candidate, meta in lookup_variants(original):
                variants.append(("same_index_lookup_relation", optimize(candidate), meta))

            seen = {hashlib.sha256(data).hexdigest()}
            unique = []
            for family, candidate, meta in variants:
                candidate_data = candidate.SerializeToString()
                digest = hashlib.sha256(candidate_data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                unique.append((family, candidate, candidate_data, digest, meta))
            items = cases(task)
            task_row: dict[str, Any] = {
                "task": task, "authority_cost": scope[task],
                "node_count": len(original.graph.node),
                "initializer_count": len(original.graph.initializer),
                "convtranspose_nodes": sum(node.op_type == "ConvTranspose" for node in original.graph.node),
                "variant_count": len(unique), "variants": [],
            }
            for family, candidate, candidate_data, digest, meta in unique:
                row: dict[str, Any] = {"family": family, "sha256": digest, "meta": meta}
                ok, error = structural(candidate)
                row["structural_ok"] = ok
                row["structural_error"] = error
                if not ok:
                    task_row["variants"].append(row)
                    continue
                known = evaluate(make_session(candidate, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1), items)
                row["known"] = known
                if not clean(known) or known["accuracy"] < 0.95:
                    task_row["variants"].append(row)
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix=f"simplify406_{task:03d}_", dir="/tmp") as work:
                        profile = scoring.score_and_verify(
                            candidate, task, work, family, require_correct=False
                        )
                except Exception as exc:  # noqa: BLE001
                    row["profile_error"] = f"{type(exc).__name__}: {exc}"
                    task_row["variants"].append(row)
                    continue
                row["profile"] = profile
                if profile is None or int(profile["cost"]) >= scope[task]:
                    task_row["variants"].append(row)
                    continue
                configs = []
                for name, optimization, threads in (
                    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
                    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
                    ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
                    ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
                ):
                    tested = evaluate(make_session(candidate, optimization, threads), items)
                    configs.append({"name": name, **tested})
                row["known_four_configs"] = configs
                if not all(clean(value) and value["accuracy"] >= 0.95 for value in configs):
                    task_row["variants"].append(row)
                    continue
                cost = int(profile["cost"])
                path = OUT / f"task{task:03d}_cost{cost}_{digest[:12]}.onnx"
                path.write_bytes(candidate_data)
                finalist = {
                    "task": task, "authority_cost": scope[task], "candidate_cost": cost,
                    "half": 2 * cost <= scope[task], "known_exact": known["right"] == known["total"],
                    "candidate_path": str(path.relative_to(ROOT)), **row,
                }
                task_row["variants"].append(finalist)
                report["finalists"].append(finalist)
            report["results"].append(task_row)
            print(json.dumps({
                "task": task, "cost": scope[task], "variants": len(unique),
                "finalists": sum("candidate_path" in value for value in task_row["variants"]),
                "convtranspose": task_row["convtranspose_nodes"],
            }), flush=True)

    report["finalists"].sort(key=lambda row: (
        int(row["task"]), int(row["candidate_cost"]), str(row["sha256"]),
    ))
    best = {}
    for row in report["finalists"]:
        best.setdefault(int(row["task"]), row)
    report["best_by_task"] = list(best.values())
    report["elapsed_seconds"] = time.monotonic() - started
    EVIDENCE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "finalist_count": len(report["finalists"]), "winner_tasks": sorted(best),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
