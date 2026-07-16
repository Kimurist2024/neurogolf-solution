#!/usr/bin/env python3
"""Independent second-pass scan for strict-lower cost<=10 task models.

This lane is deliberately fail-closed.  It never promotes or rewrites the root
submission.  It searches forms which were not covered by lowcost25_scan_293:

* every distinct historical model found in extracted trees and zip archives;
* ReverseSequence prefix reversals (including the fixed-length cost-1 trick);
* legacy attribute-only Slice/Pad and all Transpose permutations;
* strict-lower ConvTranspose initializer crops with alignment-preserving pads;
* scalarized/pruned Einsum initializer-factor variants.

Only candidates which are strictly cheaper under the official scorer, have a
truthful static runtime shape, and are exact on known plus 2x2000 fresh cases in
four ORT configurations are reported.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
PREVIOUS_AUTHORITY = ROOT / "submission_base_8010.03.zip"
PREVIOUS_AUTHORITY_SHA256 = "d772399d4535176b95039690eca59808059add3c0ca2d42e2124f17c705ec2e6"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
TASK_MAP_PATH = ROOT / "docs/golf/task_hash_map.json"
EVIDENCE_PATH = HERE / "evidence.json"
REPORT_PATH = HERE / "REPORT.md"
CANDIDATES_DIR = HERE / "candidates"

TASKS = (16, 17, 53, 61, 87, 135, 140, 197, 223, 276, 305, 307, 309, 312, 326, 337, 373)
EXPECTED_COSTS = {
    16: 10, 17: 10, 53: 6, 61: 10, 87: 5, 135: 2,
    140: 5, 197: 10, 223: 5, 276: 10, 305: 10, 307: 4,
    309: 10, 312: 10, 326: 4, 337: 10, 373: 8,
}
FRESH_PER_SEED = 2_000
EXPECTED_IO = [1, 10, 30, 30]
QUICK_CASES = 12
EXTRACTED_ROOTS = (ROOT / "artifacts", ROOT / "others", ROOT / "scripts/golf", ROOT / "submission")


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("lowcost295_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import support: {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha256(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def model_bytes(model: onnx.ModelProto) -> bytes:
    return model.SerializeToString()


def exact_runtime_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("total", -1) >= 0
        and row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("accuracy") == 1.0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def exact_four(rows: dict[str, Any]) -> bool:
    return bool(len(rows) == 4 and all(exact_runtime_row(row) for row in rows.values()))


def fresh_generation_clean(row: dict[str, Any]) -> bool:
    return bool(
        row.get("accepted") == FRESH_PER_SEED
        and row.get("generation_errors") == 0
        and row.get("conversion_skips") == 0
    )


def quick_exact(data: bytes, cases: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    try:
        runtime = SUPPORT.make_session(data, True, 1)
        row, _ = SUPPORT.evaluate_config(runtime, cases[:QUICK_CASES], None)
    except Exception as exc:  # noqa: BLE001
        row = {
            "total": min(QUICK_CASES, len(cases)), "right": 0, "wrong": 0,
            "errors": min(QUICK_CASES, len(cases)),
            "session_error": f"{type(exc).__name__}: {exc}",
        }
    # The quick screen checks signs/errors/shapes.  Non-finite and margin are
    # also fail-closed here, before the expensive four-config pass.
    return exact_runtime_row(row), row


def cheap_declared_cost(model: onnx.ModelProto) -> int | None:
    """Lower-bound filter; official_profile remains authoritative."""
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        params = SUPPORT.scoring.calculate_params(inferred)
        if params is None:
            return None
        io_names = {value.name for value in list(inferred.graph.input) + list(inferred.graph.output)}
        values = {
            value.name: value
            for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)
        }
        memory = 0
        for node in inferred.graph.node:
            for name in node.output:
                if not name or name in io_names:
                    continue
                value = values[name]
                tensor = value.type.tensor_type
                dims = [int(dim.dim_value) for dim in tensor.shape.dim]
                if not dims or any(dim <= 0 for dim in dims):
                    return None
                dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
                memory += math.prod(dims) * np.dtype(dtype).itemsize
        return int(params + memory)
    except Exception:  # noqa: BLE001
        return None


def canonical_model(node: onnx.NodeProto, opset: int, initializers: list[onnx.TensorProto] | None = None) -> onnx.ModelProto:
    graph = helper.make_graph(
        [node], "lowcost_second_pass_295",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_IO)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_IO)],
        initializer=initializers or [],
    )
    model = helper.make_model(
        graph, producer_name="lowcost_second_pass_295",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = 8
    return model


def reverse_sequence_candidates(task: int) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    del task
    # Only batch_axis=0 has a one-element lengths tensor and is strict-lower
    # for every assigned task.  All legal prefix lengths are enumerated.
    for time_axis, dimension in ((1, 10), (2, 30), (3, 30)):
        for length in range(dimension + 1):
            for dtype, dtype_name in ((np.int64, "i64"), (np.int32, "i32")):
                lengths = numpy_helper.from_array(np.asarray([length], dtype=dtype), name="lengths")
                node = helper.make_node(
                    "ReverseSequence", ["input", "lengths"], ["output"],
                    batch_axis=0, time_axis=time_axis,
                )
                yield (
                    f"reverse_sequence_axis{time_axis}_len{length}_{dtype_name}",
                    canonical_model(node, 10, [lengths]),
                    {"family": "ReverseSequence", "batch_axis": 0, "time_axis": time_axis, "length": length},
                )


def transpose_candidates(task: int) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    del task
    for perm in itertools.permutations(range(4)):
        node = helper.make_node("Transpose", ["input"], ["output"], perm=list(perm))
        label = "transpose_" + "".join(str(value) for value in perm)
        yield label, canonical_model(node, 13), {"family": "Transpose", "perm": list(perm)}


def legacy_slice_candidates(task: int) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    del task
    dimensions = {1: 10, 2: 30, 3: 30}
    axes = (1, 2, 3)
    for count in range(1, 4):
        for subset in itertools.combinations(axes, count):
            starts = [-1 for _ in subset]
            ends = [-(dimensions[axis] + 1) for axis in subset]
            steps = [-1 for _ in subset]
            node = helper.make_node(
                "Slice", ["input"], ["output"],
                starts=starts, ends=ends, axes=list(subset), steps=steps,
            )
            label = "legacy_slice_reverse_" + "_".join(str(axis) for axis in subset)
            yield label, canonical_model(node, 9), {
                "family": "legacy_attribute_Slice", "axes": list(subset),
                "starts": starts, "ends": ends, "steps": steps,
            }


def legacy_pad_candidates(task: int) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    # Spatial translations are relevant only to the spatial-transform lane;
    # channel translations are relevant only to fixed-colormap models.
    shifts: list[tuple[int, int, int]] = []
    if task in {53, 87, 135, 140, 223, 307, 326, 373}:
        for row in range(-10, 11):
            for col in range(-10, 11):
                if row or col:
                    shifts.append((0, row, col))
    if task in {16, 276, 309, 337}:
        shifts.extend((channel, 0, 0) for channel in range(-9, 10) if channel)
    for channel, row, col in shifts:
        begin = [0, channel, row, col]
        end = [0, -channel, -row, -col]
        pads = begin + end
        node = helper.make_node(
            "Pad", ["input"], ["output"], mode="constant", value=0.0, pads=pads,
        )
        label = f"legacy_pad_shift_c{channel}_r{row}_w{col}"
        yield label, canonical_model(node, 10), {
            "family": "legacy_attribute_Pad", "pads": pads, "value": 0.0,
        }


def convtranspose_crop_candidates(
    task: int, authority: onnx.ModelProto,
) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    if len(authority.graph.node) != 1 or authority.graph.node[0].op_type != "ConvTranspose":
        return
    if len(authority.graph.initializer) != 1:
        return
    node = authority.graph.node[0]
    dense = numpy_helper.to_array(authority.graph.initializer[0])
    if dense.ndim != 4 or dense.shape[:2] != (1, 1):
        return
    attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    strides = list(attrs.get("strides", [1, 1]))
    dilations = list(attrs.get("dilations", [1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    output_padding = list(attrs.get("output_padding", [0, 0]))
    kernel_shape = list(attrs.get("kernel_shape", [30, 30]))
    h, w = dense.shape[2:]
    current_cost = EXPECTED_COSTS[task]
    variants = (("preserve_nan", dense), ("nan_to_zero", np.nan_to_num(dense, nan=0.0)))
    seen: set[str] = set()
    for value_mode, array in variants:
        for r0 in range(h):
            for r1 in range(r0 + 1, h + 1):
                for c0 in range(w):
                    for c1 in range(c0 + 1, w + 1):
                        cropped = np.ascontiguousarray(array[:, :, r0:r1, c0:c1])
                        if cropped.size >= current_cost:
                            continue
                        new_top = pads[0] - r0 * strides[0]
                        new_left = pads[1] - c0 * strides[1]
                        total_h = (
                            strides[0] * (cropped.shape[2] - 1) + output_padding[0]
                            + dilations[0] * (kernel_shape[0] - 1) + 1 - 30
                        )
                        total_w = (
                            strides[1] * (cropped.shape[3] - 1) + output_padding[1]
                            + dilations[1] * (kernel_shape[1] - 1) + 1 - 30
                        )
                        new_bottom = total_h - new_top
                        new_right = total_w - new_left
                        new_pads = [new_top, new_left, new_bottom, new_right]
                        if any(value < 0 for value in new_pads):
                            continue
                        model = copy.deepcopy(authority)
                        model.graph.initializer[0].CopyFrom(
                            numpy_helper.from_array(cropped, name=authority.graph.initializer[0].name)
                        )
                        target = model.graph.node[0]
                        for index in reversed(range(len(target.attribute))):
                            if target.attribute[index].name == "pads":
                                del target.attribute[index]
                        target.attribute.append(helper.make_attribute("pads", new_pads))
                        data_hash = sha256(model_bytes(model))
                        if data_hash in seen:
                            continue
                        seen.add(data_hash)
                        label = f"conv_crop_{value_mode}_r{r0}_{r1}_c{c0}_{c1}"
                        yield label, model, {
                            "family": "ConvTranspose_initializer_crop",
                            "source_slice": [r0, r1, c0, c1], "value_mode": value_mode,
                            "new_shape": list(cropped.shape), "new_pads": new_pads,
                        }


def einsum_scalar_prune_candidates(
    task: int, authority: onnx.ModelProto,
) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    del task
    if len(authority.graph.node) != 1 or authority.graph.node[0].op_type != "Einsum":
        return
    if len(authority.graph.initializer) != 1:
        return
    node = authority.graph.node[0]
    init = authority.graph.initializer[0]
    init_name = init.name
    equation = next(
        (helper.get_attribute_value(item) for item in node.attribute if item.name == "equation"), None
    )
    if isinstance(equation, bytes):
        equation = equation.decode("ascii")
    if not equation or "->" not in equation:
        return
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(node.input):
        return
    init_terms = [term for name, term in zip(node.input, terms) if name == init_name]
    if not init_terms or any(len(term) != 1 for term in init_terms):
        return
    distinct_axes = sorted(set(init_terms))
    values = numpy_helper.to_array(init).reshape(-1)
    scalar_values = sorted({float(value) for value in values} | {-1.0, 0.0, 1.0, 1e-8, 10.0, 32.0, 1e7})
    seen: set[str] = set()
    # Keep at most one initializer factor per selected equation index.  This
    # covers every background-mask role while removing numerical power factors.
    for mask in range(1 << len(distinct_axes)):
        kept_axes = {distinct_axes[index] for index in range(len(distinct_axes)) if mask >> index & 1}
        for scalar in scalar_values if kept_axes else (None,):
            new_inputs: list[str] = []
            new_terms: list[str] = []
            already_kept: set[str] = set()
            for name, term in zip(node.input, terms):
                if name != init_name:
                    new_inputs.append(name)
                    new_terms.append(term)
                elif term in kept_axes and term not in already_kept:
                    already_kept.add(term)
                    new_inputs.append(name)
                    new_terms.append(term)
            model = copy.deepcopy(authority)
            replacement = helper.make_node(
                "Einsum", new_inputs, ["output"], equation=",".join(new_terms) + "->" + rhs,
            )
            model.graph.node[0].CopyFrom(replacement)
            if kept_axes:
                dtype = numpy_helper.to_array(init).dtype
                model.graph.initializer[0].CopyFrom(
                    numpy_helper.from_array(np.asarray([scalar], dtype=dtype), name=init_name)
                )
            else:
                del model.graph.initializer[:]
            data_hash = sha256(model_bytes(model))
            if data_hash in seen:
                continue
            seen.add(data_hash)
            scalar_label = "none" if scalar is None else format(float(scalar), ".8g").replace("-", "m")
            label = f"einsum_scalar_{scalar_label}_axes_{''.join(sorted(kept_axes)) or 'none'}"
            yield label, model, {
                "family": "Einsum_scalar_and_factor_prune",
                "original_equation": equation,
                "kept_initializer_axes": sorted(kept_axes),
                "scalar": scalar,
            }


def generated_candidates(
    task: int, authority: onnx.ModelProto,
) -> Iterable[tuple[str, onnx.ModelProto, dict[str, Any]]]:
    yield from reverse_sequence_candidates(task)
    yield from transpose_candidates(task)
    yield from legacy_slice_candidates(task)
    yield from legacy_pad_candidates(task)
    yield from convtranspose_crop_candidates(task, authority)
    yield from einsum_scalar_prune_candidates(task, authority)


def extracted_archive_models() -> Iterable[tuple[int, str, bytes]]:
    prefixes = {f"task{task:03d}": task for task in TASKS}
    for root in EXTRACTED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            name = path.name
            task = next(
                (value for prefix, value in prefixes.items() if name == prefix + ".onnx" or name.startswith(prefix + "_")),
                None,
            )
            if task is None:
                continue
            try:
                yield task, rel(path), path.read_bytes()
            except OSError:
                continue


def zip_archive_models() -> Iterable[tuple[int, str, bytes]]:
    wanted = {f"task{task:03d}.onnx": task for task in TASKS}
    for path in ROOT.rglob("*.zip"):
        if path in {AUTHORITY, PREVIOUS_AUTHORITY}:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    task = wanted.get(Path(member).name)
                    if task is None:
                        continue
                    try:
                        yield task, f"{rel(path)}::{member}", archive.read(member)
                    except (KeyError, OSError, RuntimeError):
                        continue
        except (OSError, zipfile.BadZipFile, RuntimeError):
            continue


def evaluate_candidate(
    task: int,
    label: str,
    source: str,
    data: bytes,
    authority_cost: int,
    authority_sha: str,
    known_cases: list[dict[str, Any]],
    task_map: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task, "label": label, "source": source,
        "candidate_sha256": sha256(data), "candidate_file_bytes": len(data),
        "authority_cost": authority_cost, "authority_sha256": authority_sha,
        "metadata": metadata,
    }
    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:  # noqa: BLE001
        row["classification"] = "REJECT_LOAD"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    lower_bound = cheap_declared_cost(model)
    row["declared_cost_lower_bound"] = lower_bound
    if lower_bound is None or lower_bound >= authority_cost:
        row["classification"] = "REJECT_NOT_DECLARED_STRICT_LOWER"
        return row
    # A twelve-case ORT sign/shape/margin screen removes almost every generic
    # single-op form.  Run it before profiler/shape-trace work; all survivors
    # still receive the complete structural and official-cost gates below.
    quick_pass, quick = quick_exact(data, known_cases)
    row["known_quick"] = quick
    if not quick_pass:
        row["classification"] = "REJECT_KNOWN_QUICK_NOT_EXACT"
        return row
    structure = SUPPORT.structural_audit(task, model, data)
    row["structure"] = structure
    if not structure.get("pass"):
        row["classification"] = "REJECT_STRUCTURE_FAIL_CLOSED"
        return row
    profile = SUPPORT.official_profile(task, model, f"low295_{label[:40]}")
    row["official_profile"] = profile
    if profile is None:
        row["classification"] = "REJECT_OFFICIAL_UNSCORABLE"
        return row
    actual_cost = int(profile["cost"])
    if actual_cost >= authority_cost:
        row["classification"] = "REJECT_NOT_ACTUAL_STRICT_LOWER"
        return row
    known_four = SUPPORT.evaluate_four(data, known_cases)
    row["known_four"] = known_four
    if not exact_four(known_four):
        row["classification"] = "REJECT_KNOWN_FOUR_NOT_EXACT"
        return row
    fresh_rows = []
    for seed in (295_000_000 + task, 295_100_000 + task):
        cases, generation = SUPPORT.fresh_cases(task, seed, task_map)
        four = SUPPORT.evaluate_four(data, cases)
        passed = fresh_generation_clean(generation) and exact_four(four)
        fresh_rows.append({"seed": seed, "generation": generation, "four": four, "exact_pass": passed})
    row["fresh"] = {
        "count_per_seed": FRESH_PER_SEED,
        "runs": fresh_rows,
        "exact_pass": all(item["exact_pass"] for item in fresh_rows),
    }
    row["classification"] = (
        "PASS_SAFE_STRICT_LOWER" if row["fresh"]["exact_pass"] else "REJECT_FRESH_NOT_EXACT"
    )
    return row


def main() -> int:
    started = time.monotonic()
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("8011.05 authority SHA mismatch")
    if digest(PREVIOUS_AUTHORITY) != PREVIOUS_AUTHORITY_SHA256:
        raise RuntimeError("8010.03 comparison authority SHA mismatch")
    if set(TASKS) != set(EXPECTED_COSTS) or len(TASKS) != 17:
        raise RuntimeError("task inventory mismatch")
    task_map = json.loads(TASK_MAP_PATH.read_text(encoding="utf-8"))
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    authority_models: dict[int, onnx.ModelProto] = {}
    authority_data: dict[int, bytes] = {}
    authority_inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as latest, zipfile.ZipFile(PREVIOUS_AUTHORITY) as previous:
        for task in TASKS:
            data = latest.read(f"task{task:03d}.onnx")
            old = previous.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            profile = SUPPORT.official_profile(task, model, "authority801105")
            if profile is None or int(profile["cost"]) != EXPECTED_COSTS[task]:
                raise RuntimeError(f"task{task:03d} authority cost mismatch: {profile}")
            authority_data[task] = data
            authority_models[task] = model
            authority_inventory.append({
                "task": task, "cost": int(profile["cost"]), "score": profile["score"],
                "sha256": sha256(data), "file_bytes": len(data),
                "changed_from_8010_03": data != old,
                "previous_sha256": sha256(old),
                "ops": [node.op_type for node in model.graph.node],
                "initializer_elements": SUPPORT.scoring.calculate_params(model),
            })

    known_by_task = {task: SUPPORT.known_cases(task)[0] for task in TASKS}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    generation_counts: Counter[str] = Counter()

    # Generated second-pass families.
    for task in TASKS:
        for label, model, metadata in generated_candidates(task, authority_models[task]):
            data = model_bytes(model)
            key = (task, sha256(data))
            generation_counts[metadata["family"]] += 1
            if key in seen or key == (task, sha256(authority_data[task])):
                continue
            seen.add(key)
            row = evaluate_candidate(
                task, label, "generated", data, EXPECTED_COSTS[task], sha256(authority_data[task]),
                known_by_task[task], task_map, metadata,
            )
            rows.append(row)
            if len(rows) % 250 == 0:
                print(json.dumps({"phase": "generated", "rows": len(rows), "latest_task": task}), flush=True)

    # Historical extracted and zipped archives.  De-duplicate before parsing.
    archive_locations: dict[tuple[int, str], list[str]] = defaultdict(list)
    archive_data: dict[tuple[int, str], bytes] = {}
    archive_seen_locations = 0
    for task, source, data in itertools.chain(extracted_archive_models(), zip_archive_models()):
        archive_seen_locations += 1
        key = (task, sha256(data))
        archive_locations[key].append(source)
        archive_data.setdefault(key, data)
    for index, ((task, candidate_sha), locations) in enumerate(sorted(archive_locations.items()), start=1):
        key = (task, candidate_sha)
        if key in seen or candidate_sha == sha256(authority_data[task]):
            continue
        seen.add(key)
        row = evaluate_candidate(
            task, f"historical_{candidate_sha[:12]}", locations[0], archive_data[key],
            EXPECTED_COSTS[task], sha256(authority_data[task]), known_by_task[task], task_map,
            {
                "family": "historical_archive", "location_count": len(locations),
                "locations_sample": locations[:20],
            },
        )
        rows.append(row)
        if index % 100 == 0:
            print(json.dumps({"phase": "archives", "unique_index": index, "rows": len(rows)}), flush=True)

    passes_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["classification"] == "PASS_SAFE_STRICT_LOWER":
            passes_by_task[int(row["task"])].append(row)
    reported: list[dict[str, Any]] = []
    for task, candidates in sorted(passes_by_task.items()):
        candidates.sort(key=lambda item: (
            int(item["official_profile"]["cost"]), item["candidate_file_bytes"], item["candidate_sha256"],
        ))
        winner = candidates[0]
        data = next(
            (archive_data[(task, winner["candidate_sha256"])]
             for _ in [0] if (task, winner["candidate_sha256"]) in archive_data),
            None,
        )
        if data is None:
            # Regenerate by hash without retaining thousands of generated byte blobs.
            for label, model, _ in generated_candidates(task, authority_models[task]):
                candidate = model_bytes(model)
                if sha256(candidate) == winner["candidate_sha256"]:
                    data = candidate
                    break
        if data is None:
            raise RuntimeError("winner serialization could not be recovered")
        cost = int(winner["official_profile"]["cost"])
        output = CANDIDATES_DIR / f"task{task:03d}_{winner['label']}_cost{cost}.onnx"
        output.write_bytes(data)
        if digest(output) != winner["candidate_sha256"]:
            raise RuntimeError("saved winner hash mismatch")
        winner["saved_path"] = rel(output)
        reported.append({
            "task": task, "path": rel(output), "sha256": winner["candidate_sha256"],
            "cost": cost, "authority_cost": EXPECTED_COSTS[task],
            "score": winner["official_profile"]["score"], "label": winner["label"],
        })

    classifications = Counter(row["classification"] for row in rows)
    task_results = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        passing = next((item for item in reported if item["task"] == task), None)
        task_results.append({
            "task": task, "authority_cost": EXPECTED_COSTS[task],
            "status": "SAFE_STRICT_LOWER_FOUND" if passing else "NO_SAFE_STRICT_LOWER_FOUND",
            "attempts": len(task_rows),
            "structure_pass": sum(bool(row.get("structure", {}).get("pass")) for row in task_rows),
            "actual_strict_lower": sum(
                bool(row.get("official_profile") and int(row["official_profile"]["cost"]) < EXPECTED_COSTS[task])
                for row in task_rows
            ),
            "known_quick_exact": sum(
                exact_runtime_row(row["known_quick"])
                for row in task_rows
                if "known_quick" in row
            ),
            "winner": passing,
        })

    payload = {
        "lane": "lowcost_second_pass_295",
        "decision": "SAFE_STRICT_LOWER_FOUND" if reported else "NO_SAFE_STRICT_LOWER_FOUND",
        "authority": {
            "zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256, "leaderboard": 8011.05,
            "comparison_zip": rel(PREVIOUS_AUTHORITY), "comparison_sha256": PREVIOUS_AUTHORITY_SHA256,
            "targets_changed_from_8010_03": [
                row["task"] for row in authority_inventory if row["changed_from_8010_03"]
            ],
        },
        "policy": {
            "existing_score25_skipped": [67, 129, 179, 241],
            "strict_lower_by_at_least_one": True,
            "known_exact_required": True,
            "fresh_count_per_seed": FRESH_PER_SEED,
            "fresh_seeds": ["295000000+task", "295100000+task"],
            "four_ort_configs": [name for name, _, _ in SUPPORT.CONFIGS],
            "nonfinite_fail_closed": True,
            "runtime_shape_truthful_required": True,
            "margin_small_positive_fail_closed": True,
            "automatic_promotion": False,
            "root_or_stage_modified": False,
            "kimi_used": False,
        },
        "coverage": {
            "tasks": list(TASKS), "task_count": len(TASKS),
            "generated_family_attempts_before_hash_dedup": dict(generation_counts),
            "archive_model_locations_seen": archive_seen_locations,
            "archive_unique_task_hashes": len(archive_locations),
            "evaluated_unique_candidate_rows": len(rows),
            "classification_counts": dict(classifications),
            "safe_improvement_task_count": len(reported),
        },
        "authority_inventory": authority_inventory,
        "task_results": task_results,
        "reported_candidates": reported,
        "notable_rejections": [{
            "task": 223,
            "candidate_sha256": "9b39d460a626086b3a8d427c6379242b0db532c8d8f6dffacd2bd2b64e9906b9",
            "candidate_form": "MaxRoiPool dense ROI initializer converted to nnz=2 sparse initializer",
            "checker_and_ort": "full checker, strict shape, and all 265 known cases in four ORT configs pass",
            "decision": "REJECT_OFFICIAL_SCORER_CRASH",
            "official_error": "AttributeError: name",
            "detail": "scripts/golf/lowcost_second_pass_295/sparse_task223_audit.json",
        }],
        "candidate_rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "only scripts/golf/lowcost_second_pass_295",
    }
    EVIDENCE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Low-cost second pass (8011.05 authority)", "",
        f"Decision: **{payload['decision']}**", "",
        "既存25点の4件を除外し、cost<=10の残り17件を最新authorityに対して再探索した。",
        "ReverseSequence、legacy属性だけのSlice/Pad、全Transpose、ConvTransposeの",
        "initializer crop、Einsum scalar/factor prune、および全ローカル履歴archiveを対象にした。",
        "",
        f"- generated attempts (pre-dedup): {sum(generation_counts.values())}",
        f"- historical locations: {archive_seen_locations}",
        f"- historical unique task/hash pairs: {len(archive_locations)}",
        f"- evaluated unique candidates: {len(rows)}",
        f"- safe strict-lower tasks: {len(reported)}",
        f"- targets changed from 8010.03: {payload['authority']['targets_changed_from_8010_03']}",
        "",
    ]
    if reported:
        report_lines.append("## Verified candidates")
        report_lines.append("")
        for item in reported:
            report_lines.append(
                f"- task{item['task']:03d}: cost {item['authority_cost']} -> {item['cost']} "
                f"({item['path']}, sha256 {item['sha256']})"
            )
    else:
        report_lines.extend([
            "既知完全一致からfresh 2x2000へ進める、strict-lowerかつfail-closedな候補はなかった。",
            "root submission / all_scores / stage は変更していない。",
        ])
    report_lines.extend([
        "", "## Notable rejected lead", "",
        "task223 の sparse initializer 版（dense 5要素 → stored 2要素）は full checker、",
        "strict shape、ORT既知265件×4設定を通過した。しかし公式 scorer の",
        "`calculate_memory` が `SparseTensorProto.name`（存在しないfield）を参照して",
        "`AttributeError: name` になるため、スコア不能のエラータスクとして棄却した。",
        "再現結果: `sparse_task223_audit.json`",
    ])
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "coverage": payload["coverage"],
        "reported": reported, "evidence": rel(EVIDENCE_PATH), "report": rel(REPORT_PATH),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
