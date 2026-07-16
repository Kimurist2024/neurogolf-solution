#!/usr/bin/env python3
"""Fail-closed cost-0/1 and strict-lower scan for the 8010.03 extra-15 set."""

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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8010.03.zip"
AUTHORITY_SHA256 = "d772399d4535176b95039690eca59808059add3c0ca2d42e2124f17c705ec2e6"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS = (130, 73, 372, 103, 164, 172, 210, 311, 314, 63, 43, 45, 47, 52, 166)
AUTHORITY_COSTS = {
    130: 11, 73: 12, 372: 13, 103: 15, 164: 16,
    172: 16, 210: 16, 311: 16, 314: 19,
    63: 20, 43: 20, 45: 20, 47: 20, 52: 20, 166: 20,
}
FRESH_PER_SEED = 2_000
CANDIDATE_DIR = HERE / "candidates"
EVIDENCE = HERE / "evidence.json"
EXPECTED = (1, 10, 30, 30)

SPEC = {
    130: "3x3 block compression with gray-noise rejection; output is 3x3.",
    73: "Move each blue tower cap to the gray base row while retaining the gray support.",
    372: "Overlay the two colored halves separated by a gray horizontal divider.",
    103: "Emit blue/orange at [0,0] according to horizontal/vertical symmetry.",
    164: "Concatenate a square color grid with its horizontal reflection.",
    172: "Concatenate a square color grid with its vertical reflection.",
    210: "Concatenate a blue mask with its vertical reflection.",
    311: "Concatenate an orange mask with its horizontal reflection.",
    314: "Complete same-color pairs inside fixed 3x3 cells.",
    63: "Paint green empty interior rows/columns inside a colored border construction.",
    43: "Paint red Cartesian intersections from gray top/right markers.",
    45: "Fill rows whose left/right endpoint colors agree.",
    47: "Draw cyan/orange row/column crosses and red cross-intersections.",
    52: "Mark each uniform input row gray and other rows black.",
    166: "Fill the bounding strip rectangle red behind cyan runs.",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module("extra15_scan287_helpers", ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def make_model(
    node: onnx.NodeProto,
    initializers: list[onnx.TensorProto] | None = None,
    output_dtype: int = TensorProto.FLOAT,
    name: str = "extra15_candidate",
) -> onnx.ModelProto:
    graph = helper.make_graph(
        [node], name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(EXPECTED))],
        [helper.make_tensor_value_info("output", output_dtype, list(EXPECTED))],
        initializer=initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    model.producer_name = "extra15_cost25_scan_294"
    return model


def scalar(name: str, value: float | int, dtype: np.dtype[Any]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def add_variant(
    rows: list[dict[str, Any]], seen: set[str], model: onnx.ModelProto,
    name: str, family: str, proof: str,
) -> None:
    data = model.SerializeToString()
    digest = sha256(data)
    if digest in seen:
        return
    seen.add(digest)
    rows.append({
        "name": name, "family": family, "proof": proof,
        "sha256": digest, "_model": model, "_data": data,
    })


def generic_variants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    unary = ("Identity", "Abs", "Relu", "Sign", "Sqrt", "Floor", "Ceil", "Round", "Erf", "Tanh")
    for op in unary:
        add_variant(rows, seen, make_model(helper.make_node(op, ["input"], ["output"])),
                    op.lower(), "unary_no_param", f"one output-only {op}")
    for op in ("Add", "Mul", "Max", "Min", "Mean", "Sum", "MatMul"):
        add_variant(rows, seen, make_model(helper.make_node(op, ["input", "input"], ["output"])),
                    f"{op.lower()}_self", "binary_no_param",
                    f"output-only {op}(input,input); Add/Mul preserve one-hot sign")
    add_variant(
        rows, seen,
        make_model(helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])),
        "transpose_hw", "score25_template", "exact task179/task241 score-25 transpose template",
    )
    for upper in (0, 1):
        add_variant(rows, seen, make_model(helper.make_node("Trilu", ["input"], ["output"], upper=upper)),
                    f"trilu_{'upper' if upper else 'lower'}", "spatial_no_param", "fixed triangular mask")
    einsums = {
        "task067_filter": "bkrc,bjcw->bkrc",
        "same_row_filter": "bkrc,bjrw->bkrc",
        "same_col_filter": "bkrc,bjhc->bkrc",
        "same_cell_filter": "bkrc,bjrc->bkrc",
        "row_by_col_outer": "bkrw,bjhc->bkrc",
        "col_by_row_outer": "bkhc,bjrw->bkrc",
        "channel_row_by_any_col": "bkrw,bjhc->bkrc",
        "channel_col_by_any_row": "bkhc,bjrw->bkrc",
        "matrix_square": "bkrh,bkhc->bkrc",
        "matrix_transpose_left": "bkhr,bkhc->bkrc",
        "matrix_transpose_right": "bkrh,bkch->bkrc",
    }
    for name, equation in einsums.items():
        add_variant(rows, seen,
                    make_model(helper.make_node("Einsum", ["input", "input"], ["output"], equation=equation)),
                    f"einsum_{name}", "score25_einsum_template",
                    f"two-input non-giant Einsum {equation}")
    for kh in (2, 3, 4):
        for kw in (2, 3, 4):
            for top in range(kh):
                for left in range(kw):
                    pads = [top, left, kh - 1 - top, kw - 1 - left]
                    node = helper.make_node(
                        "MaxPool", ["input"], ["output"],
                        kernel_shape=[kh, kw], strides=[1, 1], pads=pads,
                    )
                    add_variant(rows, seen, make_model(node),
                                f"maxpool_{kh}x{kw}_p{top}_{left}", "fixed_pool_no_param",
                                f"output-only fixed MaxPool kernel={kh}x{kw} pads={pads}")
    for axis in (1, 2, 3):
        axis_init = scalar("axis", axis, np.int64)
        for reverse, exclusive in itertools.product((0, 1), repeat=2):
            node = helper.make_node(
                "CumSum", ["input", "axis"], ["output"],
                reverse=reverse, exclusive=exclusive,
            )
            add_variant(rows, seen, make_model(node, [axis_init]),
                        f"cumsum_a{axis}_r{reverse}_e{exclusive}", "one_param_cumsum",
                        "one scalar axis initializer; fixed prefix/suffix propagation")
    for axis in (2, 3):
        lens = numpy_helper.from_array(np.asarray([30], dtype=np.int64), "length")
        node = helper.make_node(
            "ReverseSequence", ["input", "length"], ["output"],
            batch_axis=0, time_axis=axis,
        )
        add_variant(rows, seen, make_model(node, [lens]),
                    f"reverse_full_axis{axis}", "one_param_flip",
                    "one sequence-length initializer; fixed full-axis reversal")
    for op, value, dtype, out_type in (
        ("Add", 0.0, np.float32, TensorProto.FLOAT),
        ("Mul", 1.0, np.float32, TensorProto.FLOAT),
        ("Pow", 2.0, np.float32, TensorProto.FLOAT),
        ("Greater", 0.0, np.float32, TensorProto.BOOL),
        ("GreaterOrEqual", 1.0, np.float32, TensorProto.BOOL),
        ("Equal", 1.0, np.float32, TensorProto.BOOL),
    ):
        item = scalar("s", value, dtype)
        node = helper.make_node(op, ["input", "s"], ["output"])
        add_variant(rows, seen, make_model(node, [item], out_type),
                    f"scalar_{op.lower()}_{value:g}", "one_param_threshold",
                    f"one scalar; {op} threshold/sign-preserving probe")
    scalar_x = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "x")
    add_variant(
        rows, seen,
        make_model(helper.make_node("ConvTranspose", ["x", "input"], ["output"]), [scalar_x]),
        "scalar_convtranspose_dynamic_weight", "one_param_dynamic_weight",
        "one finite scalar X with canonical input used as ConvTranspose weights",
    )
    for value in (0.0, 1.0):
        item = scalar("s", value, np.float32)
        node = helper.make_node("GroupNormalization", ["input", "s", "s"], ["output"], num_groups=1)
        add_variant(rows, seen, make_model(node, [item]),
                    f"groupnorm_shared_scalar_{value:g}", "one_param_normalization",
                    "share one finite scalar as GroupNormalization scale and bias")
    return rows


def remove_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    keep = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    del model.graph.value_info[:]


def conv_prefix_variants(task: int, base: onnx.ModelProto) -> list[tuple[str, onnx.ModelProto, str]]:
    result = []
    init = next((item for item in base.graph.initializer if item.name in {"x", "X"}), None)
    if init is None:
        return result
    array = numpy_helper.to_array(init)
    long_axes = [axis for axis in (2, 3) if array.shape[axis] > 1]
    if len(long_axes) != 1:
        return result
    axis = long_axes[0]
    length = array.shape[axis]
    for bypass in ((False, True) if task == 73 else (False,)):
        source = copy.deepcopy(base)
        if bypass:
            last = copy.deepcopy(source.graph.node[-1])
            last.input[1] = "input"
            del source.graph.node[:]
            source.graph.node.extend([last])
            remove_unused_initializers(source)
        node = source.graph.node[-1]
        attrs = {attr.name: attr for attr in node.attribute}
        pads = list(attrs["pads"].ints)
        strides = list(attrs.get("strides", helper.make_attribute("strides", [1, 1])).ints)
        spatial = axis - 2
        for stop in range(1, length + 1):
            trailing = (length - stop) * strides[spatial]
            if pads[spatial + 2] < trailing:
                continue
            for replacement in (1.0, 1_000.0):
                model = copy.deepcopy(source)
                target = next(item for item in model.graph.initializer if item.name == init.name)
                sliced = np.take(array, range(stop), axis=axis)
                sliced = np.nan_to_num(
                    sliced, nan=0.0, posinf=replacement, neginf=-replacement
                ).astype(np.float32)
                target.CopyFrom(numpy_helper.from_array(sliced, init.name))
                target_node = model.graph.node[-1]
                target_pads = next(attr for attr in target_node.attribute if attr.name == "pads")
                updated = list(target_pads.ints)
                updated[spatial + 2] -= trailing
                del target_pads.ints[:]
                target_pads.ints.extend(updated)
                remove_unused_initializers(model)
                name = f"task{task:03d}_convprefix{stop}_finite{int(replacement)}{'_raw' if bypass else ''}"
                proof = (
                    "crop-preserving prefix slice: trailing taps removed and trailing pad "
                    "reduced by removed_count*stride; all retained NaN/Inf replaced finitely"
                )
                result.append((name, model, proof))
    return result


def task_specific_variants(task: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, model, proof in conv_prefix_variants(task, base):
        add_variant(rows, seen, model, name, "initializer_crop_finite", proof)
    if task == 130:
        raw_roi = copy.deepcopy(base)
        roi = copy.deepcopy(raw_roi.graph.node[-1])
        roi.input[0] = "input"
        del raw_roi.graph.node[:]
        raw_roi.graph.node.extend([roi])
        remove_unused_initializers(raw_roi)
        add_variant(rows, seen, raw_roi, "task130_raw_roi", "graph_output_direct",
                    "remove GroupNormalization and feed canonical input directly to RoiAlign")
        gn = copy.deepcopy(base)
        first = copy.deepcopy(gn.graph.node[0])
        first.output[0] = "output"
        del gn.graph.node[:]
        gn.graph.node.extend([first])
        remove_unused_initializers(gn)
        add_variant(rows, seen, gn, "task130_groupnorm_output", "graph_output_direct",
                    "make the first node the canonical graph output and remove RoiAlign")
        for source in ("scale", "bias"):
            shared = copy.deepcopy(gn)
            shared.graph.node[0].input[1] = source
            shared.graph.node[0].input[2] = source
            remove_unused_initializers(shared)
            add_variant(rows, seen, shared, f"task130_groupnorm_shared_{source}",
                        "one_param_normalization", "reuse one scalar for scale and bias")
    if task == 103:
        no_bias = copy.deepcopy(base)
        del no_bias.graph.node[0].input[2]
        remove_unused_initializers(no_bias)
        add_variant(rows, seen, no_bias, "task103_drop_conv_bias", "initializer_removal",
                    "remove optional Conv bias")
    if task == 314:
        no_bias = copy.deepcopy(base)
        del no_bias.graph.node[0].input[2]
        remove_unused_initializers(no_bias)
        add_variant(rows, seen, no_bias, "task314_drop_bias", "initializer_removal",
                    "remove optional ConvTranspose bias")
        kernel = numpy_helper.to_array(next(item for item in base.graph.initializer if item.name == "x"))
        for r0 in range(3):
            for r1 in range(r0 + 1, 4):
                for c0 in range(3):
                    for c1 in range(c0 + 1, 4):
                        if (r0, r1, c0, c1) == (0, 3, 0, 3):
                            continue
                        model = copy.deepcopy(base)
                        item = next(value for value in model.graph.initializer if value.name == "x")
                        item.CopyFrom(numpy_helper.from_array(kernel[:, :, r0:r1, c0:c1], "x"))
                        pads = next(attr for attr in model.graph.node[0].attribute if attr.name == "pads")
                        updated = list(pads.ints)
                        updated[0] -= 3 * r0
                        updated[1] -= 3 * c0
                        updated[2] -= 3 * (3 - r1)
                        updated[3] -= 3 * (3 - c1)
                        if min(updated) < 0:
                            continue
                        del pads.ints[:]
                        pads.ints.extend(updated)
                        add_variant(rows, seen, model,
                                    f"task314_kernel_r{r0}{r1}_c{c0}{c1}",
                                    "initializer_crop_finite",
                                    "crop ConvTranspose X and compensate pads by stride")
    if base.graph.node and base.graph.node[0].op_type == "Einsum":
        node = base.graph.node[0]
        equation = next(helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "equation").decode()
        lhs, rhs = equation.split("->")
        terms = lhs.split(",")
        input_terms = [term for term, name in zip(terms, node.input) if name == "input"]
        for mask in range(1, 1 << len(input_terms)):
            selected = [term for index, term in enumerate(input_terms) if mask & (1 << index)]
            if any(label not in set("".join(selected)) for label in rhs if label.isalpha()):
                continue
            candidate = make_model(
                helper.make_node("Einsum", ["input"] * len(selected), ["output"],
                                 equation=",".join(selected) + "->" + rhs)
            )
            add_variant(rows, seen, candidate,
                        f"task{task:03d}_einsum_input_terms_{mask:x}",
                        "initializer_removal",
                        "remove every learned initializer operand and retain a valid input-only sub-equation")
    return rows


def row_clean_exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0 and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0 and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def four_exact(rows: dict[str, Any]) -> bool:
    if len(rows) != 4 or not all(row_clean_exact(row) for row in rows.values()):
        return False
    raw = {row.get("raw_sha256") for row in rows.values()}
    sign = {row.get("sign_sha256") for row in rows.values()}
    return len(raw) == 1 and len(sign) == 1


def compact_runtime(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error",
        "first_shape_mismatch", "first_sign_mismatch", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def audited_cost(structure: dict[str, Any]) -> int | None:
    trace = structure.get("runtime_intermediate_trace", {})
    memory = trace.get("single_example_intermediate_bytes")
    if not isinstance(memory, int):
        return None
    return int(structure["initializer_elements"]) + memory


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8010.03 authority changed")
    if set(AUTHORITY_COSTS) != set(TASKS):
        raise RuntimeError("authority cost table/task list mismatch")
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    generator_sources = {
        task: ROOT / f"inputs/arc-gen-repo/tasks/task_{task_map[f'{task:03d}']}.py"
        for task in TASKS
    }
    generic = generic_variants()
    structure_cache: dict[str, dict[str, Any]] = {}
    task_rows = []
    candidate_rows = []
    finalists = []
    aggregate = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task in TASKS:
        base_data = authority_data[task]
        base = onnx.load_from_string(base_data)
        profile = SCAN.official_profile(task, base, "authority_801003")
        if profile is None or int(profile["cost"]) != AUTHORITY_COSTS[task] or not profile["correct"]:
            raise RuntimeError(f"authority profile mismatch task{task:03d}: {profile}")
        authority_structure = SCAN.structural_audit(task, base, base_data)
        variants = []
        per_task_seen: set[str] = set()
        for source in [*generic, *task_specific_variants(task, base)]:
            digest = source["sha256"]
            if digest in per_task_seen:
                continue
            per_task_seen.add(digest)
            variants.append(source)
        cases, known_counts = SCAN.known_cases(task)
        task_row = {
            "task": task,
            "generator_hash": task_map[f"{task:03d}"],
            "generator_source": rel(generator_sources[task]),
            "generator_source_sha256": sha256(generator_sources[task].read_bytes()),
            "generator_rule": SPEC[task],
            "authority_sha256": sha256(base_data),
            "authority_profile": profile,
            "authority_structure": authority_structure,
            "known_counts": known_counts,
            "variant_count": len(variants),
        }
        best_accuracy = -1.0
        best_names: list[str] = []
        fresh_passes = []
        for source in variants:
            data = source["_data"]
            digest = source["sha256"]
            if digest not in structure_cache:
                structure_cache[digest] = SCAN.structural_audit(task, source["_model"], data)
            structure = structure_cache[digest]
            cost = audited_cost(structure)
            row = {
                "task": task, "name": source["name"], "family": source["family"],
                "proof": source["proof"], "sha256": digest,
                "authority_cost": AUTHORITY_COSTS[task], "audited_cost": cost,
                "structure_pass": structure["pass"],
                "structure_reasons": structure["reasons"],
            }
            aggregate["variants"] += 1
            if not structure["pass"]:
                row["classification"] = "REJECT_STRUCTURE"
                candidate_rows.append(row)
                continue
            aggregate["structure_pass"] += 1
            if cost is None or cost >= AUTHORITY_COSTS[task]:
                row["classification"] = "REJECT_NOT_STRICT_LOWER_ACTUAL"
                candidate_rows.append(row)
                continue
            aggregate["strict_lower"] += 1
            try:
                runtime = SCAN.make_session(data, True, 1)
                known_base, _ = SCAN.evaluate_config(runtime, cases, None)
            except Exception as exc:  # noqa: BLE001
                known_base = {
                    "total": len(cases), "right": 0, "wrong": 0, "accuracy": 0.0,
                    "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}",
                    "nonfinite_cases": 0, "nonfinite_elements": 0,
                    "runtime_shape_mismatches": 0, "small_positive_elements_0_to_0_25": 0,
                    "sign_mismatch_cases_vs_disable_threads1": 0,
                    "sign_mismatch_cells_vs_disable_threads1": 0,
                }
            aggregate["known_base_case_executions"] += len(cases)
            row["known_disable_threads1"] = compact_runtime(known_base)
            accuracy = float(known_base.get("accuracy", 0.0))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_names = [source["name"]]
            elif accuracy == best_accuracy:
                best_names.append(source["name"])
            if not row_clean_exact(known_base):
                row["classification"] = "REJECT_KNOWN_BASE_NOT_EXACT_OR_UNSAFE"
                candidate_rows.append(row)
                continue
            aggregate["known_base_exact"] += 1
            known_four = SCAN.evaluate_four(data, cases)
            aggregate["known_four_case_config_executions"] += 4 * len(cases)
            row["known_four"] = {name: compact_runtime(value) for name, value in known_four.items()}
            if not four_exact(known_four):
                row["classification"] = "REJECT_KNOWN_FOUR_CONFIG"
                candidate_rows.append(row)
                continue
            official = SCAN.official_profile(task, source["_model"], source["name"])
            row["official_profile"] = official
            if official is None or int(official["cost"]) != cost or cost >= AUTHORITY_COSTS[task]:
                row["classification"] = "REJECT_OFFICIAL_ACTUAL_COST"
                candidate_rows.append(row)
                continue
            aggregate["known_four_exact"] += 1
            fresh_runs = []
            for seed in (294_000_000 + task, 294_100_000 + task):
                fresh, generation = SCAN.fresh_cases(task, seed, task_map)
                fresh_four = SCAN.evaluate_four(data, fresh)
                aggregate["fresh_case_config_executions"] += 4 * len(fresh)
                fresh_runs.append({
                    "seed": seed, "generation": generation,
                    "runtime": {name: compact_runtime(value) for name, value in fresh_four.items()},
                    "pass": four_exact(fresh_four),
                })
            fresh_pass = all(
                item["pass"] and item["generation"]["accepted"] == FRESH_PER_SEED
                and item["generation"]["generation_errors"] == 0
                and item["generation"]["conversion_skips"] == 0
                for item in fresh_runs
            )
            row["fresh"] = {"count_per_seed": FRESH_PER_SEED, "runs": fresh_runs, "pass": fresh_pass}
            if not fresh_pass:
                row["classification"] = "REJECT_FRESH_FOUR_NOT_EXACT_OR_UNSAFE"
                candidate_rows.append(row)
                continue
            row["classification"] = "PASS_EXACT_SAFE_STRICT_LOWER"
            candidate_rows.append(row)
            fresh_passes.append((cost, digest, source, row))
        if fresh_passes:
            _, _, winner, winner_row = sorted(fresh_passes, key=lambda item: (item[0], item[1]))[0]
            path = CANDIDATE_DIR / f"task{task:03d}_{winner['name']}_cost{winner_row['audited_cost']}.onnx"
            path.write_bytes(winner["_data"])
            winner_row["saved_path"] = rel(path)
            finalists.append({
                "task": task, "path": rel(path), "sha256": winner["sha256"],
                "cost": winner_row["audited_cost"], "authority_cost": AUTHORITY_COSTS[task],
                "name": winner["name"], "family": winner["family"],
            })
            task_row["decision"] = "PASS_SAFE_STRICT_LOWER"
            task_row["winner"] = finalists[-1]
        else:
            task_row["decision"] = "NO_SAFE_STRICT_LOWER_CANDIDATE"
        task_row["best_known_disable_threads1_accuracy"] = best_accuracy
        task_row["best_known_variant_names"] = best_names[:20]
        task_rows.append(task_row)
        print(json.dumps({
            "task": task, "decision": task_row["decision"],
            "variants": len(variants), "best_known": best_accuracy,
        }), flush=True)
    structure_rows = {
        digest: {
            "pass": row["pass"], "reasons": row["reasons"],
            "full_check": row["full_check"],
            "strict_shape_data_prop": row["strict_shape_data_prop"],
            "canonical_io": row["canonical_io"],
            "missing_node_outputs": row["missing_node_outputs"],
            "nonstatic_node_outputs": row["nonstatic_node_outputs"],
            "op_histogram": row["op_histogram"],
            "nonstandard_domains": row["nonstandard_domains"],
            "lookup_ops": row["lookup_ops"], "banned_ops": row["banned_ops"],
            "nested_graphs": row["nested_graphs"], "functions": row["functions"],
            "nonfinite_initializers": row["nonfinite_initializers"],
            "initializer_elements": row["initializer_elements"],
            "largest_initializer_elements": row["largest_initializer_elements"],
            "max_einsum_inputs": row["max_einsum_inputs"],
            "giant_einsum": row["giant_einsum"],
            "conv_bias_ub_findings": row["conv_bias_ub_findings"],
            "runtime_intermediate_trace": row["runtime_intermediate_trace"],
            "file_bytes": row["file_bytes"],
        }
        for digest, row in structure_cache.items()
    }
    classifications = Counter(row["classification"] for row in candidate_rows)
    payload = {
        "lane": "extra15_cost25_scan_294",
        "decision": "PASS_CANDIDATES_FOUND" if finalists else "NO_SAFE_STRICT_LOWER_CANDIDATE",
        "authority": {"zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256, "lb": 8010.03},
        "policy": {
            "tasks": list(TASKS), "fresh_per_seed": FRESH_PER_SEED,
            "fresh_seeds": ["294000000 + task", "294100000 + task"],
            "configs": [
                "ORT_DISABLE_ALL threads1", "ORT_DISABLE_ALL threads4",
                "ORT_ENABLE_ALL threads1", "ORT_ENABLE_ALL threads4",
            ],
            "exact_accuracy_required": 1.0,
            "strict_lower_actual_cost": True,
            "full_checker_strict_static_required": True,
            "lookup_giant_nonfinite_conv_ub_shape_cloak_fail_closed": True,
            "score25_task129_giant_49_input_einsum_excluded": True,
            "root_submission_all_scores_others71407_written": False,
            "kimi_used": False,
        },
        "task_rows": task_rows,
        "candidate_rows": candidate_rows,
        "structure_by_sha256": structure_rows,
        "classification_counts": dict(classifications),
        "aggregate": dict(aggregate),
        "reported_candidates": finalists,
        "elapsed_seconds": time.monotonic() - started,
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "reported": finalists,
        "classifications": dict(classifications), "aggregate": dict(aggregate),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

