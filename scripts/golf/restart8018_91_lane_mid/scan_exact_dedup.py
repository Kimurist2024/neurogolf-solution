#!/usr/bin/env python3
"""Exact initializer/Constant deduplication and neutral-op scan for 8018.91."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8018.91.zip"
AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def import_base():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("restart8018_mid_exact_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.SUPPORT.POLICY_THRESHOLD = 1.0
    module.SUPPORT.FRESH_PER_SEED = 2_000
    return module


def tensor_key(tensor: onnx.TensorProto) -> tuple[Any, ...] | None:
    try:
        array = np.asarray(numpy_helper.to_array(tensor))
    except Exception:
        return None
    if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
        return None
    return (array.dtype.str, tuple(array.shape), array.tobytes(order="C"))


def constant_tensor(node: onnx.NodeProto) -> onnx.TensorProto | None:
    if node.op_type != "Constant" or len(node.output) != 1:
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def replace_name(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def prune_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def prune_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    outputs = {value.name for value in model.graph.output}
    removed = [item.name for item in model.graph.initializer if item.name not in used | outputs]
    if removed:
        kept = [item for item in model.graph.initializer if item.name not in set(removed)]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        prune_value_info(model, set(removed))
    return removed


def serialize(model: onnx.ModelProto, family: str, detail: dict[str, Any]):
    prune_unused_initializers(model)
    data = model.SerializeToString()
    return data, {"family": family, "detail": detail,
                  "sha256": hashlib.sha256(data).hexdigest()}


def variants(base: onnx.ModelProto) -> Iterable[tuple[bytes, dict[str, Any]]]:
    graph_outputs = {value.name for value in base.graph.output}
    graph_inputs = {value.name for value in base.graph.input}

    # All exact duplicate initializers can be globally aliased at once.
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for item in base.graph.initializer:
        key = tensor_key(item)
        if key is not None:
            groups[key].append(item.name)
    duplicate_groups = [names for names in groups.values() if len(names) > 1]
    if duplicate_groups:
        model = copy.deepcopy(base)
        removed: set[str] = set()
        aliases = []
        for names in duplicate_groups:
            canonical = names[0]
            for name in names[1:]:
                if name in graph_inputs | graph_outputs:
                    continue
                replace_name(model, name, canonical)
                removed.add(name)
                aliases.append([name, canonical])
        if removed:
            kept = [item for item in model.graph.initializer if item.name not in removed]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept)
            prune_value_info(model, removed)
            yield serialize(model, "initializer_exact_dedup", {"aliases": aliases})

    # Remove all unused parameters; semantically exact and no added outputs.
    model = copy.deepcopy(base)
    removed = prune_unused_initializers(model)
    if removed:
        yield serialize(model, "unused_initializer_prune", {"removed": removed})

    # Alias duplicate Constant nodes and Constants equal to an initializer.
    init_by_key: dict[tuple[Any, ...], str] = {}
    for item in base.graph.initializer:
        key = tensor_key(item)
        if key is not None:
            init_by_key.setdefault(key, item.name)
    const_by_key: dict[tuple[Any, ...], tuple[int, str]] = {}
    for index, node in enumerate(base.graph.node):
        tensor = constant_tensor(node)
        key = tensor_key(tensor) if tensor is not None else None
        if key is None or node.output[0] in graph_outputs:
            continue
        source = init_by_key.get(key)
        if source is None and key in const_by_key:
            source = const_by_key[key][1]
        if source is None:
            const_by_key[key] = (index, node.output[0])
            continue
        model = copy.deepcopy(base)
        old = model.graph.node[index].output[0]
        replace_name(model, old, source)
        del model.graph.node[index]
        prune_value_info(model, {old})
        yield serialize(model, "constant_exact_alias", {
            "node_index": index, "removed_output": old, "source": source,
        })

    # Materialize Constant tensor attributes as graph initializers.  The
    # parameter count is unchanged, while a non-output Constant node's tensor
    # ceases to be charged as intermediate memory.  This is an exact rewrite.
    convertible = []
    for index, node in enumerate(base.graph.node):
        tensor = constant_tensor(node)
        if tensor is None or node.output[0] in graph_outputs:
            continue
        if tensor_key(tensor) is None:
            continue
        convertible.append((index, node.output[0], tensor))
    if convertible:
        model = copy.deepcopy(base)
        remove_indices = {index for index, _name, _tensor in convertible}
        kept_nodes = [node for index, node in enumerate(model.graph.node)
                      if index not in remove_indices]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        for _index, name, tensor in convertible:
            item = copy.deepcopy(tensor)
            item.name = name
            model.graph.initializer.append(item)
        yield serialize(model, "constant_to_initializer_all", {
            "converted": [{"node_index": index, "name": name,
                            "elements": int(np.asarray(numpy_helper.to_array(tensor)).size)}
                           for index, name, tensor in convertible],
        })

    # Neutral arithmetic and same-shape layout nodes are exact aliases.
    arrays = {item.name: np.asarray(numpy_helper.to_array(item))
              for item in base.graph.initializer}
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(base), strict_mode=True, data_prop=True
        )
    except Exception:
        inferred = None
    desc: dict[str, tuple[int, tuple[int, ...]]] = {}
    if inferred is not None:
        for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
            tt = value.type.tensor_type
            if not tt.HasField("shape") or any(not d.HasField("dim_value") for d in tt.shape.dim):
                continue
            desc[value.name] = (int(tt.elem_type), tuple(int(d.dim_value) for d in tt.shape.dim))

    # Fold Shape/Size from statically inferred inputs into int64 initializers.
    # Initializers are charged per element, whereas the node outputs are
    # charged in bytes, so these exact rewrites normally save 7 per element.
    for index, node in enumerate(base.graph.node):
        if len(node.output) != 1 or node.output[0] in graph_outputs or not node.input:
            continue
        source_desc = desc.get(node.input[0])
        if source_desc is None:
            continue
        _dtype, source_shape = source_desc
        folded: np.ndarray | None = None
        reason = ""
        if node.op_type == "Shape":
            start, end = 0, len(source_shape)
            for attr in node.attribute:
                if attr.name == "start":
                    start = int(helper.get_attribute_value(attr))
                elif attr.name == "end":
                    end = int(helper.get_attribute_value(attr))
            folded = np.asarray(source_shape[slice(start, end)], dtype=np.int64)
            reason = "static_shape"
        elif node.op_type == "Size":
            folded = np.asarray(math.prod(source_shape), dtype=np.int64)
            reason = "static_size"
        if folded is None:
            continue
        model = copy.deepcopy(base)
        target = model.graph.node[index].output[0]
        del model.graph.node[index]
        model.graph.initializer.append(numpy_helper.from_array(folded, name=target))
        yield serialize(model, "static_shape_fold", {
            "node_index": index, "op": node.op_type, "reason": reason,
            "source": node.input[0], "target": target,
            "value": folded.tolist(),
        })

    # Fold ConstantOfShape when its requested shape is already an initializer.
    for index, node in enumerate(base.graph.node):
        if node.op_type != "ConstantOfShape" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.output[0] in graph_outputs or node.input[0] not in arrays:
            continue
        shape = np.asarray(arrays[node.input[0]], dtype=np.int64).reshape(-1)
        if shape.size == 0 or np.any(shape <= 0) or int(np.prod(shape)) > 100_000:
            continue
        value = np.asarray(0.0, dtype=np.float32)
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                value = np.asarray(numpy_helper.to_array(attr.t)).reshape(-1)[0]
        folded = np.full(tuple(int(x) for x in shape), value, dtype=np.asarray(value).dtype)
        model = copy.deepcopy(base)
        target = model.graph.node[index].output[0]
        del model.graph.node[index]
        model.graph.initializer.append(numpy_helper.from_array(folded, name=target))
        yield serialize(model, "constant_of_shape_fold", {
            "node_index": index, "shape_input": node.input[0], "target": target,
            "shape": shape.tolist(), "elements": int(folded.size),
        })
    for index, node in enumerate(base.graph.node):
        if len(node.output) != 1 or not node.output[0] or not node.input:
            continue
        source = node.input[0]
        target = node.output[0]
        exact = False
        reason = ""
        if node.op_type in {"Identity", "Dropout"}:
            exact, reason = True, node.op_type
        elif node.op_type == "Cast" and desc.get(source) == desc.get(target):
            exact, reason = True, "same_dtype_cast"
        elif node.op_type == "Reshape" and desc.get(source) == desc.get(target):
            exact, reason = True, "same_static_shape_reshape"
        elif node.op_type in {"Add", "Sub", "Mul", "Div"} and len(node.input) == 2:
            const_slot = 1 if node.input[1] in arrays else (0 if node.input[0] in arrays else -1)
            if const_slot >= 0:
                value = arrays[node.input[const_slot]]
                dynamic = node.input[1 - const_slot]
                if node.op_type == "Add" and np.all(value == 0):
                    source, exact, reason = dynamic, True, "add_zero"
                elif node.op_type == "Sub" and const_slot == 1 and np.all(value == 0):
                    source, exact, reason = dynamic, True, "sub_zero"
                elif node.op_type == "Mul" and np.all(value == 1):
                    source, exact, reason = dynamic, True, "mul_one"
                elif node.op_type == "Div" and const_slot == 1 and np.all(value == 1):
                    source, exact, reason = dynamic, True, "div_one"
        if not exact:
            continue
        model = copy.deepcopy(base)
        if target in graph_outputs:
            model.graph.node[index].CopyFrom(helper.make_node(
                "Identity", [source], [target], name=f"exact_alias_{index}"
            ))
        else:
            del model.graph.node[index]
            replace_name(model, target, source)
            prune_value_info(model, {target})
        yield serialize(model, "neutral_node_alias", {
            "node_index": index, "op": node.op_type, "reason": reason,
            "source": source, "target": target,
        })


def exact_runtime(module, row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def main() -> int:
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    module = import_base()
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        tasks = [int(row["task"].removeprefix("task")) for row in csv.DictReader(handle)
                 if 250 <= int(row["cost"]) <= 399
                 and int(row["task"].removeprefix("task")) not in EXCLUDED]
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    report: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "scope": "ledger cost 250..399 excluding maintained private-zero/unsound catalogue",
        "tasks": [], "finalists": [], "counters": {},
        "policy": "gold exact + strict/static + stable margin + fresh 2000x2 exact",
    }
    counters: Counter[str] = Counter()
    seen: set[tuple[int, str]] = set()
    candidates_dir = HERE / "candidates_exact"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            raw_base = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(raw_base)
            cases, counts = module.SUPPORT.known_cases(task)
            base_profile = module.POLICY.fast_profile(module.SUPPORT, task, base, cases[0])
            if base_profile is None:
                raise RuntimeError(f"task{task:03d}: authority profile failed")
            task_row = {"task": task, "authority_cost": int(base_profile["cost"]),
                        "known_counts": counts, "attempts": []}
            for data, meta in variants(base):
                key = hashlib.sha256(data).hexdigest()
                if (task, key) in seen or data == raw_base:
                    counters["duplicate"] += 1
                    continue
                seen.add((task, key))
                counters["variants"] += 1
                attempt = {**meta}
                try:
                    model = onnx.load_model_from_string(data)
                    reasons = module.quick_preflight(model)
                except Exception as exc:
                    reasons = [f"preflight:{type(exc).__name__}:{exc}"]
                attempt["preflight_reasons"] = reasons
                if reasons:
                    counters["preflight_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                profile = module.POLICY.fast_profile(module.SUPPORT, task, model, cases[0])
                attempt["profile"] = profile
                if profile is None or int(profile["cost"]) >= int(base_profile["cost"]):
                    counters["cost_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                known_raw = module.SUPPORT.evaluate_four(data, cases)
                attempt["known_four"] = {name: module.compact_runtime(row)
                                          for name, row in known_raw.items()}
                if not all(exact_runtime(module, row) for row in known_raw.values()):
                    counters["known_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                structure = module.POLICY.structure_audit(module.SUPPORT, task, model, data)
                attempt["structure"] = structure
                if not structure["pass"]:
                    counters["structure_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                fresh_runs = []
                fresh_pass = True
                for seed in (818_910_000 + task, 818_920_000 + task):
                    fresh_cases, generation = module.SUPPORT.fresh_cases(task, seed, task_map)
                    runtime = module.SUPPORT.evaluate_four(data, fresh_cases)
                    passed = len(fresh_cases) >= 2_000 and all(
                        exact_runtime(module, row) for row in runtime.values()
                    )
                    fresh_runs.append({
                        "seed": seed, "generation": generation,
                        "case_count": len(fresh_cases),
                        "runtime": {name: module.compact_runtime(row)
                                    for name, row in runtime.items()},
                        "pass": passed,
                    })
                    fresh_pass &= passed
                attempt["fresh"] = fresh_runs
                if not fresh_pass:
                    counters["fresh_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                path = candidates_dir / (
                    f"task{task:03d}_{meta['family']}_cost{profile['cost']}_{key[:12]}.onnx"
                )
                path.write_bytes(data)
                check = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
                     "--task", str(task), "--onnx", str(path), "--timeout", "90",
                     "--label", "restart8018_exact"],
                    cwd=ROOT, capture_output=True, text=True,
                )
                try:
                    verified = json.loads(check.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    verified = {"ok": False, "reason": "unparseable"}
                attempt["nonmutating_official_gold"] = {
                    "returncode": check.returncode, "result": verified,
                    "output": (check.stdout + check.stderr)[-4000:],
                }
                if not (check.returncode == 0 and verified.get("ok") is True
                        and verified.get("correct") is True
                        and int(verified.get("cost", -1)) == int(profile["cost"])):
                    counters["official_reject"] += 1
                    task_row["attempts"].append(attempt)
                    continue
                attempt["path"] = str(path.relative_to(ROOT))
                attempt["gain"] = math.log(int(base_profile["cost"]) / int(profile["cost"]))
                report["finalists"].append({"task": task, **attempt})
                counters["finalists"] += 1
                task_row["attempts"].append(attempt)
                print(json.dumps({"STRICT_WINNER": task, "cost": profile["cost"],
                                  "gain": attempt["gain"], "path": attempt["path"]}), flush=True)
            report["tasks"].append(task_row)
            print(json.dumps({"task": task, "attempts": len(task_row["attempts"]),
                              "finalists": sum(r["task"] == task for r in report["finalists"])}),
                  flush=True)
    report["counters"] = dict(counters)
    (HERE / "exact_dedup_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
