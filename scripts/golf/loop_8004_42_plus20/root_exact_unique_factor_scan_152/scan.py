#!/usr/bin/env python3
"""Factor repeated initializer slices directly inside Einsum contractions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import string
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
LABELS = string.ascii_lowercase + string.ascii_uppercase


def equation(node: onnx.NodeProto) -> str:
    attr = next((item for item in node.attribute if item.name == "equation"), None)
    if attr is None:
        raise ValueError("Einsum lacks equation")
    return attr.s.decode("ascii")


def set_equation(node: onnx.NodeProto, value: str) -> None:
    attr = next(item for item in node.attribute if item.name == "equation")
    attr.s = value.encode("ascii")


def unique_factor(array: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    moved = np.moveaxis(array, axis, 0)
    keys: dict[bytes, int] = {}
    unique: list[np.ndarray] = []
    index: list[int] = []
    for item in moved:
        key = np.ascontiguousarray(item).tobytes()
        if key not in keys:
            keys[key] = len(unique)
            unique.append(np.array(item, copy=True))
        index.append(keys[key])
    table = np.stack(unique, axis=0)
    selector = np.zeros((moved.shape[0], len(unique)), dtype=array.dtype)
    selector[np.arange(moved.shape[0]), np.asarray(index)] = np.asarray(1, dtype=array.dtype)
    return selector, table


def plans(model: onnx.ModelProto) -> list[dict]:
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            locations[name].append((node_index, input_index))
    result: list[dict] = []
    for init in model.graph.initializer:
        locs = locations.get(init.name, [])
        if not locs or any(model.graph.node[n].op_type != "Einsum" for n, _ in locs):
            continue
        array = np.asarray(numpy_helper.to_array(init))
        if array.ndim < 2 or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            continue
        if any("..." in equation(model.graph.node[n]) for n, _ in locs):
            continue
        for axis in range(array.ndim):
            selector, table = unique_factor(array, axis)
            factor_params = int(selector.size + table.size)
            saving = int(array.size - factor_params)
            if saving <= 0:
                continue
            valid = True
            for node_index, input_index in locs:
                lhs = equation(model.graph.node[node_index]).split("->", 1)[0].split(",")
                if input_index >= len(lhs) or len(lhs[input_index]) != array.ndim:
                    valid = False
                    break
            if valid:
                result.append({
                    "initializer": init.name, "axis": axis,
                    "shape": list(array.shape), "unique_slices": int(table.shape[0]),
                    "original_params": int(array.size), "factor_params": factor_params,
                    "parameter_saving": saving, "locations": locs,
                    "selector": selector, "table": table,
                })
    return result


def build(source: onnx.ModelProto, plan: dict) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = plan["initializer"]
    axis = int(plan["axis"])
    selector_name = f"{name}__axis{axis}_selector"
    table_name = f"{name}__axis{axis}_unique"
    for node_index, _ in plan["locations"]:
        node = model.graph.node[node_index]
        eq = equation(node)
        lhs_text, rhs = eq.split("->", 1) if "->" in eq else (eq, "")
        terms = lhs_text.split(",")
        used = set("".join(terms) + rhs)
        input_index = 0
        while input_index < len(node.input):
            if node.input[input_index] != name:
                input_index += 1
                continue
            label = next((item for item in LABELS if item not in used), None)
            if label is None:
                raise ValueError("no free Einsum label")
            used.add(label)
            original_term = terms[input_index]
            selector_term = original_term[axis] + label
            table_term = label + original_term[:axis] + original_term[axis + 1:]
            inputs = list(node.input)
            inputs[input_index:input_index + 1] = [selector_name, table_name]
            terms[input_index:input_index + 1] = [selector_term, table_term]
            del node.input[:]
            node.input.extend(inputs)
            input_index += 2
        set_equation(node, ",".join(terms) + ("->" + rhs if "->" in eq else ""))
    keep = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    model.graph.initializer.append(numpy_helper.from_array(plan["selector"], selector_name))
    model.graph.initializer.append(numpy_helper.from_array(plan["table"], table_name))
    return model


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"uniqfactor152_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            baseline = None
            for ordinal, plan in enumerate(plans(model), 1):
                serial = {key: value for key, value in plan.items() if key not in {"selector", "table"}}
                try:
                    candidate = build(model, plan)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row = {
                        "task": task, "ordinal": ordinal, **serial,
                        "baseline": baseline, "candidate": current,
                        "strict_lower": current["cost"] < baseline["cost"],
                        "projected_gain": math.log(baseline["cost"] / current["cost"])
                        if current["cost"] > 0 else None,
                    }
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{ordinal:02d}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path)
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    rows.append(row)
                except Exception as exc:
                    errors.append({"task": task, "ordinal": ordinal, **serial, "error": f"{type(exc).__name__}: {exc}"})
    result = {"authority": str(AUTHORITY), "rows": rows, "errors": errors}
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row["strict_lower"]],
        "errors": errors,
    }, indent=2))


if __name__ == "__main__":
    main()
