#!/usr/bin/env python3
"""Remove globally redundant all-one operands from Einsum nodes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            value = helper.get_attribute_value(attr)
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def set_equation(node: onnx.NodeProto, value: str) -> None:
    keep = [attr for attr in node.attribute if attr.name != "equation"]
    del node.attribute[:]
    node.attribute.extend(keep)
    node.attribute.append(helper.make_attribute("equation", value))


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"einsumunit256_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def uses(model: onnx.ModelProto) -> dict[str, list[tuple[int, int]]]:
    result: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for position, name in enumerate(node.input):
            if name:
                result[name].append((node_index, position))
    return dict(result)


def rewrite_node_for_initializer(
    node: onnx.NodeProto,
    initializer_name: str,
    array: np.ndarray,
) -> tuple[list[str], str] | None:
    expression = equation(node)
    if node.op_type != "Einsum" or expression is None or "->" not in expression or "..." in expression:
        return None
    left, output = expression.split("->", 1)
    terms = left.split(",")
    if len(terms) != len(node.input):
        return None
    removed = [position for position, name in enumerate(node.input) if name == initializer_name]
    if not removed or len(removed) == len(terms):
        return None
    remaining = [position for position in range(len(terms)) if position not in removed]
    remaining_labels = set("".join(terms[position] for position in remaining))
    removed_labels = set("".join(terms[position] for position in removed))
    # Einsum output labels must still be supplied by a remaining operand.
    if any(label not in remaining_labels for label in output):
        return None
    # A removed-only contracted mode sums the ones and contributes its extent.
    # It is neutral only when that extent is exactly one.
    label_dims: dict[str, int] = {}
    for position in removed:
        term = terms[position]
        if len(term) != array.ndim:
            return None
        for label, dim in zip(term, array.shape):
            prior = label_dims.setdefault(label, int(dim))
            if prior != int(dim):
                return None
    if any(label_dims[label] != 1 for label in removed_labels - remaining_labels):
        return None
    new_terms = [terms[position] for position in remaining]
    new_inputs = [node.input[position] for position in remaining]
    return new_inputs, ",".join(new_terms) + "->" + output


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = {"einsum_nodes": 0, "all_one_initializers": 0, "globally_redundant": 0}
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            census["einsum_nodes"] += sum(node.op_type == "Einsum" for node in model.graph.node)
            arrays = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            all_one = {
                name: array for name, array in arrays.items()
                if array.size and np.all(np.isfinite(array)) and np.all(array == 1)
            }
            census["all_one_initializers"] += len(all_one)
            occurrence = uses(model)
            baseline = None
            for name, array in all_one.items():
                positions = occurrence.get(name, [])
                if not positions:
                    continue
                node_indices = sorted({node_index for node_index, _ in positions})
                plans: dict[int, tuple[list[str], str]] = {}
                eligible = True
                for node_index in node_indices:
                    plan = rewrite_node_for_initializer(model.graph.node[node_index], name, array)
                    if plan is None:
                        eligible = False
                        break
                    plans[node_index] = plan
                if not eligible:
                    continue
                census["globally_redundant"] += 1
                candidate = copy.deepcopy(model)
                for node_index, (new_inputs, new_equation) in plans.items():
                    node = candidate.graph.node[node_index]
                    del node.input[:]
                    node.input.extend(new_inputs)
                    set_equation(node, new_equation)
                keep = [item for item in candidate.graph.initializer if item.name != name]
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(keep)
                record: dict = {
                    "task": task,
                    "initializer": name,
                    "shape": list(array.shape),
                    "elements": int(array.size),
                    "uses": positions,
                    "plans": {
                        str(index): {"inputs": value[0], "equation": value[1]}
                        for index, value in plans.items()
                    },
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    record["baseline"] = baseline
                    record["candidate"] = current
                    record["strict_lower"] = current["cost"] < baseline["cost"]
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{name}.onnx"
                        onnx.save(candidate, path)
                        record["path"] = str(path.relative_to(REPO))
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "census": census,
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"census": census, "strict_lower": result["strict_lower"]}, indent=2))


if __name__ == "__main__":
    main()
