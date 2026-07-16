#!/usr/bin/env python3
"""Find exact Boolean neutral-constant rewrites in the 8009.46 archive.

The replacements preserve the original node output name and inferred shape.
Only input-independent Boolean identities are emitted; lower candidates still
need runtime-shape and corpus auditing before admission.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"boolneutral252_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def tensor_metadata(model: onnx.ModelProto) -> tuple[dict[str, int], dict[str, tuple[int, ...] | None]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    result: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...] | None] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if value.type.HasField("tensor_type"):
            result[value.name] = value.type.tensor_type.elem_type
            dims = value.type.tensor_type.shape.dim
            if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
                shapes[value.name] = tuple(dim.dim_value for dim in dims)
            else:
                shapes[value.name] = None
    for init in inferred.graph.initializer:
        result[init.name] = init.data_type
        shapes[init.name] = tuple(init.dims)
    return result, shapes


def dead_initializers(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_inputs = {item.name for item in model.graph.input}
    dropped = [
        item.name for item in model.graph.initializer
        if uses[item.name] == 0 and item.name not in graph_inputs
    ]
    keep = [item for item in model.graph.initializer if item.name not in dropped]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def uniform_bool(value: np.ndarray) -> bool | None:
    if value.dtype != np.bool_ or value.size == 0:
        return None
    first = bool(value.reshape(-1)[0])
    return first if bool(np.all(value == first)) else None


def replacement_for(
    node: onnx.NodeProto,
    values: dict[str, np.ndarray],
    types: dict[str, int],
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[list[onnx.NodeProto], onnx.TensorProto | None, str] | None:
    """Return a one-node exact rewrite, or None.

    Scalar Boolean constants broadcast without changing the other input's
    shape. Equal on Boolean tensors is also exact for both constant values.
    """
    if node.op_type not in {"And", "Or", "Xor", "Equal"} or len(node.input) != 2:
        return None
    for const_index in (0, 1):
        const_name = node.input[const_index]
        other_name = node.input[1 - const_index]
        value = values.get(const_name)
        if value is None:
            continue
        bit = uniform_bool(value)
        if bit is None or types.get(other_name) != TensorProto.BOOL:
            continue
        identity = (
            (node.op_type == "And" and bit)
            or (node.op_type in {"Or", "Xor"} and not bit)
            or (node.op_type == "Equal" and bit)
        )
        invert = (
            (node.op_type == "Xor" and bit)
            or (node.op_type == "Equal" and not bit)
        )
        output_shape = shapes.get(node.output[0])
        other_shape = shapes.get(other_name)
        broadcast = other_shape != output_shape
        shape_init = None
        source = other_name
        prefix: list[onnx.NodeProto] = []
        if broadcast:
            if output_shape is None:
                continue
            shape_name = f"{node.output[0]}__shape"
            shape_init = numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), name=shape_name)
            if invert:
                source = f"{node.output[0]}__expanded"
                prefix.append(helper.make_node("Expand", [other_name, shape_name], [source]))
            else:
                repl = helper.make_node("Expand", [other_name, shape_name], list(node.output), name=node.name)
                return [repl], shape_init, f"{node.op_type}(bool,neutral)->Expand"
        if identity:
            repl = helper.make_node("Identity", [source], list(node.output), name=node.name)
            return [*prefix, repl], shape_init, f"{node.op_type}(bool,neutral)->Identity"
        if invert:
            repl = helper.make_node("Not", [source], list(node.output), name=node.name)
            return [*prefix, repl], shape_init, f"{node.op_type}(bool,inverting)->Not"
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            try:
                types, shapes = tensor_metadata(model)
            except Exception:
                types = {}
                shapes = {}
            baseline = None
            for index, node in enumerate(model.graph.node):
                census[node.op_type] += 1
                rewrite = replacement_for(node, values, types, shapes)
                if rewrite is None:
                    continue
                candidate = copy.deepcopy(model)
                replacements, shape_init, description = rewrite
                nodes = list(candidate.graph.node)
                nodes[index:index + 1] = replacements
                del candidate.graph.node[:]
                candidate.graph.node.extend(nodes)
                if shape_init is not None:
                    candidate.graph.initializer.append(shape_init)
                dropped = dead_initializers(candidate)
                record: dict = {
                    "task": task,
                    "node_index": index,
                    "source_op": node.op_type,
                    "rewrite": description,
                    "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(
                        candidate, strict_mode=True, data_prop=True
                    )
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    record["baseline"] = baseline
                    record["candidate"] = current
                    record["strict_lower"] = current["cost"] < baseline["cost"]
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                        onnx.save(candidate, path)
                        record["path"] = str(path.relative_to(REPO))
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "operator_census": dict(sorted(census.items())),
        "identities": len(rows),
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("tasks", "identities", "strict_lower")}, indent=2))


if __name__ == "__main__":
    main()
