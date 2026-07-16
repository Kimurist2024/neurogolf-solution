#!/usr/bin/env python3
"""Replace unsigned threshold comparisons with exact Cast-to-bool forms."""

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
UNSIGNED = {
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}


def metadata(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    result: dict[str, int] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if value.type.HasField("tensor_type"):
            result[value.name] = value.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        result[init.name] = init.data_type
    return result


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"unsignedcmp253_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def drop_dead(model: onnx.ModelProto) -> list[str]:
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


def scalar_int(value: np.ndarray) -> int | None:
    if value.size != 1 or value.dtype.kind not in "iub":
        return None
    return int(value.reshape(-1)[0])


def normalize(op: str, const_left: bool) -> str:
    if not const_left:
        return op
    return {
        "Greater": "Less",
        "GreaterOrEqual": "LessOrEqual",
        "Less": "Greater",
        "LessOrEqual": "GreaterOrEqual",
        "Equal": "Equal",
    }[op]


def form(op: str, threshold: int) -> str | None:
    """Comparison of unsigned x against threshold, normalized as x OP k."""
    if (op, threshold) in {("Greater", 0), ("GreaterOrEqual", 1)}:
        return "cast"
    if (op, threshold) in {
        ("Equal", 0),
        ("Less", 1),
        ("LessOrEqual", 0),
    }:
        return "not_cast"
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    combined_rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            try:
                types = metadata(model)
            except Exception:
                types = {}
            baseline = None
            exact_cast_sites: list[tuple[int, str, str]] = []
            for index, node in enumerate(model.graph.node):
                if node.op_type not in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}:
                    continue
                if len(node.input) != 2:
                    continue
                for const_index in (0, 1):
                    threshold_array = values.get(node.input[const_index])
                    if threshold_array is None:
                        continue
                    threshold = scalar_int(threshold_array)
                    data_name = node.input[1 - const_index]
                    if threshold is None or types.get(data_name) not in UNSIGNED:
                        continue
                    normalized = normalize(node.op_type, const_index == 0)
                    rewrite = form(normalized, threshold)
                    if rewrite is None:
                        continue
                    if rewrite == "cast":
                        exact_cast_sites.append((index, data_name, f"x {normalized} {threshold}"))
                    candidate = copy.deepcopy(model)
                    original = candidate.graph.node[index]
                    cast = helper.make_node(
                        "Cast", [data_name], list(original.output),
                        name=original.name, to=TensorProto.BOOL,
                    )
                    if rewrite == "cast":
                        replacements = [cast]
                    else:
                        cast_output = f"{original.output[0]}__nonzero"
                        cast.output[0] = cast_output
                        replacements = [cast, helper.make_node("Not", [cast_output], list(original.output))]
                    nodes = list(candidate.graph.node)
                    nodes[index:index + 1] = replacements
                    del candidate.graph.node[:]
                    candidate.graph.node.extend(nodes)
                    dropped = drop_dead(candidate)
                    record: dict = {
                        "task": task,
                        "node_index": index,
                        "source_op": node.op_type,
                        "normalized": f"x {normalized} {threshold}",
                        "rewrite": rewrite,
                        "data_type": types[data_name],
                        "dropped_initializers": dropped,
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
                            path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                            onnx.save(candidate, path)
                            record["path"] = str(path.relative_to(REPO))
                            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(record)
                    break
            if exact_cast_sites:
                combined = copy.deepcopy(model)
                descriptions = []
                for index, data_name, description in exact_cast_sites:
                    original = combined.graph.node[index]
                    original.CopyFrom(helper.make_node(
                        "Cast", [data_name], list(original.output),
                        name=original.name, to=TensorProto.BOOL,
                    ))
                    descriptions.append({"node_index": index, "identity": description})
                dropped = drop_dead(combined)
                record = {
                    "task": task,
                    "kind": "combined_cast",
                    "sites": descriptions,
                    "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(combined, full_check=True)
                    onnx.shape_inference.infer_shapes(combined, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(combined, task)
                    record["baseline"] = baseline
                    record["candidate"] = current
                    record["strict_lower"] = current["cost"] < baseline["cost"]
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_combined_cast.onnx"
                        onnx.save(combined, path)
                        record["path"] = str(path.relative_to(REPO))
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                combined_rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "identities": len(rows),
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
        "combined_strict_lower": sum(bool(row.get("strict_lower")) for row in combined_rows),
        "combined_rows": combined_rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("tasks", "identities", "strict_lower", "combined_strict_lower")}, indent=2))


if __name__ == "__main__":
    main()
