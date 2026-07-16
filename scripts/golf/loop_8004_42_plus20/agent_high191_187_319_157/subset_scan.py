#!/usr/bin/env python3
"""Exhaustive exact CastLike/Identity subset scan on immutable 8009.46 members.

The transformations are all-input semantic identities:

* CastLike(x, y) -> Cast(x, to=dtype(y)); CastLike only observes y's type.
* Identity(x) -> x.
* Nodes and initializers made unreachable by those rewrites are removed.

Every structurally valid subset is measured through the competition's runtime
profile, not the advisory/static value_info profile.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline"
CANDIDATES = HERE / "candidates"
TASKS = (191, 187, 319)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HARVEST = load_module(
    "lane157_harvest",
    ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    """Return every statically known tensor dtype after strict inference."""
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    result = {item.name: int(item.data_type) for item in inferred.graph.initializer}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        if value.type.HasField("tensor_type"):
            result[value.name] = int(value.type.tensor_type.elem_type)
    return result


def rewrite_points(model: onnx.ModelProto) -> list[dict[str, object]]:
    types = tensor_types(model)
    points: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "CastLike" and len(node.input) == 2:
            reference = node.input[1]
            dtype = types.get(reference)
            if dtype:
                points.append(
                    {
                        "key": f"castlike_{index}",
                        "kind": "CastLike_to_Cast",
                        "node_index": index,
                        "reference": reference,
                        "to": dtype,
                    }
                )
        elif node.op_type == "Identity" and len(node.input) == 1:
            points.append(
                {
                    "key": f"identity_{index}",
                    "kind": "Identity_bypass",
                    "node_index": index,
                    "source": node.input[0],
                    "output": node.output[0],
                }
            )
    return points


def dce(model: onnx.ModelProto) -> dict[str, list[str]]:
    """Remove only nodes/initializers that cannot reach a graph output."""
    original_nodes = list(model.graph.node)
    live = {value.name for value in model.graph.output}
    keep_reversed: list[onnx.NodeProto] = []
    for node in reversed(original_nodes):
        if any(name and name in live for name in node.output):
            keep_reversed.append(node)
            live.update(name for name in node.input if name)
    keep_nodes = list(reversed(keep_reversed))
    removed_nodes = [
        node.name or (node.output[0] if node.output else f"index_{index}")
        for index, node in enumerate(original_nodes)
        if id(node) not in {id(item) for item in keep_nodes}
    ]
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    input_uses = {name for node in keep_nodes for name in node.input if name}
    keep_initializers = [
        item for item in model.graph.initializer if item.name in input_uses
    ]
    removed_initializers = [
        item.name for item in model.graph.initializer if item.name not in input_uses
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_initializers)

    existing = {value.name for value in model.graph.input}
    existing.update(value.name for value in model.graph.output)
    existing.update(item.name for item in model.graph.initializer)
    existing.update(name for node in model.graph.node for name in node.output if name)
    keep_vi = [item for item in model.graph.value_info if item.name in existing]
    removed_vi = [item.name for item in model.graph.value_info if item.name not in existing]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)
    return {
        "removed_nodes": removed_nodes,
        "removed_initializers": removed_initializers,
        "removed_value_info": removed_vi,
    }


def apply_subset(
    original: onnx.ModelProto,
    points: list[dict[str, object]],
    chosen: tuple[int, ...],
) -> tuple[onnx.ModelProto, dict[str, list[str]]]:
    model = copy.deepcopy(original)
    # CastLike replacements preserve output names, so do them before Identity
    # substitutions. Indexes still refer to the unchanged node list here.
    for point_index in chosen:
        point = points[point_index]
        if point["kind"] != "CastLike_to_Cast":
            continue
        index = int(point["node_index"])
        old = model.graph.node[index]
        replacement = helper.make_node(
            "Cast",
            [old.input[0]],
            list(old.output),
            name=old.name,
            to=int(point["to"]),
        )
        model.graph.node[index].CopyFrom(replacement)

    for point_index in chosen:
        point = points[point_index]
        if point["kind"] != "Identity_bypass":
            continue
        source, output = str(point["source"]), str(point["output"])
        for node in model.graph.node:
            for pos, name in enumerate(node.input):
                if name == output:
                    node.input[pos] = source
        for graph_output in model.graph.output:
            if graph_output.name == output:
                graph_output.name = source

    return model, dce(model)


def static_profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def structural(model: onnx.ModelProto) -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_checker=False, full_checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
    return result


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    baseline_audit = json.loads((HERE / "baseline_audit.json").read_text())
    output: dict[str, object] = {
        "authority": baseline_audit["authority_zip"],
        "authority_sha256": baseline_audit["authority_zip_sha256"],
        "proof": {
            "CastLike_to_Cast": "ONNX CastLike(x,y) casts x to y's element type; replacing it with Cast(x,to=statically inferred dtype(y)) is equal for every x and y value because y's values and shape are unobserved.",
            "Identity_bypass": "ONNX Identity(x)=x for every tensor value and shape.",
            "DCE": "A node/initializer unreachable from every graph output cannot affect any output.",
        },
        "tasks": {},
    }
    for task in TASKS:
        path = BASE / f"task{task:03d}.onnx"
        original = onnx.load(path)
        points = rewrite_points(original)
        base_actual = int(
            baseline_audit["tasks"][str(task)]["official_like_score"]["cost"]
        )
        with tempfile.TemporaryDirectory(prefix=f"lane157_base_{task}_") as work:
            static_path = Path(work) / path.name
            onnx.save(original, static_path)
            base_static = static_profile(static_path)
        task_rows: list[dict[str, object]] = []
        for size in range(1, len(points) + 1):
            for chosen in itertools.combinations(range(len(points)), size):
                model, cleanup = apply_subset(original, points, chosen)
                data = model.SerializeToString()
                row: dict[str, object] = {
                    "subset": [str(points[index]["key"]) for index in chosen],
                    "sha256": sha256(data),
                    "node_count": len(model.graph.node),
                    "initializer_count": len(model.graph.initializer),
                    "cleanup": cleanup,
                    "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
                }
                row.update(structural(model))
                if row.get("full_checker") and row.get("strict_data_prop"):
                    with tempfile.TemporaryDirectory(prefix=f"lane157_static_{task}_") as work:
                        candidate_path = Path(work) / f"task{task:03d}.onnx"
                        onnx.save(model, candidate_path)
                        try:
                            row["static_profile"] = static_profile(candidate_path)
                        except Exception as exc:  # noqa: BLE001
                            row["static_profile_error"] = f"{type(exc).__name__}: {exc}"
                    actual = HARVEST.actual_screen(data, task)
                    row["actual_cost"] = actual
                    row["actual_delta"] = None if actual is None else base_actual - actual
                    row["strict_lower_actual"] = actual is not None and actual < base_actual
                    if row["strict_lower_actual"]:
                        saved = CANDIDATES / (
                            f"task{task:03d}_actual{actual}_{row['sha256'][:12]}.onnx"
                        )
                        saved.write_bytes(data)
                        row["path"] = str(saved.relative_to(ROOT))
                else:
                    row["actual_cost"] = None
                    row["strict_lower_actual"] = False
                task_rows.append(row)
        output["tasks"][str(task)] = {
            "baseline_actual_cost": base_actual,
            "baseline_static_profile": base_static,
            "rewrite_points": points,
            "preexisting_prelu_count": sum(
                node.op_type == "PRelu" for node in original.graph.node
            ),
            "subsets_profiled": len(task_rows),
            "strict_lower_count": sum(
                bool(row.get("strict_lower_actual")) for row in task_rows
            ),
            "rows": task_rows,
        }
        (HERE / "subset_scan.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "task": task,
                    "points": len(points),
                    "subsets": len(task_rows),
                    "strict_lower": output["tasks"][str(task)]["strict_lower_count"],
                    "best": min(
                        (
                            int(row["actual_cost"])
                            for row in task_rows
                            if row.get("actual_cost") is not None
                        ),
                        default=None,
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
