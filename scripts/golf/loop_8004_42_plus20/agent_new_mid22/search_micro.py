#!/usr/bin/env python3
"""Conservative exact-micro search over the eight latest baseline members.

Only algebraically exact graph simplifications are attempted: collapse repeated
initializer axes, alias exact duplicate constants, remove optional zero biases,
remove optional zero Pad values, and eliminate duplicate/identity nodes.  Every
emitted probe is checked before it can be considered a candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = HERE / "current"
TASKS = (123, 316, 212, 301, 55, 86, 163, 206)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
from golf.rank_dir import cost_of  # noqa: E402
from audit_lane import known_dual, make_session, structure  # noqa: E402
from lib import scoring  # noqa: E402


def nparams(model: onnx.ModelProto) -> int:
    return sum(max(1, math.prod(init.dims)) for init in model.graph.initializer)


def sha_model(model: onnx.ModelProto) -> str:
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    for index, init in enumerate(candidate.graph.initializer):
        if init.name == name:
            candidate.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array, name=name)
            )
            return candidate
    raise KeyError(name)


def repeated_axis_probes(model: onnx.ModelProto) -> Iterable[tuple[str, onnx.ModelProto]]:
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        if array.ndim == 0:
            continue
        repeated_axes: list[int] = []
        for axis, size in enumerate(array.shape):
            if size <= 1:
                continue
            first = np.take(array, [0], axis=axis)
            if np.array_equal(array, np.repeat(first, size, axis=axis)):
                repeated_axes.append(axis)
                yield (
                    f"collapse_{init.name}_axis{axis}",
                    replace_initializer(model, init.name, first),
                )
        if len(repeated_axes) > 1:
            reduced = array
            for axis in repeated_axes:
                reduced = np.take(reduced, [0], axis=axis)
            yield (
                f"collapse_{init.name}_axes{'_'.join(map(str, repeated_axes))}",
                replace_initializer(model, init.name, reduced),
            )


def alias_duplicate_probe(model: onnx.ModelProto) -> Iterable[tuple[str, onnx.ModelProto]]:
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    names = list(arrays)
    for i, keep in enumerate(names):
        for drop in names[i + 1 :]:
            left, right = arrays[keep], arrays[drop]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                continue
            candidate = copy.deepcopy(model)
            for node in candidate.graph.node:
                for index, value in enumerate(node.input):
                    if value == drop:
                        node.input[index] = keep
            kept = [init for init in candidate.graph.initializer if init.name != drop]
            del candidate.graph.initializer[:]
            candidate.graph.initializer.extend(kept)
            yield f"alias_{drop}_to_{keep}", candidate


def optional_zero_probes(model: onnx.ModelProto) -> Iterable[tuple[str, onnx.ModelProto]]:
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    uses = {
        name: sum(name in node.input for node in model.graph.node)
        for name in arrays
    }
    for node_index, node in enumerate(model.graph.node):
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3:
            name = node.input[2]
            if name in arrays and np.count_nonzero(arrays[name]) == 0:
                candidate = copy.deepcopy(model)
                candidate.graph.node[node_index].input.pop()
                if uses[name] == 1:
                    kept = [init for init in candidate.graph.initializer if init.name != name]
                    del candidate.graph.initializer[:]
                    candidate.graph.initializer.extend(kept)
                yield f"drop_zero_bias_node{node_index}_{name}", candidate
        if node.op_type == "Pad" and len(node.input) >= 3:
            name = node.input[2]
            if name in arrays and arrays[name].size == 1 and float(arrays[name].item()) == 0.0:
                candidate = copy.deepcopy(model)
                candidate.graph.node[node_index].input[2] = ""
                if uses[name] == 1:
                    kept = [init for init in candidate.graph.initializer if init.name != name]
                    del candidate.graph.initializer[:]
                    candidate.graph.initializer.extend(kept)
                yield f"drop_zero_pad_value_node{node_index}_{name}", candidate


def node_signature(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def cse_probe(model: onnx.ModelProto) -> Iterable[tuple[str, onnx.ModelProto]]:
    seen: dict[bytes, tuple[int, list[str]]] = {}
    for index, node in enumerate(model.graph.node):
        signature = node_signature(node)
        if signature not in seen:
            seen[signature] = (index, list(node.output))
            continue
        first_index, first_outputs = seen[signature]
        if len(first_outputs) != len(node.output) or any(not value for value in node.output):
            continue
        replacements = dict(zip(node.output, first_outputs))
        candidate = copy.deepcopy(model)
        for consumer in candidate.graph.node:
            for input_index, value in enumerate(consumer.input):
                if value in replacements:
                    consumer.input[input_index] = replacements[value]
        for output in candidate.graph.output:
            if output.name in replacements:
                output.name = replacements[output.name]
        del candidate.graph.node[index]
        yield f"cse_node{index}_to{first_index}", candidate


def identity_probes(model: onnx.ModelProto) -> Iterable[tuple[str, onnx.ModelProto]]:
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Identity"} or len(node.input) != 1 or len(node.output) != 1:
            continue
        source, target = node.input[0], node.output[0]
        candidate = copy.deepcopy(model)
        for consumer in candidate.graph.node:
            for input_index, value in enumerate(consumer.input):
                if value == target:
                    consumer.input[input_index] = source
        for output in candidate.graph.output:
            if output.name == target:
                output.name = source
        del candidate.graph.node[index]
        yield f"remove_identity_node{index}", candidate


def first_case_dual(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    example = next(
        item
        for split in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(split, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    )
    result: dict[str, Any] = {}
    for label, disabled in (("disable_all", True), ("default", False)):
        try:
            _, session = make_session(model, disabled)
            raw = session.run(None, {"input": example["input"]})[0]
            result[label] = {
                "correct": bool(
                    np.array_equal((raw > 0).astype(np.float32), example["output"])
                ),
                "shape": list(raw.shape),
            }
        except Exception as exc:  # noqa: BLE001
            result[label] = {"correct": False, "error": f"{type(exc).__name__}:{exc}"}
    result["pass"] = all(row.get("correct") for row in result.values() if isinstance(row, dict))
    return result


def main() -> None:
    baseline_audit = json.loads((HERE / "baseline_audit.json").read_text())
    payload: dict[str, Any] = {"tasks": {}, "accepted": []}
    candidates_dir = HERE / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for task in TASKS:
        path = CURRENT / f"task{task:03d}.onnx"
        baseline = onnx.load(path)
        baseline_cost = int(baseline_audit["tasks"][str(task)]["score"]["cost"])
        probes = list(repeated_axis_probes(baseline))
        probes += list(alias_duplicate_probe(baseline))
        probes += list(optional_zero_probes(baseline))
        probes += list(cse_probe(baseline))
        probes += list(identity_probes(baseline))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label, model in probes:
            digest = sha_model(model)
            if digest in seen:
                continue
            seen.add(digest)
            row: dict[str, Any] = {
                "label": label,
                "sha256": digest,
                "static_params": nparams(model),
            }
            try:
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=True, data_prop=True
                )
                row["checker_strict"] = True
            except Exception as exc:  # noqa: BLE001
                row["checker_strict"] = False
                row["error"] = f"{type(exc).__name__}:{exc}"
                rows.append(row)
                continue
            row["first_case_dual"] = first_case_dual(model, task)
            if not row["first_case_dual"]["pass"]:
                rows.append(row)
                continue
            with tempfile.TemporaryDirectory(prefix=f"mid22_probe_{task:03d}_") as tmp:
                probe_path = Path(tmp) / "probe.onnx"
                onnx.save(model, probe_path)
                memory, params, cost = cost_of(str(probe_path))
            row["cost"] = {"memory": memory, "params": params, "cost": cost}
            row["strictly_cheaper"] = 0 <= cost < baseline_cost
            if row["strictly_cheaper"]:
                row["known_dual"] = known_dual(model, task)
                row["structure"] = structure(model, task)
                known_ok = all(
                    mode.get("perfect")
                    for mode in row["known_dual"].values()
                )
                row["eligible_before_fresh"] = bool(
                    known_ok and row["structure"].get("pass")
                )
                out_path = candidates_dir / f"task{task:03d}_{label}.onnx"
                onnx.save(model, out_path)
                row["path"] = str(out_path.relative_to(ROOT))
                row["saved_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
                if row["eligible_before_fresh"]:
                    payload["accepted"].append(
                        {"task": task, "label": label, "path": row["path"]}
                    )
            rows.append(row)
        payload["tasks"][str(task)] = {
            "baseline_cost": baseline_cost,
            "probe_count": len(rows),
            "first_case_survivors": sum(
                bool(row.get("first_case_dual", {}).get("pass")) for row in rows
            ),
            "strictly_cheaper": sum(bool(row.get("strictly_cheaper")) for row in rows),
            "eligible_before_fresh": sum(
                bool(row.get("eligible_before_fresh")) for row in rows
            ),
            "rows": rows,
        }
        print(
            f"task{task:03d} probes={len(rows)} first_pass="
            f"{payload['tasks'][str(task)]['first_case_survivors']} cheaper="
            f"{payload['tasks'][str(task)]['strictly_cheaper']} eligible="
            f"{payload['tasks'][str(task)]['eligible_before_fresh']}",
            flush=True,
        )
    (HERE / "micro_search.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
