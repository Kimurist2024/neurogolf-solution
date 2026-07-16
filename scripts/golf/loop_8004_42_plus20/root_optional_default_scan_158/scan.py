#!/usr/bin/env python3
"""Profile exact removal of optional inputs equal to ONNX defaults.

Families: equal Split sizes -> num_outputs, Slice default axes/steps, and Pad
zero constant/all-axes inputs.  This is cost/structure triage only; survivors
must pass independent official runtime validation.
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
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"optional158_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def drop_dead(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def set_inputs(node: onnx.NodeProto, inputs: list[str]) -> None:
    while inputs and inputs[-1] == "":
        inputs.pop()
    del node.input[:]
    node.input.extend(inputs)


def split_variant(model: onnx.ModelProto, index: int, values: dict[str, np.ndarray]) -> tuple[onnx.ModelProto, str] | None:
    node = model.graph.node[index]
    if len(node.input) < 2 or not node.input[1] or len(node.output) < 1:
        return None
    split = values.get(node.input[1])
    if split is None or split.ndim != 1 or split.size != len(node.output):
        return None
    if not np.all(split == split.reshape(-1)[0]):
        return None
    opset = next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), 0)
    if opset < 18:
        return None
    candidate = copy.deepcopy(model)
    target = candidate.graph.node[index]
    set_inputs(target, [target.input[0]])
    attrs = [attr for attr in target.attribute if attr.name != "num_outputs"]
    del target.attribute[:]
    target.attribute.extend(attrs)
    target.attribute.append(helper.make_attribute("num_outputs", len(target.output)))
    return candidate, f"Split(equal={int(split.reshape(-1)[0])})->num_outputs={len(target.output)}"


def slice_variants(model: onnx.ModelProto, index: int, values: dict[str, np.ndarray]) -> list[tuple[onnx.ModelProto, str]]:
    node = model.graph.node[index]
    if len(node.input) < 3:
        return []
    starts = values.get(node.input[1])
    axes = values.get(node.input[3]) if len(node.input) > 3 and node.input[3] else None
    steps = values.get(node.input[4]) if len(node.input) > 4 and node.input[4] else None
    if starts is None or starts.ndim != 1:
        return []
    axes_default = bool(
        axes is not None
        and axes.ndim == 1
        and axes.size == starts.size
        and np.array_equal(axes.astype(np.int64), np.arange(starts.size, dtype=np.int64))
    )
    steps_default = bool(
        steps is not None and steps.ndim == 1 and steps.size == starts.size
        and np.all(steps == 1)
    )
    plans: list[tuple[bool, bool]] = []
    if axes_default:
        plans.append((True, False))
    if steps_default:
        plans.append((False, True))
    if axes_default and steps_default:
        plans.append((True, True))
    out = []
    for omit_axes, omit_steps in plans:
        candidate = copy.deepcopy(model)
        target = candidate.graph.node[index]
        inputs = list(target.input)
        while len(inputs) < 5:
            inputs.append("")
        if omit_axes:
            inputs[3] = ""
        if omit_steps:
            inputs[4] = ""
        set_inputs(target, inputs)
        parts = []
        if omit_axes:
            parts.append("axes")
        if omit_steps:
            parts.append("steps")
        out.append((candidate, "Slice omit default " + "+".join(parts)))
    return out


def pad_variants(model: onnx.ModelProto, index: int, values: dict[str, np.ndarray]) -> list[tuple[onnx.ModelProto, str]]:
    node = model.graph.node[index]
    if len(node.input) < 2:
        return []
    pads = values.get(node.input[1])
    constant = values.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    axes = values.get(node.input[3]) if len(node.input) > 3 and node.input[3] else None
    if pads is None or pads.ndim != 1:
        return []
    zero_default = bool(constant is not None and constant.size == 1 and float(constant.reshape(-1)[0]) == 0.0)
    rank = pads.size // 2 if pads.size % 2 == 0 else -1
    axes_default = bool(
        rank >= 0 and axes is not None and axes.ndim == 1 and axes.size == rank
        and np.array_equal(axes.astype(np.int64), np.arange(rank, dtype=np.int64))
    )
    plans = []
    if zero_default:
        plans.append((True, False))
    if axes_default:
        plans.append((False, True))
    if zero_default and axes_default:
        plans.append((True, True))
    out = []
    for omit_zero, omit_axes in plans:
        candidate = copy.deepcopy(model)
        target = candidate.graph.node[index]
        inputs = list(target.input)
        while len(inputs) < 4:
            inputs.append("")
        if omit_zero:
            inputs[2] = ""
        if omit_axes:
            inputs[3] = ""
        set_inputs(target, inputs)
        parts = []
        if omit_zero:
            parts.append("zero")
        if omit_axes:
            parts.append("axes")
        out.append((candidate, "Pad omit default " + "+".join(parts)))
    return out


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            baseline = None
            for index, node in enumerate(model.graph.node):
                variants: list[tuple[onnx.ModelProto, str]] = []
                if node.op_type == "Split":
                    built = split_variant(model, index, values)
                    if built is not None:
                        variants.append(built)
                elif node.op_type == "Slice":
                    variants.extend(slice_variants(model, index, values))
                elif node.op_type == "Pad":
                    variants.extend(pad_variants(model, index, values))
                for variant_no, (candidate, rewrite) in enumerate(variants):
                    dropped = drop_dead(candidate)
                    row = {
                        "task": task, "node_index": index, "source_op": node.op_type,
                        "variant": variant_no, "rewrite": rewrite,
                        "dropped_initializers": dropped,
                    }
                    try:
                        onnx.checker.check_model(candidate, full_check=True)
                        onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                        if baseline is None:
                            baseline = profile(model, task)
                        current = profile(candidate, task)
                        row.update(baseline=baseline, candidate=current, strict_lower=current["cost"] < baseline["cost"])
                        if row["strict_lower"]:
                            path = CANDIDATES / f"task{task:03d}_{index:04d}_{variant_no}_{node.op_type}.onnx"
                            onnx.save(candidate, path)
                            row["path"] = str(path.relative_to(ROOT))
                            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)), "profiles": len(rows),
        "strict_lower_count": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "profiles": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": len([row for row in rows if "error" in row]),
    }, indent=2))


if __name__ == "__main__":
    main()
