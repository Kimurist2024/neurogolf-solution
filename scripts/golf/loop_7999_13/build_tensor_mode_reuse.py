#!/usr/bin/env python3
"""Build audited higher-rank initializer mode-reuse candidates."""

from __future__ import annotations

import io
import json
import string
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_7999.13.zip"
OUT = HERE / "lane_tensor_mode_reuse"


def unfold(array: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def rewrite(
    model: onnx.ModelProto,
    target_name: str,
    source_name: str,
    axis: int,
) -> dict[str, object]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    target = arrays[target_name]
    source = arrays[source_name]
    target_matrix = unfold(target, axis).astype(np.float64)
    source_matrix = unfold(source, axis).astype(np.float64)
    # Prefer an exact row selector when the target mode only repeats/reorders
    # source rows.  This avoids the numerically equivalent 0.5/0.5 solutions
    # returned by the pseudoinverse for duplicate source rows.
    selector = np.zeros(
        (target_matrix.shape[0], source_matrix.shape[0]), dtype=target.dtype
    )
    selector_ok = True
    for target_index, target_row in enumerate(target_matrix):
        matches = [
            source_index
            for source_index, source_row in enumerate(source_matrix)
            if np.array_equal(target_row, source_row)
        ]
        if not matches:
            selector_ok = False
            break
        selector[target_index, matches[0]] = 1
    transform = (
        selector
        if selector_ok
        else (target_matrix @ np.linalg.pinv(source_matrix)).astype(target.dtype)
    )
    transform_name = f"{target_name}__mode{axis}__from__{source_name}"
    model.graph.initializer.append(numpy_helper.from_array(transform, transform_name))

    uses = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == target_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise RuntimeError(f"non-Einsum use of {target_name}")
        attr = equation_attribute(node)
        text = attr.s.decode("ascii")
        lhs, rhs = text.split("->", 1)
        operands = lhs.split(",")
        inputs = list(node.input)
        used = set("".join(operands) + rhs)
        available = iter(ch for ch in string.ascii_letters if ch not in used)
        for position in reversed(positions):
            latent = next(available)
            target_subscripts = operands[position]
            if len(target_subscripts) != target.ndim:
                raise RuntimeError(f"bad target subscripts {target_subscripts}")
            mode_label = target_subscripts[axis]
            source_subscripts = (
                target_subscripts[:axis] + latent + target_subscripts[axis + 1 :]
            )
            operands[position : position + 1] = [source_subscripts, mode_label + latent]
            inputs[position : position + 1] = [source_name, transform_name]
            uses += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(operands) + "->" + rhs).encode("ascii")

    if not uses:
        raise RuntimeError(f"no uses of {target_name}")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining[target_name]:
        raise RuntimeError(f"target {target_name} still used")
    kept = [item for item in model.graph.initializer if item.name != target_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    rebuilt = transform.astype(np.float64) @ source_matrix
    return {
        "target": target_name,
        "source": source_name,
        "axis": axis,
        "uses": uses,
        "target_params": int(target.size),
        "transform_params": int(transform.size),
        "saving": int(target.size - transform.size),
        "max_abs_error": float(np.max(np.abs(rebuilt - target_matrix))),
        "exact_row_selector": selector_ok,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configurations = [
        (74, "tfeat_from_bfeat", "Tfeat", "Bfeat", 1),
        (254, "sf8_from_se", "SF8", "SE", 2),
        (379, "nv_from_qv", "NV", "QV", 1),
        (379, "qv_from_nv", "QV", "NV", 1),
    ]
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        for task, label, target, source, axis in configurations:
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            try:
                change = rewrite(model, target, source, axis)
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
                path = OUT / f"task{task:03d}_{label}.onnx"
                onnx.save(model, path)
                rows.append(
                    {
                        "task": task,
                        "variant": label,
                        "built": True,
                        "path": str(path.relative_to(ROOT)),
                        "change": change,
                    }
                )
            except Exception as exc:
                rows.append(
                    {"task": task, "variant": label, "built": False, "error": repr(exc)}
                )
    (OUT / "build_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
