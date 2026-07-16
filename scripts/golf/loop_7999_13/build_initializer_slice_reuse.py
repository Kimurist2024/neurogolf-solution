#!/usr/bin/env python3
"""Replace an Einsum-only initializer by an exact slice of a larger one.

If ``small == take(large, index, axis)``, each Einsum use of ``small`` can
consume ``large`` plus a shared one-hot selector.  The selector is contracted
inside the existing Einsum, so the rewrite adds no intermediate tensor and
changes the parameter count by ``large.shape[axis] - small.size``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import string
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attribute for attribute in node.attribute if attribute.name == "equation")


def rewrite(
    source_model: onnx.ModelProto,
    small_name: str,
    large_name: str,
    axis: int,
    index: int,
) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = copy.deepcopy(source_model)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    small = np.asarray(numpy_helper.to_array(initializers[small_name]))
    large = np.asarray(numpy_helper.to_array(initializers[large_name]))
    if large.ndim != small.ndim + 1:
        raise ValueError("large rank must be exactly one above small rank")
    if not 0 <= axis < large.ndim:
        raise ValueError("slice axis out of range")
    if large.shape[:axis] + large.shape[axis + 1 :] != small.shape:
        raise ValueError("slice shape does not match small initializer")
    if not np.array_equal(np.take(large, index, axis=axis), small):
        raise ValueError("requested slice is not exactly equal")
    if large.dtype != small.dtype:
        raise ValueError("initializer dtypes differ")

    selector_name = f"{small_name}__slice{axis}_{index}__of__{large_name}"
    selector = np.zeros((large.shape[axis],), dtype=large.dtype)
    selector[index] = 1
    model.graph.initializer.append(numpy_helper.from_array(selector, selector_name))

    uses = 0
    for node in model.graph.node:
        positions = [position for position, name in enumerate(node.input) if name == small_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise ValueError(f"non-Einsum use of {small_name}: {node.op_type}")
        attribute = equation_attribute(node)
        equation = attribute.s.decode("ascii")
        lhs, rhs = equation.split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        used_labels = set("".join(terms) + rhs)
        available = iter(label for label in string.ascii_letters if label not in used_labels)
        for position in reversed(positions):
            latent = next(available)
            small_term = terms[position]
            if "..." in small_term or len(small_term) != small.ndim:
                raise ValueError(f"unsupported small subscripts: {small_term}")
            large_term = small_term[:axis] + latent + small_term[axis:]
            terms[position : position + 1] = [large_term, latent]
            inputs[position : position + 1] = [large_name, selector_name]
            uses += 1
        del node.input[:]
        node.input.extend(inputs)
        attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")

    if uses == 0:
        raise ValueError(f"no uses of {small_name}")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining[small_name]:
        raise ValueError(f"failed to replace every use of {small_name}")
    kept = [tensor for tensor in model.graph.initializer if tensor.name != small_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, {
        "small": small_name,
        "large": large_name,
        "axis": axis,
        "index": index,
        "uses": uses,
        "removed_parameters": int(small.size),
        "selector_parameters": int(selector.size),
        "net_parameter_saving": int(small.size - selector.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        (13, "color_f", "Kfeat", 0, 1),
        (379, "Rflip", "QV", 1, 1),
    ]
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(baseline) as archive:
        for task, small, large, axis, index in configurations:
            try:
                source = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                model, change = rewrite(source, small, large, axis, index)
                path = out_dir / f"task{task:03d}.onnx"
                onnx.save(model, path)
                rows.append(
                    {
                        "task": task,
                        "built": True,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "change": change,
                    }
                )
            except Exception as error:
                rows.append({"task": task, "built": False, "error": repr(error)})
    document = {"baseline": str(baseline.relative_to(ROOT)), "rows": rows}
    (out_dir / "build_manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0 if all(row.get("built") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
