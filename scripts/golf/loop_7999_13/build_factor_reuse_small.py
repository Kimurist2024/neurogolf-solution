#!/usr/bin/env python3
"""Build small exact/near-exact in-Einsum factor-reuse candidates."""

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
OUT = HERE / "lane_factor_reuse_small"


def eq_attr(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def transform(target: np.ndarray, source: np.ndarray, side: str) -> np.ndarray:
    if side == "right":
        value = target.astype(np.float64) @ np.linalg.pinv(source.astype(np.float64))
    else:
        value = np.linalg.pinv(source.astype(np.float64)) @ target.astype(np.float64)
    return value.astype(target.dtype)


def rewrite(
    model: onnx.ModelProto,
    target_name: str,
    source_name: str,
    side: str,
    suffix: str,
) -> dict[str, object]:
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    target = arrays[target_name]
    source = arrays[source_name]
    matrix = transform(target, source, side)
    matrix_name = f"{target_name}__from__{source_name}__{suffix}"
    model.graph.initializer.append(numpy_helper.from_array(matrix, matrix_name))
    uses = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == target_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise RuntimeError(f"non-Einsum use of {target_name}")
        attr = eq_attr(node)
        text = attr.s.decode("ascii")
        lhs, rhs = text.split("->", 1)
        operands = lhs.split(",")
        inputs = list(node.input)
        used = set("".join(operands) + rhs)
        available = iter(ch for ch in string.ascii_letters if ch not in used)
        for position in reversed(positions):
            latent = next(available)
            subs = operands[position]
            if len(subs) != 2:
                raise RuntimeError(f"unsupported target subscripts {subs}")
            if side == "right":
                # target[q,n] = M[q,p] @ source[p,n]
                replacement_subs = [subs[0] + latent, latent + subs[1]]
                replacement_inputs = [matrix_name, source_name]
            else:
                # target[n,q] = source[n,p] @ M[p,q]
                replacement_subs = [subs[0] + latent, latent + subs[1]]
                replacement_inputs = [source_name, matrix_name]
            operands[position : position + 1] = replacement_subs
            inputs[position : position + 1] = replacement_inputs
            uses += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(operands) + "->" + rhs).encode("ascii")
    if not uses:
        raise RuntimeError(f"no uses of {target_name}")
    remaining_uses = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining_uses[target_name]:
        raise RuntimeError(f"target {target_name} remains used")
    kept = [item for item in model.graph.initializer if item.name != target_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    rebuilt = matrix.astype(np.float64) @ source.astype(np.float64) if side == "right" else source.astype(np.float64) @ matrix.astype(np.float64)
    return {
        "target": target_name,
        "source": source_name,
        "side": side,
        "uses": uses,
        "target_params": int(target.size),
        "transform_params": int(matrix.size),
        "saving": int(target.size - matrix.size),
        "max_abs_error": float(np.max(np.abs(rebuilt - target.astype(np.float64)))),
    }


def load(archive: zipfile.ZipFile, task: int) -> onnx.ModelProto:
    return onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))


def save(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        configurations = {
            "ca": [("CA_h", "CM_h", "right")],
            "rm": [("RM_h", "RA_h", "right")],
            "both": [("CA_h", "CM_h", "right"), ("RM_h", "RA_h", "right")],
        }
        for label, config in configurations.items():
            model137 = load(archive, 137)
            changes137 = [
                rewrite(model137, target, source, side, label)
                for target, source, side in config
            ]
            path137 = OUT / f"task137_{label}.onnx"
            save(model137, path137)
            rows.append(
                {"task": 137, "variant": label, "path": str(path137.relative_to(ROOT)), "changes": changes137}
            )
            if label == "ca":
                accepted = OUT / "accepted" / "task137.onnx"
                accepted.parent.mkdir(parents=True, exist_ok=True)
                save(model137, accepted)
    (OUT / "build_manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
