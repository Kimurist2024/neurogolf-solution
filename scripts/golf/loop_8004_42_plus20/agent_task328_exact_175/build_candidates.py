#!/usr/bin/env python3
"""Prepare task328 authority/control and exact parameter-only candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
EXACT554 = ROOT / "scripts/golf/loop_7999_13/lane_b26/task328_reuse_j_diagonal.onnx"
CONTROLS = HERE / "controls"
CANDIDATES = HERE / "candidates"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)
            )
            return
    raise KeyError(name)


def params(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(item.dims)) if item.dims else 1 for item in model.graph.initializer)


def combined(source: onnx.ModelProto, exponent: int, output: Path) -> dict:
    """Reuse ninvB for z[0], split the exact -3 compensation over two factors."""
    model = copy.deepcopy(source)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item)).copy()
        for item in model.graph.initializer
    }
    if set(("one", "ninvB", "CoreI", "CoreB")) - arrays.keys():
        raise RuntimeError("expected initializer missing")
    concat = next(node for node in model.graph.node if node.output == ["z"])
    if concat.op_type != "Concat" or concat.input[0] != "one":
        raise RuntimeError("unexpected z construction")
    concat.input[0] = "ninvB"

    left = np.float32(2.0**exponent)
    right = np.float32(-3.0 * (2.0 ** (-exponent)))
    product = np.float32(np.float32(arrays["ninvB"][0] * left) * right)
    if product != np.float32(1.0):
        raise RuntimeError((exponent, left, right, product))
    arrays["CoreI"][:, 0] *= left
    arrays["CoreB"][:, :, 0] *= right
    replace(model, "CoreI", arrays["CoreI"])
    replace(model, "CoreB", arrays["CoreB"])
    kept = [item for item in model.graph.initializer if item.name != "one"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    onnx.save(model, output)
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": sha(output.read_bytes()),
        "exponent": exponent,
        "core_i_scale": float(left),
        "core_b_scale": float(right),
        "compensation_product_float32": float(product),
        "params": params(model),
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    }


def main() -> None:
    CONTROLS.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    authority_zip_sha = sha(AUTHORITY_ZIP.read_bytes())
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority = archive.read("task328.onnx")
    authority_path = CONTROLS / "authority558.onnx"
    authority_path.write_bytes(authority)
    exact_path = CONTROLS / "exact554.onnx"
    exact_path.write_bytes(EXACT554.read_bytes())
    source = onnx.load(EXACT554)

    rows = []
    for exponent in range(-12, 13):
        label = f"m{-exponent}" if exponent < 0 else f"p{exponent}"
        rows.append(
            combined(source, exponent, CANDIDATES / f"task328_exact553_split_{label}.onnx")
        )
    result = {
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": authority_zip_sha,
        "authority": {
            "path": str(authority_path.relative_to(ROOT)),
            "sha256": sha(authority),
            "params": params(onnx.load_from_string(authority)),
        },
        "exact554": {
            "path": str(exact_path.relative_to(ROOT)),
            "sha256": sha(exact_path.read_bytes()),
            "params": params(source),
        },
        "identity": (
            "z0=(-1/3), CoreI[:,0]*=2^k, CoreB[:,:,0]*=-3*2^-k; "
            "their float32 product is exactly one for every emitted k"
        ),
        "candidates": rows,
    }
    (HERE / "build_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
