#!/usr/bin/env python3
"""Build current-authority-derived, non-promotable task270 regolf probes."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "baseline/task270_authority.onnx"
AUTHORITY_SHA = "0d848124abafda1daf24fe5f779ed5249c9b8b2054854264dde838b05e27a443"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [item for item in model.graph.initializer if item.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def reinfer(model: onnx.ModelProto) -> onnx.ModelProto:
    del model.graph.value_info[:]
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def make_truthful_direct() -> onnx.ModelProto:
    model = onnx.load(AUTHORITY)
    nodes = []
    for node in model.graph.node:
        out = node.output[0]
        if out in {"idx_shape2", "Ridx8_hid", "Cidx8_hid"}:
            continue
        if out in {"Ridx", "Cidx"}:
            source = out[0] + "idx8"
            node = helper.make_node("Cast", [source], [out], to=TensorProto.INT32)
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_initializers(model, {"i32like"})
    return reinfer(model)


def packed_extract(source: str, output: str, squared: bool = False) -> list[onnx.NodeProto]:
    low = f"{output}_low"
    high = f"{output}_high"
    nodes = [helper.make_node("Cast", [source], [low], to=TensorProto.UINT8)]
    if squared:
        high_float = f"{output}_high_float"
        nodes.extend([
            helper.make_node("Div", [source, "pack_scale"], [high_float]),
            helper.make_node("Cast", [high_float], [high], to=TensorProto.UINT8),
        ])
    else:
        nodes.append(helper.make_node("QuantizeLinear", [source, "pack_scale"], [high]))
    nodes.append(helper.make_node("Concat", [low, high], [output], axis=0))
    return nodes


def make_truthful_packed(shared_center_scale: bool, unsafe_pr2_quantize: bool = False) -> onnx.ModelProto:
    """Compile the same current graph with truthful indices and packed selectors."""
    model = onnx.load(AUTHORITY)
    inits = {item.name: item for item in model.graph.initializer}

    psel = np.zeros((1, 10), dtype=np.float32)
    psel[0, 3] = 1.0
    psel[0, 7] = 2048.0
    inits["Psel"].CopyFrom(numpy_helper.from_array(psel, name="Psel"))
    if shared_center_scale:
        cenc = numpy_helper.to_array(inits["Cenc"]).copy()
        cenc[0, 2] = 2048.0
        inits["Cenc"].CopyFrom(numpy_helper.from_array(cenc, name="Cenc"))
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(2048.0, dtype=np.float32), name="pack_scale")
    )

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        out = node.output[0]
        if out in {"idx_shape2", "Ridx8_hid", "Cidx8_hid"}:
            continue
        if out in {"pn8", "pr8", "pc8", "pr2u"}:
            source = {"pn8": "pn", "pr8": "pr", "pc8": "pc", "pr2u": "pr2"}[out]
            if out == "pr2u" and unsafe_pr2_quantize:
                nodes.extend(packed_extract(source, out, squared=False))
            else:
                nodes.extend(packed_extract(source, out, squared=(out == "pr2u")))
            continue
        if out in {"cr8_u", "cc8_u"} and shared_center_scale:
            # This Cast is the low packed byte and remains useful.
            nodes.append(node)
            continue
        if out in {"cr8_hiu", "cc8_hiu"} and shared_center_scale:
            source = out.replace("_hiu", "_code")
            nodes.append(helper.make_node("QuantizeLinear", [source, "pack_scale"], [out]))
            continue
        if out in {"cr8_lou", "cc8_lou"} and shared_center_scale:
            continue
        if out == "cr8" and shared_center_scale:
            node.input[1] = "cr8_u"
        if out == "cc8" and shared_center_scale:
            node.input[1] = "cc8_u"
        if out in {"Ridx", "Cidx"}:
            source = out[0] + "idx8"
            node = helper.make_node("Cast", [source], [out], to=TensorProto.INT32)
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_initializers(
        model,
        {"i32like"} | ({"sixteen_u8"} if shared_center_scale else set()),
    )
    return reinfer(model)


def save(name: str, model: onnx.ModelProto) -> None:
    path = HERE / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    print(f"{name} sha256={digest(path)} nodes={len(model.graph.node)}")


def main() -> int:
    if digest(AUTHORITY) != AUTHORITY_SHA:
        raise RuntimeError("authority guard failed")
    save("task270_truthful_direct_cost608.onnx", make_truthful_direct())
    save("task270_truthful_packed_cost595.onnx", make_truthful_packed(False))
    save("task270_truthful_shared_scale_cost592.onnx", make_truthful_packed(True))
    save(
        "task270_unsafe_pr2_saturating_cost588.onnx",
        make_truthful_packed(True, unsafe_pr2_quantize=True),
    )
    if digest(AUTHORITY) != AUTHORITY_SHA:
        raise RuntimeError("authority changed during build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
