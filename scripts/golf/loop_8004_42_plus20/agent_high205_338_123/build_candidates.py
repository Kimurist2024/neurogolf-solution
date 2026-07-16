#!/usr/bin/env python3
"""Build non-promoting task205/338 memshave probes from immutable 8009.46."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CURRENT = HERE / "current"
CANDIDATES = HERE / "candidates"
PROBES = HERE / "probes"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [item for item in model.graph.initializer if item.name != name]
    if len(kept) == len(model.graph.initializer):
        raise ValueError(f"initializer not found: {name}")
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)


def replace_node(model: onnx.ModelProto, output: str, node: onnx.NodeProto) -> None:
    for index, old in enumerate(model.graph.node):
        if output in old.output:
            model.graph.node.remove(old)
            model.graph.node.insert(index, node)
            return
    raise ValueError(f"node output not found: {output}")


def set_scalar(model: onnx.ModelProto, name: str, value: float) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            array = np.asarray(value, dtype=np.float32)
            model.graph.initializer.remove(item)
            model.graph.initializer.insert(index, numpy_helper.from_array(array, name))
            return
    raise ValueError(f"initializer not found: {name}")


def validate_and_save(model: onnx.ModelProto, path: Path) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "serialized_bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": sum(
            int(np.asarray(numpy_helper.to_array(item)).size) for item in model.graph.initializer
        ),
    }


def rowpow_to_selu(base: onnx.ModelProto) -> onnx.ModelProto:
    """Exact for valid inputs: tall_f and roww_max are non-negative."""
    model = copy.deepcopy(base)
    values = {
        item.name: float(np.asarray(numpy_helper.to_array(item)))
        for item in model.graph.initializer
        if item.name == "rowpow_thr"
    }
    gamma = values["rowpow_thr"]
    replace_node(
        model,
        "colq_scale",
        helper.make_node(
            "Selu", ["tall_f"], ["colq_scale"], name="exact_nonnegative_tall_scale",
            alpha=1.0, gamma=gamma,
        ),
    )
    replace_node(
        model,
        "roww_thr",
        helper.make_node(
            "Selu", ["roww_max"], ["roww_thr"], name="exact_nonnegative_roww_scale",
            alpha=1.0, gamma=gamma,
        ),
    )
    remove_initializer(model, "rowpow_thr")
    return model


def box_quant_to_cast(base: onnx.ModelProto, gain: float) -> onnx.ModelProto:
    """Experimental quantization rescale; must not be accepted without fresh proof."""
    model = copy.deepcopy(base)
    replace_node(
        model,
        "box_scaled",
        helper.make_node("Cast", ["color_oh"], ["box_scaled"], name="box_onehot_i8", to=TensorProto.INT8),
    )
    remove_initializer(model, "qbox80_scale")
    set_scalar(model, "gain19", gain)
    return model


def cmneg_to_neg(base: onnx.ModelProto) -> onnx.ModelProto:
    """Experimental predicate weakening; retained only as a negative-control probe."""
    model = copy.deepcopy(base)
    replace_node(model, "cmneg", helper.make_node("Neg", ["cm0"], ["cmneg"], name="cm0_neg"))
    remove_initializer(model, "neg3")
    return model


def gain_to_colq_scale(base: onnx.ModelProto) -> onnx.ModelProto:
    """Experimental reuse of tall-dependent scale in place of gain19."""
    model = copy.deepcopy(base)
    for node in model.graph.node:
        if "wcolor_counts" in node.output:
            if node.op_type != "Einsum" or node.input[-1] != "gain19":
                raise ValueError("unexpected wcolor_counts producer")
            node.input[-1] = "colq_scale"
            break
    else:
        raise ValueError("wcolor_counts producer not found")
    remove_initializer(model, "gain19")
    return model


def task338_cast(base: onnx.ModelProto) -> onnx.ModelProto:
    """Historical type-witness removal probe; expected to expose real spatial cost."""
    model = copy.deepcopy(base)
    replace_node(
        model,
        "x16",
        helper.make_node("Cast", ["xc"], ["x16"], name="truthful_fp16_cast", to=TensorProto.FLOAT16),
    )
    remove_initializer(model, "oneh")
    return model


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("submission.zip changed from immutable 8009.46 authority")
    CURRENT.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    PROBES.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "current": {},
        "built": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        bases = {task: onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx")) for task in (205, 338)}
    for task, model in bases.items():
        info = validate_and_save(model, CURRENT / f"task{task:03d}.onnx")
        manifest["current"][str(task)] = info

    builds = [
        (rowpow_to_selu(bases[205]), CANDIDATES / "task205_rowpow_selu.onnx", "exact_valid_domain"),
        (box_quant_to_cast(bases[205], 0.1902), PROBES / "task205_boxcast_gain01902.onnx", "experimental"),
        (box_quant_to_cast(rowpow_to_selu(bases[205]), 0.1902), PROBES / "task205_rowpow_selu_boxcast_gain01902.onnx", "experimental"),
        (cmneg_to_neg(rowpow_to_selu(bases[205])), PROBES / "task205_rowpow_selu_cmneg.onnx", "experimental"),
        (gain_to_colq_scale(rowpow_to_selu(bases[205])), PROBES / "task205_rowpow_selu_gaincolq.onnx", "experimental"),
        (task338_cast(bases[338]), PROBES / "task338_cast_truthful_probe.onnx", "truthful_cost_probe"),
    ]
    for model, path, intent in builds:
        info = validate_and_save(model, path)
        info["intent"] = intent
        manifest["built"].append(info)
    (HERE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
