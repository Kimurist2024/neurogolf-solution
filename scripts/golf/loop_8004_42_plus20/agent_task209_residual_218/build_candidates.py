#!/usr/bin/env python3
"""Build exact residual-regolf candidates for the staged task209 model."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "others/71407/task209.onnx"
CANDIDATES = HERE / "candidates"
EXPECTED_BASE_SHA256 = "87690aaddd78db9a54a41b4a11edb73d503966eb8d27b4b60a3569fd1db0a751"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_runtime_metadata(model: onnx.ModelProto) -> dict[str, tuple[int, list[int]]]:
    """Measure every fixed-shape node output on one valid input."""
    traced = copy.deepcopy(model)
    inferred = shape_inference.infer_shapes(copy.deepcopy(traced), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    names: list[str] = []
    existing = {item.name for item in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
            if name:
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    example = scoring.convert_to_numpy(scoring.load_examples(209)["train"][0])
    if example is None:
        raise RuntimeError("known task209 example conversion failed")
    arrays = session.run(names, {session.get_inputs()[0].name: example["input"]})
    return {
        name: (helper.np_dtype_to_tensor_dtype(np.asarray(array).dtype), list(np.asarray(array).shape))
        for name, array in zip(names, arrays)
    }


RUNTIME_METADATA: dict[str, tuple[int, list[int]]] = {}


def make_truthful(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace inherited shape cloaks by complete measured static metadata."""
    model = copy.deepcopy(model)
    model.graph.ClearField("value_info")
    output = model.graph.output[0]
    output.type.tensor_type.elem_type = TensorProto.UINT8
    output.type.tensor_type.shape.ClearField("dim")
    for value in (1, 10, 30, 30):
        output.type.tensor_type.shape.dim.add().dim_value = value
    for node in model.graph.node:
        for name in node.output:
            if not name or name == "output":
                continue
            elem_type, dims = RUNTIME_METADATA[name]
            model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, dims))
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    return model


def decloak(model: onnx.ModelProto, which: str = "both") -> onnx.ModelProto:
    result = copy.deepcopy(model)
    remove: set[str] = set()
    rewires: dict[str, str] = {}
    if which in ("rbits", "both"):
        remove.update(("__cloak_sh_rbits", "__cloak_hid_rbits"))
        rewires["__cloak_hid_rbits"] = "rbits"
    if which in ("cidx", "both"):
        remove.update(("__cloak_sh_cidx", "__cloak_hid_cidx"))
        rewires["__cloak_hid_cidx"] = "cidx_y"
    nodes = []
    for source in result.graph.node:
        if any(name in remove for name in source.output):
            continue
        node = copy.deepcopy(source)
        for index, name in enumerate(node.input):
            node.input[index] = rewires.get(name, name)
        nodes.append(node)
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    return make_truthful(result)


def reshape_to_unsqueeze(model: onnx.ModelProto, truthful: bool = True) -> onnx.ModelProto:
    """Replace two fixed rank-2 -> rank-4 reshapes by shared axes [0,3]."""
    result = copy.deepcopy(model)
    expected = {
        "snorm4": ("snorm8", "sh_src4d"),
        "pnorm4": ("pnorm2rows", "sh_pat2d"),
    }
    replaced: set[str] = set()
    nodes = []
    for source in result.graph.node:
        node = copy.deepcopy(source)
        output = node.output[0] if node.output else ""
        if output in expected:
            input_name, shape_name = expected[output]
            if node.op_type != "Reshape" or list(node.input) != [input_name, shape_name]:
                raise RuntimeError(f"unexpected {output} node")
            node.op_type = "Unsqueeze"
            node.input[:] = [input_name, "axes03_i64"]
            node.ClearField("attribute")
            replaced.add(output)
        nodes.append(node)
    if replaced != set(expected):
        raise RuntimeError(f"missing reshape nodes: {set(expected) - replaced}")
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    kept = [item for item in result.graph.initializer if item.name not in {"sh_src4d", "sh_pat2d"}]
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(kept)
    result.graph.initializer.append(
        numpy_helper.from_array(np.asarray([0, 3], dtype=np.int64), name="axes03_i64")
    )
    return make_truthful(result) if truthful else result


def roundless_log_index(
    model: onnx.ModelProto, gamma: float, truthful: bool = True
) -> onnx.ModelProto:
    """Use a support-safe gamma so direct uint8 Cast equals Round+Cast."""
    result = copy.deepcopy(model)
    nodes = []
    removed = False
    rewired = False
    for source in result.graph.node:
        if list(source.output) == ["pcround"]:
            if source.op_type != "Round" or list(source.input) != ["pclog2"]:
                raise RuntimeError("unexpected pcround producer")
            removed = True
            continue
        node = copy.deepcopy(source)
        if list(node.output) == ["pclog2"]:
            if node.op_type != "Selu":
                raise RuntimeError("unexpected pclog2 producer")
            attrs = {attr.name: attr for attr in node.attribute}
            attrs["gamma"].f = float(gamma)
        for index, name in enumerate(node.input):
            if name == "pcround":
                node.input[index] = "pclog2"
                rewired = True
        nodes.append(node)
    if not (removed and rewired):
        raise RuntimeError("roundless rewrite incomplete")
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    kept_vi = [item for item in result.graph.value_info if item.name != "pcround"]
    result.graph.ClearField("value_info")
    result.graph.value_info.extend(kept_vi)
    return make_truthful(result) if truthful else result


def structural(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
        output = inferred.graph.output[0]
        row["output_shape"] = [int(dim.dim_value) for dim in output.type.tensor_type.shape.dim]
        row["truthful_output"] = row["output_shape"] == [1, 10, 30, 30]
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, truthful_output=False, strict_error=f"{type(exc).__name__}: {exc}")
    try:
        findings = check_conv_bias(model)
        row["conv_bias_ub0"] = not findings
        row["conv_bias_findings"] = findings
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["no_shape_cloak"] = not any(node.op_type == "CenterCropPad" for node in model.graph.node)
    row["no_lookup_abuse"] = not any(node.op_type in {"TfIdfVectorizer", "Hardmax"} for node in model.graph.node)
    row["banned_ops"] = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in {"Loop", "Scan", "NonZero", "Unique", "Compress"}
            or "Sequence" in node.op_type
        }
    )
    row["pass"] = bool(
        row.get("full_check")
        and row.get("strict_data_prop")
        and row.get("truthful_output")
        and row.get("conv_bias_ub0")
        and not row.get("banned_ops")
    )
    return row


def official_profile(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    return scoring.score_and_verify(
        copy.deepcopy(model), 209, str(HERE / "profile"), label=label, require_correct=False
    )


def main() -> int:
    ort.set_default_logger_severity(4)
    if digest(BASE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("staged task209 authority changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    base = onnx.load(BASE)
    global RUNTIME_METADATA
    RUNTIME_METADATA = trace_runtime_metadata(base)
    (HERE / "runtime_metadata.json").write_text(
        json.dumps(
            {
                name: {"elem_type": elem_type, "shape": dims}
                for name, (elem_type, dims) in RUNTIME_METADATA.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # 1.443359375 is binary-exact and makes the direct Cast land on the same
    # integer as Round for every reachable log2(lowbit) support value; audited
    # independently before any candidate can be admitted.
    gamma = 1.443359375
    variants: dict[str, onnx.ModelProto] = {}
    variants["inherited_roundless"] = roundless_log_index(base, gamma, truthful=False)
    variants["decloak_rbits"] = decloak(base, "rbits")
    variants["decloak_cidx"] = decloak(base, "cidx")
    variants["decloak"] = decloak(base, "both")
    variants["decloak_unsqueeze"] = reshape_to_unsqueeze(variants["decloak"])
    variants["decloak_roundless"] = roundless_log_index(variants["decloak"], gamma)
    variants["decloak_unsqueeze_roundless"] = roundless_log_index(
        variants["decloak_unsqueeze"], gamma
    )
    rows = []
    # Reproduced in the first fail-closed build pass from all 266 known cases.
    # Avoid replaying the inherited shape-cloak warnings on every rebuild.
    base_profile = {
        "memory": 1832,
        "params": 253,
        "cost": 2085,
        "score": 17.3574758657671,
        "correct": True,
    }
    for label, model in variants.items():
        path = CANDIDATES / f"task209_{label}.onnx"
        onnx.save(model, path)
        profile = official_profile(model, label)
        row = {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "nodes": len(model.graph.node),
            "params": scoring.calculate_params(model),
            "structure": structural(model),
            "official_profile": profile,
            "strict_lower": bool(
                profile is not None
                and base_profile is not None
                and int(profile["cost"]) < int(base_profile["cost"])
            ),
            "projected_gain": (
                math.log(int(base_profile["cost"]) / int(profile["cost"]))
                if profile is not None and base_profile is not None
                else None
            ),
        }
        rows.append(row)
        print(label, profile, row["structure"], flush=True)
    payload = {
        "authority": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": digest(BASE),
            "official_profile": base_profile,
        },
        "roundless_gamma": gamma,
        "rows": rows,
    }
    (HERE / "build.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
