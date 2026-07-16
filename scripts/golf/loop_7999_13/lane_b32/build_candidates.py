#!/usr/bin/env python3
"""Extract task219@8002.63 and build semantics-preserving graph fusions."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8002.63.zip"
MEMBER = "task219.onnx"
EXPECTED_BASE_SHA = "7a2ead58107803948d316fb8e00c4fd3ff601769309f9ad99661976f1a51bd67"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(item.dims)) for item in model.graph.initializer)


def load_baseline() -> tuple[Path, onnx.ModelProto]:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        payload = archive.read(MEMBER)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_BASE_SHA:
        raise RuntimeError(f"unexpected task219 baseline SHA: {digest}")
    path = HERE / "baseline_task219.onnx"
    path.write_bytes(payload)
    return path, onnx.load_model_from_string(payload)


def replace_three_add_chains(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fuse three associative uint8 Add chains into variadic Sum nodes."""
    skip_outputs = {"ha01u8", "hb01u8", "wh01"}
    replacements = {
        "ha": ["a0u8", "a1h8", "a2h"],
        "hb": ["b0u8", "b1h8", "b2h"],
        "winhash": ["rowcode14_u8tmp", "w1h", "w2h"],
    }
    nodes = []
    found: set[str] = set()
    for node in model.graph.node:
        if any(output in skip_outputs for output in node.output):
            continue
        output = node.output[0] if node.output else ""
        if output in replacements:
            if node.op_type != "Add":
                raise RuntimeError(f"expected Add producer for {output}")
            nodes.append(
                helper.make_node(
                    "Sum", replacements[output], [output], name=output
                )
            )
            found.add(output)
        else:
            nodes.append(node)
    if found != set(replacements):
        raise RuntimeError(f"missing replacement outputs: {set(replacements) - found}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {value for node in nodes for value in node.input if value}
    kept_initializers = [
        item for item in model.graph.initializer if item.name in used
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def reshape_initializer(model: onnx.ModelProto, name: str, dims: list[int]) -> None:
    item = next((item for item in model.graph.initializer if item.name == name), None)
    if item is None:
        raise RuntimeError(f"missing initializer {name}")
    old_count = int(np.prod(item.dims))
    new_count = int(np.prod(dims))
    if old_count != new_count:
        raise RuntimeError(f"reshape changes element count for {name}")
    del item.dims[:]
    item.dims.extend(dims)


def fold_unit_dimensions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Move singleton dimensions into existing operands and remove Unsqueezes."""
    for name in ("amask8", "bmask8", "bshift8"):
        reshape_initializer(model, name, [7, 1, 1])
    for name in ("cmaskv8", "cshiftv8"):
        reshape_initializer(model, name, [7, 1, 1, 1, 1])
    reshape_initializer(model, "cpaint8", [7, 1, 2, 1, 1])
    for name in ("tr_idx", "en9", "b_f2_val"):
        reshape_initializer(model, name, [1, 1, 1, 1])

    skip_outputs = {
        "ha2", "hb2", "kernel4u8", "tr_cond4", "te_cond4", "bcond4"
    }
    rewrites = {
        "ha2": "ha",
        "hb2": "hb",
        "kernel4u8": "kcode_rev",
        "tr_cond4": "tr_cond",
        "te_cond4": "te_cond_vec",
        "bcond4": "b_cond",
    }
    nodes = []
    for node in model.graph.node:
        if any(output in skip_outputs for output in node.output):
            continue
        for index, value in enumerate(node.input):
            if value in rewrites:
                node.input[index] = rewrites[value]
        if node.output and node.output[0] == "csel_masked":
            if node.op_type != "Concat":
                raise RuntimeError("csel_masked must be produced by Concat")
            axis = next(attr for attr in node.attribute if attr.name == "axis")
            axis.i = 2
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {value for node in nodes for value in node.input if value}
    kept_initializers = [
        item for item in model.graph.initializer if item.name in used
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def fuse_boolean_scale(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Cast(bool)->uint8 then Mul(scale) by one typed Where."""
    nodes = []
    replaced = False
    for node in model.graph.node:
        if "k2mask16" in node.output:
            if node.op_type != "Cast":
                raise RuntimeError("k2mask16 must be produced by Cast")
            continue
        if "k2" in node.output:
            if node.op_type != "Mul" or list(node.input) != ["k2mask16", "k997_u16"]:
                raise RuntimeError("unexpected k2 producer")
            nodes.append(
                helper.make_node(
                    "Where", ["tall3", "k997_u16", "pad_zero"], ["k2"],
                    name="k2"
                )
            )
            replaced = True
        else:
            nodes.append(node)
    if not replaced:
        raise RuntimeError("k2 fusion was not applied")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def commute_c_extract(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite (x & (3*s))/s as (x/s)&3 for power-of-two s."""
    nodes = []
    rewritten_masks: set[str] = set()
    rewritten_results: set[str] = set()
    for node in model.graph.node:
        if "cmask1u8" in node.output:
            if node.op_type != "Gather":
                raise RuntimeError("cmask1u8 must be produced by Gather")
            continue
        output = node.output[0] if node.output else ""
        if output in {"cm0", "cm1", "cm2"}:
            source = {"cm0": "f0code", "cm1": "f1code", "cm2": "f2code"}[output]
            nodes.append(helper.make_node("Div", [source, "cshift1u8"], [output], name=output))
            rewritten_masks.add(output)
        elif output in {"cs0", "cs1", "cs2"}:
            source = {"cs0": "cm0", "cs1": "cm1", "cs2": "cm2"}[output]
            nodes.append(
                helper.make_node(
                    "BitwiseAnd", [source, "tr_vals"], [output], name=output
                )
            )
            rewritten_results.add(output)
        else:
            nodes.append(node)
    if rewritten_masks != {"cm0", "cm1", "cm2"}:
        raise RuntimeError("not all C mask nodes were rewritten")
    if rewritten_results != {"cs0", "cs1", "cs2"}:
        raise RuntimeError("not all C result nodes were rewritten")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {value for node in nodes for value in node.input if value}
    kept_initializers = [
        item for item in model.graph.initializer if item.name in used
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    baseline_path, baseline = load_baseline()
    unit_candidate = fold_unit_dimensions(
        onnx.ModelProto.FromString(baseline.SerializeToString())
    )
    unit_candidate.producer_name = "codex-lane-b32"
    unit_candidate.graph.name = "task219_unit_dims_folded"
    unit_path = HERE / "task219_unit_dims_folded.onnx"
    onnx.save(unit_candidate, unit_path)

    candidate = fuse_boolean_scale(
        onnx.ModelProto.FromString(unit_candidate.SerializeToString())
    )
    candidate.producer_name = "codex-lane-b32"
    candidate.graph.name = "task219_unit_dims_bool_scale_fused"
    candidate_path = HERE / "task219_unit_dims_bool_scale_fused.onnx"
    onnx.save(candidate, candidate_path)

    winner = commute_c_extract(
        onnx.ModelProto.FromString(candidate.SerializeToString())
    )
    winner.producer_name = "codex-lane-b32"
    winner.graph.name = "task219_b32_winner"
    winner_path = HERE / "task219_b32_winner.onnx"
    onnx.save(winner, winner_path)

    manifest = {
        "baseline": {
            "zip": str(BASE_ZIP.relative_to(ROOT)),
            "member": MEMBER,
            "path": str(baseline_path.relative_to(ROOT)),
            "sha256": sha(baseline_path),
            "nodes": len(baseline.graph.node),
            "params": params(baseline),
        },
        "candidates": {
            "unit_dims_folded": {
                "path": str(unit_path.relative_to(ROOT)),
                "sha256": sha(unit_path),
                "nodes": len(unit_candidate.graph.node),
                "params": params(unit_candidate),
                "removed_intermediates": [
                    "ha2", "hb2", "kernel4u8",
                    "tr_cond4", "te_cond4", "bcond4"
                ],
                "expected_memory_reduction": 23,
                "value_semantics": "unit-dimension broadcasting only",
            },
            "unit_dims_bool_scale_fused": {
                "path": str(candidate_path.relative_to(ROOT)),
                "sha256": sha(candidate_path),
                "nodes": len(candidate.graph.node),
                "params": params(candidate),
                "additional_removed_intermediate": "k2mask16",
                "expected_additional_memory_reduction": 1,
                "value_semantics": "Where(tall3,98,0) == Cast(tall3,uint8)*98",
            },
            "winner": {
                "path": str(winner_path.relative_to(ROOT)),
                "sha256": sha(winner_path),
                "nodes": len(winner.graph.node),
                "params": params(winner),
                "additional_removed_intermediate": "cmask1u8",
                "removed_initializer": "cmaskv8",
                "identity": "(x & (3*s))/s == (x/s)&3 for power-of-two s",
            }
        },
        "rejected_designs": {
            "variadic_sum_uint8": "ONNX full checker rejects uint8 Sum"
        }
    }
    (HERE / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
