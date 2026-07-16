#!/usr/bin/env python3
"""Independent static, profile, graph-delta, and support proof for task158."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / "others/71407/task158.onnx"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_residual_215"
    / "candidates/task158_exact_anchor_role_bits.onnx"
)
PARENT_SHA = "127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd"
CANDIDATE_SHA = "e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_dims(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def conv_bias_issues(model: onnx.ModelProto) -> list[dict[str, Any]]:
    shapes = {item.name: list(item.dims) for item in model.graph.initializer}
    rows = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) < 3:
            continue
        weight = shapes.get(node.input[1])
        bias = shapes.get(node.input[2])
        if not weight or not bias:
            continue
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        group = int(attrs.get("group", 1))
        out_channels = weight[0] if node.op_type == "Conv" else weight[1] * group
        if bias[0] != out_channels:
            rows.append(
                {
                    "node": node.name or node.output[0],
                    "op": node.op_type,
                    "bias": bias[0],
                    "out_channels": out_channels,
                }
            )
    return rows


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checks["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        checks["full_check"] = False
        errors.append(f"full_check:{type(exc).__name__}:{exc}")

    inferred: onnx.ModelProto | None = None
    for data_prop, key in ((False, "strict_shape"), (True, "strict_shape_data_prop")):
        try:
            value = shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=data_prop
            )
            checks[key] = True
            if data_prop:
                inferred = value
        except Exception as exc:  # noqa: BLE001
            checks[key] = False
            errors.append(f"{key}:{type(exc).__name__}:{exc}")

    inspected = inferred if inferred is not None else model
    values = list(inspected.graph.input) + list(inspected.graph.value_info) + list(inspected.graph.output)
    initializer_arrays = [np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer]
    checks.update(
        canonical_io=(
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        standard_domains=(
            all(item.domain in ("", "ai.onnx") for item in model.opset_import)
            and all(node.domain in ("", "ai.onnx") for node in model.graph.node)
        ),
        no_functions=not model.functions,
        no_sparse_initializers=not model.graph.sparse_initializer,
        no_external_data=all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in model.graph.initializer
        ),
        no_nested_graphs=all(
            attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attr in node.attribute
        ),
        no_banned_or_sequence_ops=all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        all_inferred_shapes_static_positive=all(tensor_dims(value) is not None for value in values),
        finite_initializers=all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in initializer_arrays
        ),
        conv_family_bias_ub0=not conv_bias_issues(model),
        no_hardmax_or_tfidf=all(
            node.op_type not in ("Hardmax", "TfIdfVectorizer") for node in model.graph.node
        ),
    )
    return {
        "checks": checks,
        "errors": errors,
        "pass": all(checks.values()),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params_static": scoring.calculate_params(model),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "conv_bias_issues": conv_bias_issues(model),
        "max_initializer_elements": max((array.size for array in initializer_arrays), default=0),
    }


def runtime_shape_truth(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    ordered = [value.name for value in traced.graph.output]
    for node in traced.graph.node:
        for name in node.output:
            if name and name not in ordered:
                if name not in typed:
                    return {"pass": False, "error": f"missing inferred type for {name}"}
                traced.graph.output.append(copy.deepcopy(typed[name]))
                ordered.append(name)
    declared = [tensor_dims(typed[name]) for name in ordered]
    sanitized = scoring.sanitize_model(traced)
    if sanitized is None:
        return {"pass": False, "error": "sanitize_model rejected traced graph"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    examples = scoring.load_examples(158)
    first = next(
        scoring.convert_to_numpy(item)
        for split in ("train", "test", "arc-gen")
        for item in examples[split]
        if scoring.convert_to_numpy(item) is not None
    )
    arrays = session.run(None, {session.get_inputs()[0].name: first["input"]})
    mismatches = []
    nonfinite = 0
    for name, expected, array in zip(ordered, declared, arrays, strict=True):
        actual = list(np.asarray(array).shape)
        if expected != actual:
            mismatches.append({"name": name, "declared": expected, "actual": actual})
        if np.asarray(array).dtype.kind in "fc":
            nonfinite += int(np.asarray(array).size - np.count_nonzero(np.isfinite(array)))
    return {
        "traced_outputs": len(ordered),
        "declared_actual_mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "pass": not mismatches and nonfinite == 0,
    }


def official_profile(model: onnx.ModelProto, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="review158_219_", dir="/tmp") as directory:
        row = scoring.score_and_verify(
            copy.deepcopy(model), 158, directory, label=label, require_correct=True
        )
    if row is None:
        raise RuntimeError(f"official profile rejected {label}")
    return row


def graph_delta(parent: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    def by_output(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
        return {node.output[0]: node for node in model.graph.node if node.output and node.output[0]}

    pnodes, cnodes = by_output(parent), by_output(candidate)
    common = set(pnodes) & set(cnodes)
    changed = sorted(
        name
        for name in common
        if pnodes[name].SerializeToString(deterministic=True)
        != cnodes[name].SerializeToString(deterministic=True)
    )
    pinits = {item.name: np.asarray(numpy_helper.to_array(item)) for item in parent.graph.initializer}
    cinits = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    common_inits_equal = all(
        pinits[name].dtype == cinits[name].dtype
        and pinits[name].shape == cinits[name].shape
        and np.array_equal(pinits[name], cinits[name])
        for name in set(pinits) & set(cinits)
    )
    removed_nodes = sorted(set(pnodes) - set(cnodes))
    added_nodes = sorted(set(cnodes) - set(pnodes))
    removed_inits = sorted(set(pinits) - set(cinits))
    added_inits = sorted(set(cinits) - set(pinits))
    exact_whitelist = bool(
        removed_nodes == ["more_role_low", "more_role_mid", "more_role_u8", "role_threshold"]
        and added_nodes == ["anchor_low_bits", "anchor_score_u8"]
        and changed == ["anchor_high", "low_mask", "phase_ge_0"]
        and removed_inits == ["more_role_1", "more_role_2", "more_role_3", "phase_cut_0"]
        and added_inits == ["anchor_low_bit_mask"]
        and common_inits_equal
        and int(cinits["anchor_low_bit_mask"].reshape(())) == 10
        and cnodes["anchor_score_u8"].op_type == "Cast"
        and list(cnodes["anchor_score_u8"].input) == ["top_values"]
        and cnodes["anchor_low_bits"].op_type == "BitwiseAnd"
        and list(cnodes["anchor_low_bits"].input) == ["anchor_score_u8", "anchor_low_bit_mask"]
        and cnodes["low_mask"].op_type == "Greater"
        and list(cnodes["low_mask"].input) == ["anchor_low_bits", "pq_u8_zero"]
        and cnodes["anchor_high"].op_type == "Xor"
        and list(cnodes["anchor_high"].input) == ["anchor_valid", "low_mask"]
        and cnodes["phase_ge_0"].op_type == "Greater"
        and list(cnodes["phase_ge_0"].input) == ["anchor_score_u8", "lutnp_shift4"]
    )
    return {
        "removed_node_outputs": removed_nodes,
        "added_node_outputs": added_nodes,
        "changed_common_node_outputs": changed,
        "removed_initializers": removed_inits,
        "added_initializers": added_inits,
        "common_initializers_bitwise_equal": common_inits_equal,
        "exact_rewrite_whitelist": exact_whitelist,
    }


def support_proof(parent: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    pinit = {item.name: np.asarray(numpy_helper.to_array(item)) for item in parent.graph.initializer}
    cinit = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    stencil = pinit["anchor_stencil"].reshape(3, 3).astype(np.int64)
    if stencil.tolist() != [[2, 8, -10], [24, 72, -108], [-26, -108, 0]]:
        raise AssertionError(f"unexpected stencil: {stencil.tolist()}")
    if int(cinit["anchor_low_bit_mask"].reshape(())) != 0b1010:
        raise AssertionError("candidate bitmask is not 0b1010")

    positive: set[int] = set()
    nonzero: set[int] = set()
    configurations = []
    # A local box is translated far from a boundary.  Conv origins are every
    # second coordinate.  Enumerating both residues therefore covers the full
    # infinite stride lattice; real top/bottom/left/right boundaries only
    # remove some of these windows and cannot introduce a new value.
    for magnitude in (1, 2, 3):
        for diagonal in (0, 1):
            endpoints = ((0, 0), (2, 2)) if diagonal == 0 else ((0, 2), (2, 0))
            for low_endpoint in (0, 1):
                for row_residue in (0, 1):
                    for col_residue in (0, 1):
                        base_r = 20 + row_residue
                        base_c = 20 + col_residue
                        cells: dict[tuple[int, int], int] = {}
                        for endpoint_index, (block_r, block_c) in enumerate(endpoints):
                            code = 1 if endpoint_index == low_endpoint else 2
                            r0 = base_r + block_r * magnitude
                            c0 = base_c + block_c * magnitude
                            for dr in range(magnitude):
                                for dc in range(magnitude):
                                    cells[(r0 + dr, c0 + dc)] = code
                        rows = [coord[0] for coord in cells]
                        cols = [coord[1] for coord in cells]
                        local_scores = set()
                        for origin_r in range(min(rows) - 2, max(rows) + 1):
                            if origin_r % 2:
                                continue
                            for origin_c in range(min(cols) - 2, max(cols) + 1):
                                if origin_c % 2:
                                    continue
                                score = 0
                                for kr in range(3):
                                    for kc in range(3):
                                        score += cells.get((origin_r + kr, origin_c + kc), 0) * int(stencil[kr, kc])
                                local_scores.add(score)
                                if score:
                                    nonzero.add(score)
                                if score > 0:
                                    positive.add(score)
                        configurations.append(
                            {
                                "magnitude": magnitude,
                                "diagonal": diagonal,
                                "low_endpoint": low_endpoint,
                                "row_residue": row_residue,
                                "col_residue": col_residue,
                                "scores": sorted(local_scores),
                            }
                        )

    expected = {2, 4, 8, 10, 16, 20, 24, 26, 48, 52, 72, 106, 144, 212}
    if positive != expected:
        raise AssertionError(f"positive support mismatch: {sorted(positive)}")

    table = []
    for value in [0, *sorted(positive)]:
        threshold = 144 if value >= 62 else 48 if value >= 22 else 16 if value >= 6 else 4
        valid = value >= 2
        parent_high = value >= threshold
        parent_low = valid != parent_high
        candidate_low = (value & 0b1010) > 0
        candidate_high = valid != candidate_low
        parent_phase0 = value >= 6
        candidate_phase0 = value > 4
        if (parent_low, parent_high, parent_phase0) != (
            candidate_low,
            candidate_high,
            candidate_phase0,
        ):
            raise AssertionError(f"classification mismatch for score {value}")
        table.append(
            {
                "score": value,
                "parent_threshold": threshold,
                "valid": valid,
                "parent_low": parent_low,
                "candidate_score_and_0b1010": value & 0b1010,
                "candidate_low": candidate_low,
                "parent_high": parent_high,
                "candidate_high": candidate_high,
                "parent_phase0_ge6": parent_phase0,
                "candidate_phase0_gt4": candidate_phase0,
            }
        )

    # At most ceil((m+2)/2)^2 <= 9 sampled 3x3 windows intersect one m x m
    # endpoint block.  There are at most 4 generator boxes and 2 endpoints.
    affected_upper_bound = 4 * 2 * 9
    zero_lower_bound = 13 * 13 - affected_upper_bound
    if zero_lower_bound < 8:
        raise AssertionError("not enough guaranteed zero scores for TopK(8)")
    return {
        "generator_dimensions": {
            "magnifications": [1, 2, 3],
            "flip_classes": "two opposite diagonals x which geometric endpoint has lower colour id",
            "translation_classes": "row/col modulo Conv stride 2",
            "configuration_count": len(configurations),
        },
        "generator_completeness_reason": (
            "hflip/vflip of opposite 3x3 corners reduce exactly to diagonal x endpoint assignment; "
            "every magnification is 1..3; translation is periodic modulo stride 2"
        ),
        "distinct_box_isolation": (
            "common.overlaps spacing=2 permits boxes only when their nearest cells differ by at "
            "least 3 on a separating axis; a 3-cell Conv window spans only 2"
        ),
        "fill_background_independence": (
            "endpoint_code is nonzero only for the two endpoint-colour channels, so fill and "
            "background contribute exactly zero to anchor_score"
        ),
        "boundary_completeness": (
            "the infinite-lattice enumeration covers all stride residues; finite boundaries only "
            "delete windows and therefore cannot add a score"
        ),
        "all_nonzero_local_scores": sorted(nonzero),
        "positive_support": sorted(positive),
        "topk_support": [0, *sorted(positive)],
        "topk_affected_window_upper_bound": affected_upper_bound,
        "topk_exact_zero_window_lower_bound": zero_lower_bound,
        "topk_negative_exclusion": zero_lower_bound >= 8,
        "classification_table": table,
        "all_role_and_phase_classifications_equal": True,
        "configurations": configurations,
        "pass": True,
    }


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(exist_ok=True)
    hashes = {"parent": digest(PARENT), "candidate": digest(CANDIDATE)}
    if hashes != {"parent": PARENT_SHA, "candidate": CANDIDATE_SHA}:
        raise RuntimeError(f"payload hash mismatch: {hashes}")
    parent = onnx.load(PARENT)
    candidate = onnx.load(CANDIDATE)
    proof = support_proof(parent, candidate)
    (HERE / "evidence/support_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    result = {
        "task": 158,
        "hashes": hashes,
        "paths": {
            "parent": str(PARENT.relative_to(ROOT)),
            "candidate": str(CANDIDATE.relative_to(ROOT)),
        },
        "structure": {"parent": structure(parent), "candidate": structure(candidate)},
        "graph_delta": graph_delta(parent, candidate),
        "runtime_shape_truth": {
            "parent": runtime_shape_truth(parent),
            "candidate": runtime_shape_truth(candidate),
        },
        "official_profile": {
            "parent": official_profile(parent, "parent219"),
            "candidate": official_profile(candidate, "candidate219"),
        },
        "support_proof_summary": {
            key: proof[key]
            for key in (
                "positive_support",
                "topk_support",
                "topk_affected_window_upper_bound",
                "topk_exact_zero_window_lower_bound",
                "topk_negative_exclusion",
                "all_role_and_phase_classifications_equal",
                "pass",
            )
        },
    }
    profile = result["official_profile"]
    result["gain"] = {
        "cost_before": int(profile["parent"]["cost"]),
        "cost_after": int(profile["candidate"]["cost"]),
        "cost_reduction": int(profile["parent"]["cost"] - profile["candidate"]["cost"]),
        "score_gain": math.log(profile["parent"]["cost"] / profile["candidate"]["cost"]),
    }
    result["pass"] = bool(
        result["structure"]["parent"]["pass"]
        and result["structure"]["candidate"]["pass"]
        and result["graph_delta"]["exact_rewrite_whitelist"]
        and result["runtime_shape_truth"]["parent"]["pass"]
        and result["runtime_shape_truth"]["candidate"]["pass"]
        and profile["parent"]["correct"]
        and profile["candidate"]["correct"]
        and int(profile["parent"]["cost"]) == 7525
        and int(profile["candidate"]["cost"]) == 7498
        and proof["pass"]
    )
    (HERE / "evidence/static_math_profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "gain": result["gain"], "support": result["support_proof_summary"]}, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
