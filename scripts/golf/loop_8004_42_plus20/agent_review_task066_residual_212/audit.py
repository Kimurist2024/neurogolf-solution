#!/usr/bin/env python3
"""Independent fail-closed review of the task066 cost-551 residual candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
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
PARENT = ROOT / "others/71407/task066.onnx"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task066_residual_208/task066_residual_cost551.onnx"
)
GEOMETRY_PROOF = (
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_task066_selu_200/geometry_proof.json"
)
EXPECTED_PARENT_SHA = "2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b"
EXPECTED_CANDIDATE_SHA = "622b3b28271806949bb18e8b9517335d49cb0383410caf36a19e064d95798dd3"
EXPECTED_GEOMETRY_SHA = "653bb75258ef8e80d9967d1b64ac8a61d75aff0400b38bdb524c1fae447c121f"
FRESH = ((66_212_001, 2000), (66_212_002, 2000))
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
TRACE = (
    "Gv",
    "Gh",
    "Gf",
    "G",
    "Ov",
    "Oh",
    "O",
    "aMask",
    "bMask",
    "selF",
    "selLog",
    "selQ",
    "ti",
)
RISKY_LOOKUP_OPS = {
    "TfIdfVectorizer",
    "CategoryMapper",
    "Hardmax",
    "GatherND",
    "ScatterND",
    "TopK",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    return value


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def attr_equation(node: onnx.NodeProto) -> str:
    attr = next(item for item in node.attribute if item.name == "equation")
    value = helper.get_attribute_value(attr)
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def graph_delta(parent_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    parent = onnx.load_model_from_string(parent_data)
    candidate = onnx.load_model_from_string(candidate_data)
    assert len(parent.graph.node) == len(candidate.graph.node)
    changed = []
    for index, (left, right) in enumerate(zip(parent.graph.node, candidate.graph.node, strict=True)):
        if left.SerializeToString(deterministic=True) != right.SerializeToString(deterministic=True):
            changed.append(
                {
                    "index": index,
                    "parent_op": left.op_type,
                    "candidate_op": right.op_type,
                    "output": list(right.output),
                    "parent_inputs": list(left.input),
                    "candidate_inputs": list(right.input),
                    "parent_equation": attr_equation(left),
                    "candidate_equation": attr_equation(right),
                }
            )

    p_init = {item.name: item for item in parent.graph.initializer}
    c_init = {item.name: item for item in candidate.graph.initializer}
    common_equal = all(
        p_init[name].SerializeToString(deterministic=True)
        == c_init[name].SerializeToString(deterministic=True)
        for name in set(p_init) & set(c_init)
    )
    p_shell = copy.deepcopy(parent)
    c_shell = copy.deepcopy(candidate)
    del p_shell.graph.node[:]
    del c_shell.graph.node[:]
    del p_shell.graph.initializer[:]
    del c_shell.graph.initializer[:]
    shell_equal = (
        p_shell.SerializeToString(deterministic=True)
        == c_shell.SerializeToString(deterministic=True)
    )
    expected_candidate_equations = {
        "Gv": "bchw,bdrw,qc,qz,z,ad,ed,aj,aj,ek,k,fl,fl,fm,m,h->b",
        "Gh": "bchw,bdhr,qc,qz,z,ad,ed,aj,aj,ek,k,fl,fl,fm,m,w->b",
    }
    exact = bool(
        [row["index"] for row in changed] == [22, 23]
        and [row["output"] for row in changed] == [["Gv"], ["Gh"]]
        and all(row["parent_op"] == row["candidate_op"] == "Einsum" for row in changed)
        and all(
            row["candidate_equation"] == expected_candidate_equations[row["output"][0]]
            for row in changed
        )
        and set(p_init) - set(c_init) == {"greenhalf10"}
        and not (set(c_init) - set(p_init))
        and common_equal
        and shell_equal
    )
    return {
        "changed_nodes": changed,
        "removed_initializers": sorted(set(p_init) - set(c_init)),
        "added_initializers": sorted(set(c_init) - set(p_init)),
        "common_initializers_proto_equal": common_equal,
        "model_and_graph_shell_equal": shell_equal,
        "whitelist_exact": exact,
    }


def selector_proof(parent_data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(parent_data)
    init = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    u = init["Uchan"]
    v = init["Vchan"]
    tr = init["Trow"]
    tc = init["Tcol"]
    z = init["z1"]

    # Keep all row selectors explicit.  The first contraction chooses U row 0,
    # the second chooses V row 2, and the scalar contraction chooses row 2.
    left_terms = u[:, None, :] * tr[:, :, None] * tc[:, :, None]
    right_terms = v[:, None, :] * tc[:, :, None] * z[None, :, None]
    scalar_terms = (
        u[:, :, None]
        * v[:, :, None]
        * tc[:, None, :]
        * z[None, None, :]
    )
    left = np.sum(left_terms, axis=(0, 1), dtype=np.float32)
    right = np.sum(right_terms, axis=(0, 1), dtype=np.float32)
    scalar = np.sum(scalar_terms, dtype=np.float32)
    rebuilt = left * right * scalar
    original = init["greenhalf10"]

    left_nonzero = np.argwhere(left_terms != 0)
    right_nonzero = np.argwhere(right_terms != 0)
    scalar_nonzero = np.argwhere(scalar_terms != 0)
    combined_nonzero = []
    for a, j, d in left_nonzero:
        for e, k, d2 in right_nonzero:
            if d2 != d:
                continue
            for f, ell, m in scalar_nonzero:
                value = (
                    left_terms[a, j, d]
                    * right_terms[e, k, d]
                    * scalar_terms[f, ell, m]
                )
                if value != 0:
                    combined_nonzero.append(
                        {
                            "a": int(a),
                            "j": int(j),
                            "e": int(e),
                            "k": int(k),
                            "f": int(f),
                            "l": int(ell),
                            "m": int(m),
                            "d": int(d),
                            "value": float(value),
                        }
                    )

    exact_values = set(np.unique(np.concatenate([left_terms.ravel(), right_terms.ravel(), scalar_terms.ravel()])).tolist())
    pass_proof = bool(
        np.array_equal(left, u[0])
        and np.array_equal(right, v[2])
        and float(scalar) == -1.0
        and np.array_equal(rebuilt, original)
        and exact_values <= {-1.0, 0.0, 1.0}
        and combined_nonzero
        == [{"a": 0, "j": 0, "e": 2, "k": 1, "f": 2, "l": 0, "m": 1, "d": 3, "value": 1.0}]
    )
    return {
        "selected_U_row": left.tolist(),
        "selected_V_row": right.tolist(),
        "scalar": float(scalar),
        "rebuilt": rebuilt.tolist(),
        "original_greenhalf10": original.tolist(),
        "factor_value_set": sorted(exact_values),
        "combined_nonzero_terms": combined_nonzero,
        "unique_nonzero_term": len(combined_nonzero) == 1,
        "float32_exact_reason": (
            "the reconstructed d-selector has exactly one nonzero multiplication path; "
            "all its factors are +/-1 and all other paths are zero"
        ),
        "generator_rounding_bound": (
            "Gv/Gh are nonnegative sums with two green cells and pow2 support 0..19, "
            "so max=2*(2**20-1)<2**24 and every result is exact in float32"
        ),
        "pass": pass_proof,
    }


def inherited_geometry() -> dict[str, Any]:
    data = GEOMETRY_PROOF.read_bytes()
    assert sha256(data) == EXPECTED_GEOMETRY_SHA
    proof = json.loads(data)
    counts = proof["geometry_parameter_tuples"]
    valid = bool(
        proof["all_assertions_passed"]
        and proof["counterexample"] is None
        and counts
        == {"S": 15336, "U": 449928, "with_flip_and_xpose": 1861056}
    )
    return {
        "path": str(GEOMETRY_PROOF.relative_to(ROOT)),
        "sha256": EXPECTED_GEOMETRY_SHA,
        "counts": counts,
        "consequence": proof["consequence"],
        "all_assertions_passed": proof["all_assertions_passed"],
        "counterexample": proof["counterexample"],
        "still_applicable_reason": (
            "only Gv/Gh changed; their exact reconstructed selector leaves Gv/Gh and "
            "therefore every premise and downstream carrier of the reviewed proof unchanged"
        ),
        "pass": valid,
    }


def static_audit(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_error"] = f"{type(exc).__name__}: {exc}"

    inferred_models = {}
    for label, data_prop in (("strict", False), ("strict_data_prop", True)):
        try:
            inferred_models[label] = shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=data_prop
            )
            result[label] = True
        except Exception as exc:  # noqa: BLE001
            result[label] = False
            result[f"{label}_error"] = f"{type(exc).__name__}: {exc}"

    unresolved = []
    inferred = inferred_models.get("strict_data_prop")
    if inferred is not None:
        typed = {
            value.name: value
            for value in list(inferred.graph.value_info) + list(inferred.graph.output)
        }
        for node in inferred.graph.node:
            for name in node.output:
                if not name:
                    continue
                value = typed.get(name)
                if value is None or any(
                    dim is None or dim <= 0 for dim in tensor_shape(value)
                ):
                    unresolved.append(name)

    excluded = sorted(
        {
            node.op_type
            for node in model.graph.node
            if any(token in node.op_type.upper() for token in scoring._EXCLUDED_OP_TYPES)
            or "SEQUENCE" in node.op_type.upper()
        }
    )
    nested = [
        f"{node.op_type}:{attr.name}"
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    lookup = sorted({node.op_type for node in model.graph.node} & RISKY_LOOKUP_OPS)
    init_sizes = {
        item.name: int(np.asarray(numpy_helper.to_array(item)).size)
        for item in model.graph.initializer
    }
    result.update(
        {
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "params_direct": sum(init_sizes.values()),
            "largest_initializer": max(init_sizes.items(), key=lambda item: item[1]),
            "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
            and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nested_graphs": nested,
            "banned_ops": excluded,
            "risky_lookup_ops": lookup,
            "conv_bias_ub": check_conv_bias(model),
            "unresolved_or_dynamic_node_outputs": unresolved,
        }
    )
    result["pass"] = bool(
        result["full_check"]
        and result["strict"]
        and result["strict_data_prop"]
        and result["standard_domains"]
        and not result["functions"]
        and not result["sparse_initializers"]
        and not result["nested_graphs"]
        and not result["banned_ops"]
        and not result["risky_lookup_ops"]
        and not result["conv_bias_ub"]
        and not result["unresolved_or_dynamic_node_outputs"]
    )
    return result


def official_profile(data: bytes, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task066_review212_", dir="/tmp") as work:
        row = scoring.score_and_verify(
            onnx.load_model_from_string(data), 66, work, label=label, require_correct=True
        )
    assert row is not None
    return {
        "memory": int(row["memory"]),
        "params": int(row["params"]),
        "cost": int(row["cost"]),
        "correct": bool(row["correct"]),
    }


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def traced_model(data: bytes, names: tuple[str, ...]) -> bytes:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    existing = {value.name for value in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(copy.deepcopy(typed[name]))
            existing.add(name)
    return model.SerializeToString()


def known_cases() -> list[dict[str, Any]]:
    examples = scoring.load_examples(66)
    return [row for split in ("train", "test", "arc-gen") for row in examples[split]]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_2dd70a9a")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [generator.generate() for _ in range(count)]


def raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def evaluate_pair(
    parent: ort.InferenceSession,
    candidate: ort.InferenceSession,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "parent_gold": 0,
        "candidate_gold": 0,
        "final_raw_equal": 0,
        "trace_raw_equal": {name: 0 for name in TRACE},
        "parent_nonfinite": {"output": 0, **{name: 0 for name in TRACE}},
        "candidate_nonfinite": {"output": 0, **{name: 0 for name in TRACE}},
        "runtime_errors": 0,
        "first_difference": None,
    }
    for index, example in enumerate(cases):
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            continue
        result["valid"] += 1
        outputs = {}
        for label, session in (("parent", parent), ("candidate", candidate)):
            try:
                outputs[label] = [
                    np.asarray(value)
                    for value in session.run(
                        None, {session.get_inputs()[0].name: converted["input"]}
                    )
                ]
            except Exception as exc:  # noqa: BLE001
                result["runtime_errors"] += 1
                result["first_difference"] = result["first_difference"] or {
                    "index": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) != 2:
            continue
        p_final, *p_trace = outputs["parent"]
        c_final, *c_trace = outputs["candidate"]
        expected = converted["output"].astype(bool)
        result["parent_gold"] += int(np.array_equal(p_final > 0, expected))
        result["candidate_gold"] += int(np.array_equal(c_final > 0, expected))
        result["final_raw_equal"] += int(raw_equal(p_final, c_final))
        for model_label, final, trace in (
            ("parent", p_final, p_trace),
            ("candidate", c_final, c_trace),
        ):
            result[f"{model_label}_nonfinite"]["output"] += int(
                final.size - np.count_nonzero(np.isfinite(final))
            )
            for name, value in zip(TRACE, trace, strict=True):
                result[f"{model_label}_nonfinite"][name] += int(
                    value.size - np.count_nonzero(np.isfinite(value))
                )
        different = []
        for name, left, right in zip(TRACE, p_trace, c_trace, strict=True):
            equal = raw_equal(left, right)
            result["trace_raw_equal"][name] += int(equal)
            if not equal:
                different.append(name)
        if not raw_equal(p_final, c_final) or different:
            result["first_difference"] = result["first_difference"] or {
                "index": index,
                "final_raw_equal": raw_equal(p_final, c_final),
                "different_trace": different,
            }
    valid = result["valid"]
    result["pass_through"] = bool(
        valid == len(cases)
        and result["final_raw_equal"] == valid
        and all(count == valid for count in result["trace_raw_equal"].values())
        and result["runtime_errors"] == 0
        and result["parent_nonfinite"]["output"] == 0
        and result["candidate_nonfinite"]["output"] == 0
        and result["candidate_nonfinite"]["Gv"] == 0
        and result["candidate_nonfinite"]["Gh"] == 0
    )
    return result


def isolated_g_model(data: bytes) -> bytes:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {value.name: value for value in list(inferred.graph.value_info) + list(inferred.graph.output)}
    nodes = [copy.deepcopy(model.graph.node[index]) for index in (22, 23)]
    needed = {name for node in nodes for name in node.input if name != "input"}
    initializers = [copy.deepcopy(item) for item in model.graph.initializer if item.name in needed]
    graph = helper.make_graph(
        nodes,
        "task066_review212_isolated_g",
        [copy.deepcopy(model.graph.input[0])],
        [copy.deepcopy(typed["Gv"]), copy.deepcopy(typed["Gh"])],
        initializer=initializers,
    )
    isolated = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(item) for item in model.opset_import],
        ir_version=model.ir_version,
    )
    onnx.checker.check_model(isolated, full_check=True)
    return isolated.SerializeToString()


def directed_inputs() -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    rows.append(("all_positive_zero", np.zeros((1, 10, 30, 30), dtype=np.float32)))

    vertical_max = np.zeros((1, 10, 30, 30), dtype=np.float32)
    vertical_max[0, 8, :20, 5] = 1
    vertical_max[0, 3, (3, 4), 5] = 1
    rows.append(("vertical_max_exact_2pow21", vertical_max))
    horizontal_max = np.zeros((1, 10, 30, 30), dtype=np.float32)
    horizontal_max[0, 8, 5, :20] = 1
    horizontal_max[0, 3, 5, (3, 4)] = 1
    rows.append(("horizontal_max_exact_2pow21", horizontal_max))

    misaligned = np.zeros((1, 10, 30, 30), dtype=np.float32)
    misaligned[0, 8, 0, 0] = 1
    misaligned[0, 3, 1, 2] = 1
    rows.append(("both_zero_misaligned", misaligned))

    for h in range(20):
        for r in range(20):
            x = np.zeros((1, 10, 30, 30), dtype=np.float32)
            w = (7 * h + 3 * r) % 20
            x[0, 8, h, w] = 1
            x[0, 3, r, w] = 1
            rows.append((f"vertical_basis_h{h}_r{r}", x))
            y = np.zeros((1, 10, 30, 30), dtype=np.float32)
            row = (5 * h + 11 * r) % 20
            y[0, 8, row, h] = 1
            y[0, 3, row, r] = 1
            rows.append((f"horizontal_basis_w{h}_r{r}", y))
    return rows


def directed_g_audit(parent_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    p_model = isolated_g_model(parent_data)
    c_model = isolated_g_model(candidate_data)
    cases = directed_inputs()
    modes = {}
    for disable, threads, label in CONFIGS:
        p_session = make_session(p_model, disable, threads)
        c_session = make_session(c_model, disable, threads)
        raw_counts = {"Gv": 0, "Gh": 0}
        nonfinite = {"Gv": 0, "Gh": 0}
        signed_zero_differences = {"Gv": 0, "Gh": 0}
        first_difference = None
        extrema = {"Gv_max": 0.0, "Gh_max": 0.0}
        for name, x in cases:
            p_values = [np.asarray(v) for v in p_session.run(None, {"input": x})]
            c_values = [np.asarray(v) for v in c_session.run(None, {"input": x})]
            for output_name, left, right in zip(("Gv", "Gh"), p_values, c_values, strict=True):
                equal = raw_equal(left, right)
                raw_counts[output_name] += int(equal)
                nonfinite[output_name] += int(right.size - np.count_nonzero(np.isfinite(right)))
                extrema[f"{output_name}_max"] = max(
                    extrema[f"{output_name}_max"], float(np.max(right))
                )
                if np.array_equal(left, right) and not equal and np.all(left == 0) and np.all(right == 0):
                    signed_zero_differences[output_name] += 1
                if not equal and first_difference is None:
                    first_difference = {"case": name, "output": output_name}
        modes[label] = {
            "cases": len(cases),
            "raw_equal": raw_counts,
            "nonfinite": nonfinite,
            "signed_zero_raw_differences": signed_zero_differences,
            "extrema": extrema,
            "first_difference": first_difference,
            "pass": bool(
                all(count == len(cases) for count in raw_counts.values())
                and not any(nonfinite.values())
                and first_difference is None
            ),
        }
    return {
        "case_families": (
            "all-zero, misaligned zero, exact max-bound vertical/horizontal, "
            "and all 20x20 cyan/green basis-coordinate pairs per orientation"
        ),
        "modes": modes,
        "pass": all(row["pass"] for row in modes.values()),
    }


def runtime_shape_truth(data: bytes, disable: bool) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    case = scoring.convert_to_numpy(known_cases()[0])
    assert case is not None
    session = make_session(traced.SerializeToString(), disable, 1)
    outputs = session.run(names, {session.get_inputs()[0].name: case["input"]})
    mismatches = []
    for name, output in zip(names, outputs, strict=True):
        expected = typed[name]
        actual = np.asarray(output)
        actual_dtype = helper.np_dtype_to_tensor_dtype(actual.dtype)
        if tensor_shape(expected) != list(actual.shape) or expected.type.tensor_type.elem_type != actual_dtype:
            mismatches.append(
                {
                    "name": name,
                    "declared_shape": tensor_shape(expected),
                    "runtime_shape": list(actual.shape),
                    "declared_dtype": int(expected.type.tensor_type.elem_type),
                    "runtime_dtype": int(actual_dtype),
                }
            )
    return {
        "optimization": "disable_all" if disable else "default",
        "node_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    parent_data = PARENT.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    assert sha256(parent_data) == EXPECTED_PARENT_SHA
    assert sha256(candidate_data) == EXPECTED_CANDIDATE_SHA

    delta = graph_delta(parent_data, candidate_data)
    selector = selector_proof(parent_data)
    inherited = inherited_geometry()
    assert delta["whitelist_exact"] and selector["pass"] and inherited["pass"]

    parent_trace = traced_model(parent_data, TRACE)
    candidate_trace = traced_model(candidate_data, TRACE)
    known = known_cases()
    fresh = [(seed, fresh_cases(seed, count)) for seed, count in FRESH]
    evaluations: dict[str, Any] = {
        "known": {},
        "fresh": [{"seed": seed, "count": len(cases), "modes": {}} for seed, cases in fresh],
    }
    evaluation_rows = []
    print(f"known={len(known)} fresh={[len(cases) for _, cases in fresh]}", flush=True)
    for disable, threads, label in CONFIGS:
        p_session = make_session(parent_trace, disable, threads)
        c_session = make_session(candidate_trace, disable, threads)
        row = evaluate_pair(p_session, c_session, known)
        evaluations["known"][label] = row
        evaluation_rows.append(row)
        print(
            f"known {label}: gold={row['candidate_gold']}/{row['valid']} "
            f"raw={row['final_raw_equal']} Gv={row['trace_raw_equal']['Gv']} "
            f"Gh={row['trace_raw_equal']['Gh']}",
            flush=True,
        )
        for index, (seed, cases) in enumerate(fresh):
            row = evaluate_pair(p_session, c_session, cases)
            evaluations["fresh"][index]["modes"][label] = row
            evaluation_rows.append(row)
            print(
                f"fresh {seed} {label}: gold={row['candidate_gold']}/{row['valid']} "
                f"raw={row['final_raw_equal']} Gv={row['trace_raw_equal']['Gv']} "
                f"Gh={row['trace_raw_equal']['Gh']}",
                flush=True,
            )

    profiles = {
        "parent": official_profile(parent_data, "task066_parent_review212"),
        "candidate": official_profile(candidate_data, "task066_candidate_review212"),
    }
    static = static_audit(candidate_data)
    directed = directed_g_audit(parent_data, candidate_data)
    shapes = [runtime_shape_truth(candidate_data, disable) for disable in (True, False)]
    summary = {
        "strict_lower": profiles["candidate"]["cost"] < profiles["parent"]["cost"],
        "cost_delta": profiles["parent"]["cost"] - profiles["candidate"]["cost"],
        "score_gain": math.log(profiles["parent"]["cost"] / profiles["candidate"]["cost"]),
        "graph_delta_exact": delta["whitelist_exact"],
        "selector_reconstruction_exact": selector["pass"],
        "inherited_1861056_support_proof_valid": inherited["pass"],
        "directed_Gv_Gh_raw_exact": directed["pass"],
        "static_pass": static["pass"],
        "truthful_shapes": all(row["truthful"] for row in shapes),
        "known_gold_four_configs": all(
            row["candidate_gold"] == row["valid"] for row in evaluations["known"].values()
        ),
        "all_sampled_raw_pass_through": all(row["pass_through"] for row in evaluation_rows),
        "runtime_errors_total": sum(row["runtime_errors"] for row in evaluation_rows),
        "final_nonfinite_total": sum(
            row[model]["output"]
            for row in evaluation_rows
            for model in ("parent_nonfinite", "candidate_nonfinite")
        ),
        "candidate_nonfinite_totals": {
            name: sum(row["candidate_nonfinite"][name] for row in evaluation_rows)
            for name in ("Gv", "Gh", "Gf", "selF", "selLog", "selQ", "ti", "output")
        },
    }
    summary["pass"] = bool(
        summary["strict_lower"]
        and summary["graph_delta_exact"]
        and summary["selector_reconstruction_exact"]
        and summary["inherited_1861056_support_proof_valid"]
        and summary["directed_Gv_Gh_raw_exact"]
        and summary["static_pass"]
        and summary["truthful_shapes"]
        and summary["known_gold_four_configs"]
        and summary["all_sampled_raw_pass_through"]
        and summary["runtime_errors_total"] == 0
        and summary["final_nonfinite_total"] == 0
    )
    result = {
        "parent": {"path": str(PARENT.relative_to(ROOT)), "sha256": sha256(parent_data)},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": sha256(candidate_data)},
        "profiles": profiles,
        "graph_delta": delta,
        "selector_proof": selector,
        "inherited_geometry_proof": inherited,
        "directed_Gv_Gh_audit": directed,
        "static": static,
        "runtime_shapes": shapes,
        "evaluations": evaluations,
        "summary": summary,
    }
    print("AUDIT_SUMMARY")
    print(json.dumps(safe(summary), indent=2))
    assert summary["pass"]


if __name__ == "__main__":
    main()
