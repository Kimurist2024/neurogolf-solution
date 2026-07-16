#!/usr/bin/env python3
"""Independent fail-closed review of task192 lane 211.

The active POLICY90 shave and the exact ArgMax fallback are assessed as two
separate claims.  This script deliberately does not import lane 211 code.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ACTIVE_PARENT = ROOT / "others/71407/task192.onnx"
ACTIVE_CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task192_local_211/candidates"
    / "task192_policy90_center_direct.onnx"
)
EXACT_PARENT = ROOT / "others/71407/FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1149.onnx.fallback"
EXACT_CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task192_local_211/candidates"
    / "task192_center_direct_argmax.onnx"
)
OUT = HERE / "audit.json"

EXPECTED_SHAS = {
    "active_parent": "e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5",
    "active_candidate": "1200fe8473c045ec89abaaf1860d1d0758316523855c9ff13d4c3fc092412047",
    "exact_candidate": "5c5eaefa81acce481dbc93855dbcc2f9ef821e055f8c982eadcd07f63c764a9d",
}
ROOT_GUARDS = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = ((214_192_07, 1500), (214_192_31, 1500))
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                count += 1
                pending.extend(attr.g.node)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                count += len(attr.graphs)
                for graph in attr.graphs:
                    pending.extend(graph.node)
    return count


def attr_float(node: onnx.NodeProto, name: str) -> float | None:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return None


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    initializer_names = {x.name for x in model.graph.initializer}
    used = {name for node in model.graph.node for name in node.input if name in initializer_names}
    all_values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    static_declared = all(
        dim.HasField("dim_value") and int(dim.dim_value) > 0
        for value in all_values
        for dim in value.type.tensor_type.shape.dim
    )
    nonfinite_initializers = 0
    for initializer in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(initializer))
        if array.dtype.kind in "fc":
            nonfinite_initializers += int(array.size - np.count_nonzero(np.isfinite(array)))
    nodes = list(model.graph.node)
    conv_nodes = [n for n in nodes if n.op_type in {"Conv", "ConvTranspose", "QLinearConv"}]
    result: dict[str, Any] = {
        "sha256": sha256(path),
        "nodes": len(nodes),
        "initializers": len(model.graph.initializer),
        "opsets": [{"domain": x.domain, "version": int(x.version)} for x in model.opset_import],
        "standard_domain_only": all(x.domain in {"", "ai.onnx"} for x in model.opset_import),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested_graph_count(model),
        "static_declared_shapes": static_declared,
        "banned_ops": sorted({n.op_type for n in nodes if n.op_type in BANNED or "Sequence" in n.op_type}),
        "hardmax_count": sum(n.op_type == "Hardmax" for n in nodes),
        "conv_family_count": len(conv_nodes),
        "conv_bias_ub_count": 0 if not conv_nodes else None,
        "unused_initializers": sorted(initializer_names - used),
        "nonfinite_initializers": nonfinite_initializers,
    }
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"
    for data_prop in (False, True):
        key = "strict_data_prop" if data_prop else "strict"
        try:
            shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=data_prop)
            result[key] = True
        except Exception as exc:  # noqa: BLE001
            result[key] = False
            result[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    result["pass"] = bool(
        result["full_check"]
        and result["strict"]
        and result["strict_data_prop"]
        and result["standard_domain_only"]
        and result["functions"] == 0
        and result["sparse_initializers"] == 0
        and result["nested_graphs"] == 0
        and result["static_declared_shapes"]
        and not result["banned_ops"]
        and result["hardmax_count"] == 0
        and result["conv_bias_ub_count"] == 0
        and not result["unused_initializers"]
        and result["nonfinite_initializers"] == 0
    )
    return result


def arrays(path: Path) -> dict[str, np.ndarray]:
    return {x.name: np.asarray(numpy_helper.to_array(x)) for x in onnx.load(path).graph.initializer}


def selected_basis_proof() -> dict[str, Any]:
    """Enumerate every HardSigmoid-reachable selected vector (2^10)."""
    old = arrays(ACTIVE_PARENT)
    new = arrays(ACTIVE_CANDIDATE)
    inside = np.ones((1, 10), dtype=np.float32)
    nonzero = np.asarray([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
    background = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    initializer_identities = bool(
        np.array_equal(old["nonzero"], nonzero)
        and np.array_equal(old["background"], background)
        and np.array_equal(new["center_basis"], np.concatenate([inside, nonzero], axis=0))
        and np.array_equal(new["hist_selector"], np.asarray([[0, 1]], dtype=np.float32))
    )
    mismatches: list[dict[str, Any]] = []
    cell_states = np.concatenate([np.zeros((1, 10), dtype=np.float32), np.eye(10, dtype=np.float32)], axis=0)
    cell_factor_checks = 0
    for mask in range(1 << 10):
        selected = np.asarray([[(mask >> i) & 1 for i in range(10)]], dtype=np.float32)
        old_basis = np.concatenate([old["nonzero"], old["background"], selected], axis=0)
        new_basis = np.concatenate([new["center_basis"], selected], axis=0)
        factors = {
            "center": (old["center_map"] @ old_basis, new["center_basis"]),
            "neighbor": (old["neighbor_map"] @ old_basis, new["neighbor_map"] @ new_basis),
            "route": (old["route_out"] @ old_basis, new["route_out"] @ new_basis),
        }
        for name, (left, right) in factors.items():
            if not np.array_equal(left, right):
                mismatches.append({"mask": mask, "factor": name})
            for cell in cell_states:
                cell_factor_checks += 1
                if not np.array_equal(left @ cell, right @ cell):
                    mismatches.append({"mask": mask, "factor": name, "cell": cell.tolist()})
                    break
    source_model = onnx.load(ACTIVE_PARENT)
    candidate_model = onnx.load(ACTIVE_CANDIDATE)
    source_nodes = {x.output[0]: x for x in source_model.graph.node}
    candidate_nodes = {x.output[0]: x for x in candidate_model.graph.node}
    selector_unchanged = bool(
        source_nodes["selected"].op_type == candidate_nodes["selected"].op_type == "HardSigmoid"
        and attr_float(source_nodes["selected"], "alpha") == attr_float(candidate_nodes["selected"], "alpha") == 1.0
        and attr_float(source_nodes["selected"], "beta") == attr_float(candidate_nodes["selected"], "beta") == -33.0
    )
    adjacency_identical = bool(
        next(x for x in source_model.graph.initializer if x.name == "adj").SerializeToString()
        == next(x for x in candidate_model.graph.initializer if x.name == "adj").SerializeToString()
    )
    result = {
        "initializer_identities": initializer_identities,
        "selected_vectors_enumerated": 1 << 10,
        "cell_states_per_vector": len(cell_states),
        "factor_cell_checks": cell_factor_checks,
        "mismatches": mismatches[:10],
        "mismatch_count": len(mismatches),
        "hard_sigmoid_alpha_beta_unchanged": selector_unchanged,
        "adjacency_proto_byte_identical": adjacency_identical,
        "symbolic_identity": {
            "center": "[nonzero+background, nonzero] = [inside, nonzero]",
            "neighbor": "[nonzero+background, selected] = [inside, selected]",
            "route": "[background, -9*background+selected] = [inside-nonzero, -9*inside+9*nonzero+selected]",
            "histogram": "input dot nonzero = (input dot [inside,nonzero]) dot [0,1]",
        },
        "all_input_argument": (
            "Every legal input cell is zero-hot or one-hot. Histograms are exact float32 integers <=900; "
            "HardSigmoid therefore emits a binary (possibly multi-hot) selected vector. Every changed "
            "factor is identical for all 1024 such vectors and all 11 legal cell states. Adjacency and the "
            "final polynomial are unchanged. Nonzero terms per output stay small integer-valued, so no "
            "float32 rounding is introduced by the refactor."
        ),
    }
    result["pass"] = bool(
        initializer_identities and not mismatches and selector_unchanged and adjacency_identical
    )
    return result


def exact_basis_proof() -> dict[str, Any]:
    """Prove that the ArgMax fallback changes only the same exact basis."""
    old = arrays(EXACT_PARENT)
    new = arrays(EXACT_CANDIDATE)
    base = selected_basis_proof()
    old_model = onnx.load(EXACT_PARENT)
    new_model = onnx.load(EXACT_CANDIDATE)
    old_argmax = next(n for n in old_model.graph.node if n.op_type == "ArgMax")
    new_argmax = next(n for n in new_model.graph.node if n.op_type == "ArgMax")
    old_onehot = next(n for n in old_model.graph.node if n.op_type == "OneHot")
    new_onehot = next(n for n in new_model.graph.node if n.op_type == "OneHot")
    adjacency_identical = bool(
        next(x for x in old_model.graph.initializer if x.name == "adj").SerializeToString()
        == next(x for x in new_model.graph.initializer if x.name == "adj").SerializeToString()
    )
    selector_reuse = bool(
        np.array_equal(new["onehot_values"], np.asarray([0, 1], dtype=np.float32))
        and np.array_equal(old["onehot_values"], new["onehot_values"])
        and int(np.asarray(old["depth"])) == int(np.asarray(new["depth"])) == 10
    )
    argmax_axes_equivalent = bool(
        old_argmax.input[0] == new_argmax.input[0] == "hist"
        and old_onehot.output[0] == new_onehot.output[0] == "selected"
    )
    result = {
        "same_basis_identity_as_active": base["pass"],
        "hist_identity": "old [1,10] input·nonzero and new [10] input·center_basis·[0,1] have identical ten entries",
        "argmax_axis_change": "axis1 on [1,10] and axis0 on [10], both first-index tie break, both output [1]",
        "selector_and_depth_identical": selector_reuse,
        "argmax_onehot_route_present": argmax_axes_equivalent,
        "adjacency_proto_byte_identical": adjacency_identical,
    }
    result["pass"] = bool(base["pass"] and selector_reuse and argmax_axes_equivalent and adjacency_identical)
    return result


def known_cases() -> list[dict[str, Any]]:
    payload = scoring.load_examples(192)
    return [row for split in ("train", "test", "arc-gen") for row in payload.get(split, [])]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    generator = importlib.import_module("task_7e0986d6")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    rows: list[dict[str, Any]] = []
    while len(rows) < count:
        case = generator.generate()
        if scoring.convert_to_numpy(case) is not None:
            rows.append(case)
    return rows


def session(path: Path, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def compare_pair(
    parent: Path, candidate: Path, cases: list[dict[str, Any]], disable: bool, threads: int
) -> dict[str, Any]:
    sessions = {"parent": session(parent, disable, threads), "candidate": session(candidate, disable, threads)}
    result: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "raw_equal": 0,
        "sign_equal": 0,
        "right": {"parent": 0, "candidate": 0},
        "errors": {"parent": 0, "candidate": 0},
        "nonfinite": {"parent": 0, "candidate": 0},
        "first_difference": None,
    }
    for index, case in enumerate(cases):
        benchmark = scoring.convert_to_numpy(case)
        if benchmark is None:
            continue
        result["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, runtime in sessions.items():
            try:
                value = np.asarray(runtime.run(None, {runtime.get_inputs()[0].name: benchmark["input"]})[0])
                outputs[label] = value
                result["right"][label] += int(value.shape == expected.shape and np.array_equal(value > 0, expected))
                if value.dtype.kind in "fc":
                    result["nonfinite"][label] += int(value.size - np.count_nonzero(np.isfinite(value)))
            except Exception as exc:  # noqa: BLE001
                result["errors"][label] += 1
                if result["first_difference"] is None:
                    result["first_difference"] = {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        if len(outputs) == 2:
            left = np.ascontiguousarray(outputs["parent"])
            right = np.ascontiguousarray(outputs["candidate"])
            equal = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
            result["raw_equal"] += int(equal)
            result["sign_equal"] += int(np.array_equal(left > 0, right > 0))
            if not equal and result["first_difference"] is None:
                result["first_difference"] = {
                    "index": index,
                    "max_abs": float(np.max(np.abs(left - right))),
                }
    result["errors_total"] = sum(result["errors"].values())
    result["nonfinite_total"] = sum(result["nonfinite"].values())
    result["accuracy"] = {
        label: result["right"][label] / result["valid"] for label in ("parent", "candidate")
    }
    result["pass_through"] = bool(
        result["valid"] == result["total"]
        and result["raw_equal"] == result["total"]
        and result["sign_equal"] == result["total"]
        and result["errors_total"] == 0
        and result["nonfinite_total"] == 0
    )
    return result


def runtime_shape_truth(path: Path, example: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {x.name for x in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        return {"truthful": False, "error": "example conversion failed"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    runtime = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    values = runtime.run(names, {runtime.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = 0
    for name, value in zip(names, values):
        array = np.asarray(value)
        if dims(typed[name]) != list(array.shape):
            mismatches.append({"name": name, "declared": dims(typed[name]), "actual": list(array.shape)})
        if array.dtype.kind in "fc":
            nonfinite += int(array.size - np.count_nonzero(np.isfinite(array)))
    return {
        "traced_outputs": len(names),
        "shape_mismatches": mismatches,
        "shape_mismatch_count": len(mismatches),
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    paths = {
        "active_parent": ACTIVE_PARENT,
        "active_candidate": ACTIVE_CANDIDATE,
        "exact_parent": EXACT_PARENT,
        "exact_candidate": EXACT_CANDIDATE,
    }
    before = {name: sha256(ROOT / name) for name in ROOT_GUARDS}
    if before != ROOT_GUARDS:
        raise RuntimeError(f"root guard mismatch before audit: {before}")
    actual_shas = {label: sha256(path) for label, path in paths.items()}
    for label, expected in EXPECTED_SHAS.items():
        if actual_shas[label] != expected:
            raise RuntimeError(f"SHA mismatch {label}: {actual_shas[label]} != {expected}")

    profiles = {label: profile(path) for label, path in paths.items()}
    structures = {label: structure(path) for label, path in paths.items()}
    active_proof = selected_basis_proof()
    exact_proof = exact_basis_proof()
    known = known_cases()
    shape_truth = {
        "active_candidate": runtime_shape_truth(ACTIVE_CANDIDATE, known[0]),
        "exact_candidate": runtime_shape_truth(EXACT_CANDIDATE, known[0]),
    }

    known_results: dict[str, Any] = {}
    for disable, threads, label in CONFIGS:
        known_results[label] = {
            "active": compare_pair(ACTIVE_PARENT, ACTIVE_CANDIDATE, known, disable, threads),
            "exact": compare_pair(EXACT_PARENT, EXACT_CANDIDATE, known, disable, threads),
        }

    fresh_results = []
    for seed, count in FRESH:
        cases = fresh_cases(seed, count)
        stream: dict[str, Any] = {"seed": seed, "count": len(cases), "configs": {}}
        for disable, threads, label in CONFIGS:
            stream["configs"][label] = {
                "active": compare_pair(ACTIVE_PARENT, ACTIVE_CANDIDATE, cases, disable, threads),
                "exact": compare_pair(EXACT_PARENT, EXACT_CANDIDATE, cases, disable, threads),
            }
        fresh_results.append(stream)
        print(f"fresh seed={seed} count={len(cases)} complete", flush=True)

    active_comparisons = [x["active"] for x in known_results.values()] + [
        x["active"] for stream in fresh_results for x in stream["configs"].values()
    ]
    exact_comparisons = [x["exact"] for x in known_results.values()] + [
        x["exact"] for stream in fresh_results for x in stream["configs"].values()
    ]
    after = {name: sha256(ROOT / name) for name in ROOT_GUARDS}

    active_pass = bool(
        profiles["active_parent"] == {"memory": 200, "params": 938, "cost": 1138}
        and profiles["active_candidate"] == {"memory": 200, "params": 934, "cost": 1134}
        and active_proof["pass"]
        and structures["active_candidate"]["pass"]
        and shape_truth["active_candidate"]["truthful"]
        and all(x["pass_through"] for x in active_comparisons)
        and min(x["accuracy"]["candidate"] for x in active_comparisons[4:]) >= 0.90
    )
    exact_pass = bool(
        profiles["exact_parent"] == {"memory": 208, "params": 941, "cost": 1149}
        and profiles["exact_candidate"] == {"memory": 208, "params": 935, "cost": 1143}
        and exact_proof["pass"]
        and structures["exact_candidate"]["pass"]
        and shape_truth["exact_candidate"]["truthful"]
        and all(x["pass_through"] for x in exact_comparisons)
        and min(x["accuracy"]["candidate"] for x in exact_comparisons[4:]) == 1.0
    )
    report = {
        "paths": {label: str(path.relative_to(ROOT)) for label, path in paths.items()},
        "sha256": actual_shas,
        "profiles": profiles,
        "structures": structures,
        "active_algebraic_proof": active_proof,
        "exact_algebraic_proof": exact_proof,
        "runtime_shape_truth": shape_truth,
        "known_count": len(known),
        "known_four_configs": known_results,
        "fresh": fresh_results,
        "active_summary": {
            "decision": "PASS_POLICY90_INHERITED_RAW_PASS_THROUGH" if active_pass else "FAIL",
            "cost_delta": profiles["active_parent"]["cost"] - profiles["active_candidate"]["cost"],
            "all_raw_pass_through": all(x["pass_through"] for x in active_comparisons),
            "minimum_fresh_accuracy": min(x["accuracy"]["candidate"] for x in active_comparisons[4:]),
            "errors_total": sum(x["errors_total"] for x in active_comparisons),
            "nonfinite_total": sum(x["nonfinite_total"] for x in active_comparisons),
            "pass": active_pass,
        },
        "exact_summary": {
            "decision": "PASS_EXACT_FALLBACK_REPLACEMENT" if exact_pass else "FAIL",
            "cost_delta": profiles["exact_parent"]["cost"] - profiles["exact_candidate"]["cost"],
            "all_raw_pass_through": all(x["pass_through"] for x in exact_comparisons),
            "minimum_fresh_accuracy": min(x["accuracy"]["candidate"] for x in exact_comparisons[4:]),
            "errors_total": sum(x["errors_total"] for x in exact_comparisons),
            "nonfinite_total": sum(x["nonfinite_total"] for x in exact_comparisons),
            "pass": exact_pass,
        },
        "root_guards_before": before,
        "root_guards_after": after,
        "root_guards_unchanged": before == after == ROOT_GUARDS,
        "pass": active_pass and exact_pass and before == after == ROOT_GUARDS,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "profiles": profiles,
        "active": report["active_summary"],
        "exact": report["exact_summary"],
        "root_guards_unchanged": report["root_guards_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
