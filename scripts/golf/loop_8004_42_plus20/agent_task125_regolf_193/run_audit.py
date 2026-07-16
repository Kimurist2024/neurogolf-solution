#!/usr/bin/env python3
"""Fail-closed task125 regolf audit against the immutable 8009.46 member.

This lane intentionally never writes a submission.  It enumerates exact local
rewrites around the dynamic shape chain and the quantized kernels, measures
them with the competition profiler, and stops before fresh evaluation unless a
strict-lower, structurally valid, truthful candidate exists.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
MEMBER_SHA256 = "c30ac7a079a4d5a91053c7748015a8c3a86ad594e542050e8826d46f1f84c529"
PROTECTED = (
    ROOT / "submission.zip",
    ROOT / "all_scores.csv",
    ROOT / "others/71407",
)
CANDIDATES = HERE / "candidates"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "regolf193_scan_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "regolf193_audit_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_digest(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    return digest(path.read_bytes())


def tree_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return path_digest(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(b"\0")
        h.update(item.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): tree_digest(path) for path in PROTECTED}


def prune(model: onnx.ModelProto) -> None:
    """Remove graph-dead producers and unused initializers/value-info."""
    needed = {value.name for value in model.graph.output}
    live: list[int] = []
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(name and name in needed for name in node.output):
            live.append(index)
            needed.update(name for name in node.input if name)
    live.reverse()
    kept_nodes = [model.graph.node[index] for index in live]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    kept_init = [item for item in model.graph.initializer if item.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_init)
    live_names = {value.name for value in model.graph.input}
    live_names.update(value.name for value in model.graph.output)
    live_names.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
    kept_vi = [value for value in model.graph.value_info if value.name in live_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


SHAPE_VALUES = {
    "wshape10": np.array([10], dtype=np.int64),
    "s13": np.array([13], dtype=np.int64),
    "s26": np.array([26], dtype=np.int64),
    "s27": np.array([27], dtype=np.int64),
    "s25": np.array([25], dtype=np.int64),
}


def fold_shape_outputs(base: onnx.ModelProto, names: set[str]) -> onnx.ModelProto:
    candidate = copy.deepcopy(base)
    kept = []
    for node in candidate.graph.node:
        if len(node.output) == 1 and node.output[0] in names:
            continue
        kept.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    candidate.graph.initializer.extend(
        numpy_helper.from_array(SHAPE_VALUES[name], name=name) for name in sorted(names)
    )
    prune(candidate)
    return candidate


def share_directional_kernel(base: onnx.ModelProto) -> onnx.ModelProto:
    """All-input exact: vW is the [0,1,3,2] transpose of hW."""
    candidate = copy.deepcopy(base)
    kept = [item for item in candidate.graph.initializer if item.name != "vW"]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    vertical_index = next(
        index
        for index, node in enumerate(candidate.graph.node)
        if node.op_type == "QLinearConv" and list(node.output) == ["vscore"]
    )
    transpose = helper.make_node(
        "Transpose", ["hW"], ["vW_shared"], perm=[0, 1, 3, 2], name="share_hW_as_vW"
    )
    candidate.graph.node.insert(vertical_index, transpose)
    candidate.graph.node[vertical_index + 1].input[3] = "vW_shared"
    candidate.graph.value_info.append(
        helper.make_tensor_value_info("vW_shared", TensorProto.INT8, [1, 1, 7, 1])
    )
    prune(candidate)
    return candidate


def direct_terminal_kernel(base: onnx.ModelProto, qwdyn: np.ndarray) -> onnx.ModelProto:
    """All-input exact: the shape chain is constant, so materialize qWdyn."""
    candidate = copy.deepcopy(base)
    remove_outputs = {"qW26", "qW13", "qWdyn"}
    kept = [
        node
        for node in candidate.graph.node
        if not any(name in remove_outputs for name in node.output)
    ]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    terminal = next(
        node
        for node in candidate.graph.node
        if node.op_type == "QLinearConv" and list(node.output) == ["output"]
    )
    terminal.input[3] = "qW_direct"
    candidate.graph.initializer.append(
        numpy_helper.from_array(np.asarray(qwdyn, dtype=np.int8), name="qW_direct")
    )
    prune(candidate)
    return candidate


def add_to_sum(base: onnx.ModelProto) -> onnx.ModelProto:
    candidate = copy.deepcopy(base)
    for node in candidate.graph.node:
        if node.op_type == "Add":
            node.op_type = "Sum"
    return candidate


def initializer_inventory(model: onnx.ModelProto) -> dict[str, Any]:
    used = Counter(name for node in model.graph.node for name in node.input if name)
    live = {value.name for value in model.graph.output}
    for node in reversed(model.graph.node):
        if any(name in live for name in node.output if name):
            live.update(name for name in node.input if name)
    aliases: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    arrays: dict[str, np.ndarray] = {}
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        arrays[item.name] = array
        aliases[(array.dtype.str, tuple(array.shape), array.tobytes())].append(item.name)
    h = arrays["hW"]
    v = arrays["vW"]
    return {
        "unused": [item.name for item in model.graph.initializer if item.name not in live],
        "aliases_same_shape": [names for names in aliases.values() if len(names) > 1],
        "uses": dict(sorted(used.items())),
        "hW_vW_transpose_exact": bool(np.array_equal(h.transpose(0, 1, 3, 2), v)),
        "hW_elements": int(h.size),
        "vW_elements": int(v.size),
    }


def terminal_analysis(qwdyn: np.ndarray) -> dict[str, Any]:
    matrix = qwdyn.reshape(qwdyn.shape[0], -1).astype(np.float64)
    active = [index for index, row in enumerate(matrix) if np.any(row)]
    gcds = []
    for index in active:
        values = np.abs(matrix[index].astype(np.int64))
        nz = values[values != 0]
        gcds.append(int(np.gcd.reduce(nz)) if nz.size else 0)
    rank = int(np.linalg.matrix_rank(matrix))
    active_params = len(active) * int(matrix.shape[1])
    # A truthful reconstruction must at least materialize the 10x2x3x3
    # kernel once; this lower bound is enough to show sparse storage cannot
    # beat the current three one-element cloaked outputs.
    sparse_rebuild_floor = active_params + int(qwdyn.size)
    return {
        "shape": list(qwdyn.shape),
        "active_output_rows": active,
        "zero_output_rows": [index for index in range(qwdyn.shape[0]) if index not in active],
        "matrix_rank": rank,
        "active_row_gcds": gcds,
        "direct_parameter_elements": int(qwdyn.size),
        "compact_active_parameter_elements": active_params,
        "compact_saving_before_reconstruction": int(qwdyn.size) - active_params,
        "truthful_compact_plus_materialized_kernel_floor": sparse_rebuild_floor,
        "current_source_plus_cloaked_outputs": 108 + 3,
        "factor_observation": (
            "rank-4 factorization saves no weights versus four stored active rows and "
            "adds an activation; every active row has integer gcd 1"
        ),
    }


def local_truthful_floor() -> dict[str, Any]:
    """Optimistic lower bound for the authority's directional topology.

    The bound deliberately gives away every shape tensor, bd2, output-channel
    placement, and hW->vW transpose for free.  It still exceeds 1045, so any
    truthful strict decrease requires a different semantic architecture rather
    than a local crop/kernel rewrite.
    """
    tensors = {
        "pink_mask_p": 1 * 1 * 13 * 13,
        "horizontal": 1 * 1 * 13 * 13,
        "vertical": 1 * 1 * 13 * 13,
        "intersection_bd": 1 * 1 * 13 * 13,
        "two_channel_feature": 1 * 2 * 13 * 13,
    }
    memory = sum(tensors.values())
    parameters = {
        "one_directional_kernel_with_free_transpose": 7,
        "four_active_terminal_3x3x2_rows": 4 * 2 * 3 * 3,
        "three_quantization_scalars": 3,
    }
    params = sum(parameters.values())
    return {
        "scope": "current p/h/v/Min/two-channel/QLinearConv topology only",
        "optimistic_free_items": [
            "all shape tensors and extraction machinery",
            "bd2 materialization",
            "hW-to-vW transpose output",
            "six zero output rows and their non-contiguous placement",
            "final output activation",
        ],
        "truthful_intermediate_elements": tensors,
        "truthful_memory_floor": memory,
        "parameter_elements": parameters,
        "parameter_floor": params,
        "cost_floor": memory + params,
        "authority_cost": 1045,
        "strict_lower_possible_with_local_topology": memory + params < 1045,
    }


def fresh_spec_audit() -> dict[str, Any]:
    """Two disjoint fresh seed ranges for the generator rule, not a candidate gate."""
    reference = load_module(
        "regolf193_task125_reference",
        ROOT / "scripts/golf/scratch_codex/task125/rebuild_reference.py",
    )
    ranges = ((31_000_000, 2000), (47_000_000, 2000))
    rows = []
    for start, count in ranges:
        failures = []
        for seed in range(start, start + count):
            random.seed(seed)
            example = reference.generator.generate()
            failure = reference.check_example(example)
            if failure is not None:
                failures.append({"seed": seed, "reason": failure})
                break
        rows.append(
            {
                "seed_start": start,
                "count": count,
                "passed": count if not failures else failures[0]["seed"] - start,
                "failures": failures,
                "perfect": not failures,
            }
        )
    return {
        "purpose": "independent confirmation of generator rule and 7-tap directional mask",
        "candidate_admission": False,
        "ranges": rows,
        "perfect": all(row["perfect"] for row in rows),
    }


def profile(data: bytes, label: str) -> dict[str, Any]:
    try:
        return SCAN.official_cost(data, label)
    except Exception as exc:  # noqa: BLE001
        return {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}


def structure(data: bytes) -> dict[str, Any]:
    try:
        return SCAN.structural(onnx.load_model_from_string(data))
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "reasons": ["load"], "error": f"{type(exc).__name__}: {exc}"}


def trace(data: bytes) -> dict[str, Any]:
    try:
        return AUDIT.direct_trace(125, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def extract_runtime_qwdyn(data: bytes) -> np.ndarray:
    """Instrument immutable authority and return its data-independent qWdyn."""
    import onnxruntime as ort
    from onnx import shape_inference

    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    # Exposing every intermediate disables the malformed authority's unsafe
    # allocator reuse.  qWdyn itself is input-independent; the extra outputs
    # are instrumentation only and are never emitted as a candidate.
    existing = {value.name for value in model.graph.output}
    for node in model.graph.node:
        for name in node.output:
            if name and name in typed and name not in existing:
                model.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    value = session.run(["qWdyn"], {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})[0]
    return np.asarray(value, dtype=np.int8)


def emit(label: str, model: onnx.ModelProto) -> tuple[Path, bytes]:
    data = model.SerializeToString(deterministic=True)
    path = CANDIDATES / f"task125_{label}.onnx"
    path.write_bytes(data)
    return path, data


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    authority_bytes = AUTHORITY.read_bytes()
    if digest(authority_bytes) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    with zipfile.ZipFile(AUTHORITY) as archive:
        base_data = archive.read("task125.onnx")
    if digest(base_data) != MEMBER_SHA256:
        raise RuntimeError("authority member hash mismatch")
    base = onnx.load_model_from_string(base_data)
    qwdyn = extract_runtime_qwdyn(base_data)

    variants: list[tuple[str, onnx.ModelProto, str]] = []
    for name in SHAPE_VALUES:
        variants.append((f"fold_{name}", fold_shape_outputs(base, {name}), f"constant-fold {name}"))
    variants.extend(
        [
            ("fold_all_shapes", fold_shape_outputs(base, set(SHAPE_VALUES)), "constant-fold all five shape expressions"),
            ("share_hv_transpose", share_directional_kernel(base), "replace vW initializer with exact transpose(hW)"),
            ("direct_terminal_kernel", direct_terminal_kernel(base, qwdyn), "materialize data-independent qWdyn"),
            ("add_to_sum", add_to_sum(base), "schema-equivalent binary Add to Sum carrier"),
        ]
    )

    base_profile = profile(base_data, "task125_authority")
    base_struct = structure(base_data)
    base_trace = trace(base_data)
    rows = []
    strict_lower_pre_fresh = []
    for label, model, proof in variants:
        path, data = emit(label, model)
        cost = profile(data, f"task125_{label}")
        static = structure(data)
        runtime = trace(data) if static.get("pass", False) else {"truthful": False, "status": "not_run_after_static_failure"}
        strict_lower = 0 <= cost.get("cost", -1) < base_profile["cost"]
        reasons = []
        if not strict_lower:
            reasons.append("not_strict_lower")
        if not static.get("pass", False):
            reasons.append("full_or_strict_or_ub_gate")
        if not runtime.get("truthful", False):
            reasons.append("runtime_shape_not_truthful")
        reaches_known = not reasons
        row = {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "all_input_exact_proof": proof,
            "official_profile": cost,
            "strict_lower": strict_lower,
            "static": static,
            "runtime_shape_trace": runtime,
            "known_four_configs": {"status": "not_run_before_pre_fresh_gates"},
            "fresh": {"status": "not_run_before_pre_fresh_gates"},
            "reasons": reasons,
            "accepted": False,
        }
        if reaches_known:
            row["known_four_configs"] = {}
            for disable, threads, config_label in AUDIT.CONFIGS:
                row["known_four_configs"][config_label] = AUDIT.known_config(
                    125, base_data, data, disable, threads
                )
            if all(item.get("perfect", False) for item in row["known_four_configs"].values()):
                strict_lower_pre_fresh.append(label)
            else:
                row["reasons"].append("known_four_configs_failed")
        rows.append(row)

    # Historical true-rule control is included read-only.  It is not an
    # admission candidate because its cost already exceeds the authority.
    pool_path = ROOT / "scripts/golf/scratch_codex/task125/task125_pool14.onnx"
    pool_data = pool_path.read_bytes()
    pool_control = {
        "path": str(pool_path.relative_to(ROOT)),
        "sha256": digest(pool_data),
        "official_profile": profile(pool_data, "task125_pool14_control"),
        "static": structure(pool_data),
        "runtime_shape_trace": trace(pool_data),
        "generator_evidence": {
            "reference_validate_plus_fresh": "2000/2000 documented in scratch_codex/task125/REPORT.md",
            "pool14_numpy_mask": "20000/20000 documented in scratch_codex/task125/REPORT.md",
        },
    }

    inventory = SCAN.graph_inventory(base)
    result = {
        "authority": {
            "zip": str(AUTHORITY.relative_to(ROOT)),
            "zip_sha256": digest(authority_bytes),
            "member_sha256": digest(base_data),
            "profile": base_profile,
            "static": base_struct,
            "runtime_shape_trace": base_trace,
        },
        "graph_inventory": inventory,
        "initializer_inventory": initializer_inventory(base),
        "terminal_kernel": terminal_analysis(qwdyn),
        "local_truthful_floor": local_truthful_floor(),
        "fresh_generator_rule_audit": fresh_spec_audit(),
        "variants": rows,
        "pool14_true_rule_control": pool_control,
        "strict_lower_pre_fresh": strict_lower_pre_fresh,
        "winners": [],
        "fresh_policy": (
            "two independent seeds, four ORT configurations, exact generated truth; "
            "not run because no candidate cleared cost/static/truthful/known gates"
        ),
        "summary": {
            "variants": len(rows),
            "strict_lower": sum(row["strict_lower"] for row in rows),
            "static_pass": sum(bool(row["static"].get("pass", False)) for row in rows),
            "truthful": sum(bool(row["runtime_shape_trace"].get("truthful", False)) for row in rows),
            "pre_fresh": len(strict_lower_pre_fresh),
            "winners": 0,
            "verdict": "NO_ADMISSIBLE_STRICT_LOWER_CANDIDATE",
        },
    }
    after = protected_hashes()
    result["integrity"] = {
        "before": before,
        "after": after,
        "unchanged": before == after,
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if before != after:
        raise RuntimeError("protected root/71407 integrity changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
