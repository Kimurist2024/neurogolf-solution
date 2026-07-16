#!/usr/bin/env python3
"""Fail-closed normal-POLICY90 review of task071 actual-lower SHA 6cc540e9."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DISCOVERY = ROOT / (
    "scripts/golf/loop_8004_42_plus20/agent_mid20d_88/"
    "audit/actual_lower_four_config.json"
)
SOURCE = ROOT / "others/2/7616/task071_rebuilt_cost186.onnx"
DUPLICATE_SOURCES = (
    SOURCE,
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task071_r01_static186.onnx",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task071_r01_static186.onnx",
)
EXPECTED_CANDIDATE_SHA256 = "6cc540e94a37ca160273d7cb471492913943c9bf966d60012d6944b37773c68e"
BASELINE_ZIP = ROOT / "submission_base_8005.17.zip"
BASELINE_ZIP_SHA256 = "c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04"
EXPECTED_IO = [1, 10, 30, 30]
GIANT_EINSUM_MIN_INPUTS = 15
GIANT_INITIALIZER_MIN_ELEMENTS = 10_000
QUARANTINE = HERE / "quarantine/task071_sha6cc540_REJECT_SHAPE_CLOAK.onnx"
OUTPUT = HERE / "evidence.json"
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("default_threads1", "default", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads4", "default", 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                count += 1
                pending.extend(attribute.g.node)
            elif attribute.type == AttributeProto.GRAPHS:
                count += len(attribute.graphs)
                for graph in attribute.graphs:
                    pending.extend(graph.node)
    return count


def profile_bytes(data: bytes, name: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"task071_276_{name}_", dir="/tmp") as work:
        path = Path(work) / "task071.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def static_memory(inferred: onnx.ModelProto) -> int:
    typed = {
        value.name: value
        for value in [*inferred.graph.value_info, *inferred.graph.output]
    }
    graph_outputs = {value.name for value in inferred.graph.output}
    total = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in graph_outputs:
                continue
            value = typed[name]
            shape = dims(value)
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            total += math.prod(shape) * np.dtype(dtype).itemsize
    return int(total)


def static_audit(model: onnx.ModelProto) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    nonstatic = [
        name for name, value in typed.items()
        if not dims(value) or any(dim is None or dim <= 0 for dim in dims(value))
    ]
    missing_node_outputs = [
        name for node in inferred.graph.node for name in node.output
        if name and name not in typed
    ]
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializers = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": (
                int(np.count_nonzero(~np.isfinite(array)))
                if array.dtype.kind in "fc" else 0
            ),
        }
        for name, array in arrays.items()
    }
    giant_initializers = [
        {"name": name, "elements": item["elements"]}
        for name, item in initializers.items()
        if item["elements"] >= GIANT_INITIALIZER_MIN_ELEMENTS
    ]
    domains = sorted({
        domain
        for domain in [
            *(item.domain for item in model.opset_import),
            *(node.domain for node in model.graph.node),
        ]
        if domain not in ("", "ai.onnx")
    })
    banned = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in BANNED or "Sequence" in node.op_type
    })
    external = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    conv_bias_findings = check_conv_bias(model)
    gather_nodes = [node for node in model.graph.node if node.op_type == "Gather"]
    gather_data_sources = [node.input[0] for node in gather_nodes]
    explicit_lookup_ops = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND"}
    })
    no_lookup = bool(
        not explicit_lookup_ops
        and all(name == "input" for name in gather_data_sources)
        and not giant_initializers
    )
    graph_input_shape = dims(inferred.graph.input[0]) if len(inferred.graph.input) == 1 else None
    graph_output_shape = dims(inferred.graph.output[0]) if len(inferred.graph.output) == 1 else None
    row.update({
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "input_shape": graph_input_shape,
        "output_shape": graph_output_shape,
        "canonical_input": graph_input_shape == EXPECTED_IO,
        "canonical_output": graph_output_shape == EXPECTED_IO,
        "nonstatic_typed_values": nonstatic,
        "missing_typed_node_outputs": missing_node_outputs,
        "all_typed_node_outputs_static_positive": not nonstatic and not missing_node_outputs,
        "nonstandard_domains": domains,
        "standard_ops_only": not domains,
        "banned_ops": banned,
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "initializers": initializers,
        "finite_initializers": all(item["nonfinite"] == 0 for item in initializers.values()),
        "initializer_elements": sum(item["elements"] for item in initializers.values()),
        "max_einsum_inputs": max_einsum_inputs,
        "giant_einsum_min_inputs": GIANT_EINSUM_MIN_INPUTS,
        "giant_einsum": max_einsum_inputs >= GIANT_EINSUM_MIN_INPUTS,
        "giant_initializer_min_elements": GIANT_INITIALIZER_MIN_ELEMENTS,
        "giant_initializers": giant_initializers,
        "explicit_lookup_ops": explicit_lookup_ops,
        "gather_data_sources": gather_data_sources,
        "no_lookup_or_fixture_table": no_lookup,
        "conv_bias_findings": conv_bias_findings,
        "conv_bias_ub0": not conv_bias_findings,
        "static_memory": static_memory(inferred) if row.get("strict_data_prop") else None,
        "params": scoring.calculate_params(model),
    })
    row["static_cost"] = (
        row["static_memory"] + row["params"]
        if row["static_memory"] is not None and row["params"] is not None else None
    )
    row["structurally_clean_except_truth_and_canonical_output"] = bool(
        row.get("full_check") and row.get("strict_data_prop")
        and row["canonical_input"] and row["all_typed_node_outputs_static_positive"]
        and row["standard_ops_only"] and not banned and row["nested_graphs"] == 0
        and row["functions"] == 0 and row["sparse_initializers"] == 0
        and not external and row["finite_initializers"]
        and not row["giant_einsum"] and not giant_initializers
        and no_lookup and row["conv_bias_ub0"]
    )
    return row


def trace_all_outputs(
    model: onnx.ModelProto,
    benchmark: dict[str, np.ndarray],
    optimization: str,
    threads: int,
) -> dict[str, Any]:
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        typed = {
            value.name: value
            for value in [*inferred.graph.value_info, *inferred.graph.output]
            if value.type.HasField("tensor_type")
            and dims(value) and all(dim is not None and dim > 0 for dim in dims(value))
        }
        names = []
        for node in inferred.graph.node:
            for name in node.output:
                if name and name in typed and name not in names:
                    names.append(name)
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        options = ort.SessionOptions()
        if optimization == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif optimization != "default":
            raise ValueError(optimization)
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.log_severity_level = 4
        runtime = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        arrays = runtime.run(names, {runtime.get_inputs()[0].name: benchmark["input"]})
        mismatches = [
            {"name": name, "declared": dims(typed[name]), "actual": list(array.shape)}
            for name, array in zip(names, arrays)
            if dims(typed[name]) != list(array.shape)
        ]
        nonfinite = sum(
            int(np.count_nonzero(~np.isfinite(array)))
            for array in arrays if np.asarray(array).dtype.kind in "fc"
        )
        output_index = names.index("output")
        output = np.asarray(arrays[output_index])
        expected = benchmark["output"].astype(bool)
        return {
            "session_created": True,
            "traced_node_outputs": len(names),
            "traced_names": names,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "nonfinite_values": nonfinite,
            "output_actual_shape": list(output.shape),
            "first_case_correct": bool(np.array_equal(output > 0, expected)),
            "truthful": not mismatches and nonfinite == 0,
            "runtime_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "session_created": False,
            "traced_node_outputs": 0,
            "mismatch_count": None,
            "mismatches": [],
            "nonfinite_values": None,
            "truthful": False,
            "runtime_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    rows = [row for row in discovery["rows"] if int(row["task"]) == 71]
    if len(rows) != 1:
        raise RuntimeError(f"expected one task071 actual-lower row, got {len(rows)}")
    discovered = rows[0]
    if discovered["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("task071 actual-lower SHA changed")
    if int(discovered["actual_cost"]) != 186:
        raise RuntimeError("task071 actual-lower cost changed")

    data = SOURCE.read_bytes()
    if digest(data) != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("selected source does not match pinned actual-lower SHA")
    duplicate_rows = []
    for path in DUPLICATE_SOURCES:
        current = path.read_bytes()
        duplicate_rows.append({
            "path": relative(path),
            "sha256": digest(current),
            "byte_identical_to_selected": current == data,
        })
    if not all(row["byte_identical_to_selected"] for row in duplicate_rows):
        raise RuntimeError("historical duplicate sources disagree")

    if digest(BASELINE_ZIP.read_bytes()) != BASELINE_ZIP_SHA256:
        raise RuntimeError("historical baseline ZIP changed")
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        authority = archive.read("task071.onnx")

    QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
    QUARANTINE.write_bytes(data)
    if digest(QUARANTINE.read_bytes()) != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("quarantine copy changed bytes")

    model = onnx.load_model_from_string(data)
    static = static_audit(model)
    candidate_profile = profile_bytes(data, "candidate")
    authority_profile = profile_bytes(authority, "authority")
    if candidate_profile != {"memory": 100, "params": 86, "cost": 186}:
        raise RuntimeError(f"unexpected candidate actual profile: {candidate_profile}")
    if authority_profile["cost"] != 188:
        raise RuntimeError(f"unexpected authority actual profile: {authority_profile}")

    examples = scoring.load_examples(71)
    known_counts = {name: len(examples[name]) for name in ("train", "test", "arc-gen")}
    benchmark = scoring.convert_to_numpy(examples["train"][0])
    if benchmark is None:
        raise RuntimeError("first known train example is not convertible")
    traces = {
        label: trace_all_outputs(model, benchmark, optimization, threads)
        for label, optimization, threads in CONFIGS
    }
    all_sessions_created = all(row["session_created"] for row in traces.values())
    all_truthful = all(row["truthful"] for row in traces.values())
    any_runtime_error = any(row["runtime_error"] is not None for row in traces.values())
    any_shape_mismatch = any((row["mismatch_count"] or 0) > 0 for row in traces.values())
    any_nonfinite = any((row["nonfinite_values"] or 0) > 0 for row in traces.values())
    default_sessions_created = all(
        traces[label]["session_created"] for label in ("default_threads1", "default_threads4")
    )
    no_shape_cloak = bool(
        static["canonical_output"] and all_truthful and not any_shape_mismatch
    )
    pre_gates = {
        "pinned_target_sha": digest(data) == EXPECTED_CANDIDATE_SHA256,
        "actual_profile_cost186": candidate_profile["cost"] == 186,
        "full_check": bool(static.get("full_check")),
        "strict_data_prop": bool(static.get("strict_data_prop")),
        "canonical_static_io": bool(static["canonical_input"] and static["canonical_output"]),
        "all_typed_node_outputs_static_positive": static["all_typed_node_outputs_static_positive"],
        "standard_ops": static["standard_ops_only"] and not static["banned_ops"],
        "no_lookup": static["no_lookup_or_fixture_table"],
        "no_giant_einsum": not static["giant_einsum"],
        "no_giant_initializer": not static["giant_initializers"],
        "finite_initializers": static["finite_initializers"],
        "conv_bias_ub0": static["conv_bias_ub0"],
        "all_four_sessions_created": all_sessions_created,
        "default_sessions_created": default_sessions_created,
        "runtime_errors_zero": not any_runtime_error,
        "runtime_nonfinite_zero": not any_nonfinite,
        "all_intermediate_runtime_shapes_truthful": all_truthful,
        "runtime_shape_mismatches_zero": not any_shape_mismatch,
        "no_shape_cloak": no_shape_cloak,
    }
    pre_gate_pass = all(pre_gates.values())
    failed = [name for name, value in pre_gates.items() if not value]
    if pre_gate_pass:
        raise RuntimeError("unexpected healthy pre-gate; implement known/fresh continuation before accepting")

    payload = {
        "task": 71,
        "lane": "root_task071_policy90_276",
        "decision": "REJECT_SHAPE_CLOAK_PRE_GATE",
        "accepted": False,
        "target_disambiguation": {
            "discovery_path": relative(DISCOVERY),
            "actual_lower_row_sha256": discovered["sha256"],
            "actual_lower_row_cost": discovered["actual_cost"],
            "actual_lower_row_sources": discovered["sources"],
            "selected_source": relative(SOURCE),
            "selected_sha256": digest(data),
            "historical_duplicate_sources": duplicate_rows,
            "castlike_nodes": static["op_histogram"].get("CastLike", 0),
            "max_einsum_inputs": static["max_einsum_inputs"],
            "giant_einsum": static["giant_einsum"],
            "note": (
                "This pins the non-giant actual-lower row. It is not identified by the "
                "presence of CastLike alone and is not the separate giant/default-unstable artifact."
            ),
        },
        "authority": {
            "zip": relative(BASELINE_ZIP),
            "zip_sha256": BASELINE_ZIP_SHA256,
            "member": "task071.onnx",
            "member_sha256": digest(authority),
            "actual_profile": authority_profile,
        },
        "candidate": {
            "source": relative(SOURCE),
            "quarantine": relative(QUARANTINE),
            "sha256": digest(data),
            "file_bytes": len(data),
            "actual_profile": candidate_profile,
            "strict_lower_by": authority_profile["cost"] - candidate_profile["cost"],
        },
        "static": static,
        "known_corpus": {
            "split_counts": known_counts,
            "total": sum(known_counts.values()),
        },
        "truthful_runtime_shape_trace": traces,
        "pre_gates": pre_gates,
        "pre_gate_pass": pre_gate_pass,
        "failed_pre_gates": failed,
        "policy90_runtime": {
            "known_four_configs_executed": False,
            "fresh_seeds": [276_071_001, 276_171_001],
            "fresh_per_seed": 10_000,
            "fresh_executed": False,
            "reason": (
                "Immediate fail-closed rejection: three declared/runtime shape mismatches "
                "exist in every smoke configuration, including two intermediates and output."
            ),
        },
        "policy": {
            "normal_policy90_threshold": 0.90,
            "fail_closed": True,
            "root_or_71407_written": False,
            "quarantine_only": True,
            "candidate_promoted": False,
            "kimi_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "candidate_sha256": digest(data),
        "candidate_profile": candidate_profile,
        "authority_profile": authority_profile,
        "failed_pre_gates": failed,
        "trace_summary": {
            label: {
                "session_created": row["session_created"],
                "traced": row["traced_node_outputs"],
                "mismatch_count": row["mismatch_count"],
                "runtime_error": row["runtime_error"],
            }
            for label, row in traces.items()
        },
        "known4_executed": False,
        "fresh_executed": False,
        "evidence": relative(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
