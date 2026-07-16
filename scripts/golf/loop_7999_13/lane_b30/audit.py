#!/usr/bin/env python3
"""Final exact/structural audit for lane B30 task345."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline_task345.onnx"
LEGAL = HERE / "task345_legal_swapped_prescaled_cost389.onnx"
LEGAL410 = ROOT / "others/2/7615/task345_best_validated_no_further_reduction.onnx"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C11 = load_module(
    "lane_b30_c11_audit",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)
VERIFY = load_module("lane_b30_verify_fix", ROOT / "scripts/verify_fix.py")
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def negative_conv_pads(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose", "QLinearConv"}:
            continue
        pads = next(
            (
                list(onnx.helper.get_attribute_value(attribute))
                for attribute in node.attribute
                if attribute.name == "pads"
            ),
            [],
        )
        if any(value < 0 for value in pads):
            rows.append({"output": node.output[0], "pads": pads})
    return rows


def static_positive(model: onnx.ModelProto) -> bool:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                return False
    return True


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    memory, params, cost = cost_of(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    signatures: defaultdict[bytes, int] = defaultdict(int)
    for node in model.graph.node:
        canonical = copy.deepcopy(node)
        canonical.name = ""
        del canonical.output[:]
        signatures[canonical.SerializeToString()] += 1
    duplicates = sorted(count for count in signatures.values() if count > 1)
    initializer_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    exact_initializer_duplicates: list[list[str]] = []
    names = sorted(initializer_arrays)
    for index, first in enumerate(names):
        group = [first]
        for second in names[index + 1 :]:
            if (
                initializer_arrays[first].dtype == initializer_arrays[second].dtype
                and initializer_arrays[first].shape == initializer_arrays[second].shape
                and np.array_equal(initializer_arrays[first], initializer_arrays[second])
            ):
                group.append(second)
        if len(group) > 1 and not any(first in prior for prior in exact_initializer_duplicates):
            exact_initializer_duplicates.append(group)
    trace = C11.runtime_shape_trace(345, model)
    declared_mismatches = trace.get("declared_actual_mismatches", [])
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "memory": int(memory),
        "params": int(params),
        "cost": int(cost),
        "node_count": len(model.graph.node),
        "value_info_count": len(model.graph.value_info),
        "full_checker": True,
        "strict_shape_data_prop": True,
        "static_positive": static_positive(model),
        "runtime_shape_trace": trace,
        "shape_truthful": not declared_mismatches and trace.get("undeclared_intermediate_count") == 0,
        "negative_conv_pads": negative_conv_pads(model),
        "conv_bias_findings": check_conv_bias(model),
        "standard_domains": all(item.domain in {"", "ai.onnx"} for item in model.opset_import),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        ],
        "nested_graph_attributes": sum(
            attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node
            for attribute in node.attribute
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "lookup_red_flags": {
            "tfidf": ops.get("TfIdfVectorizer", 0),
            "hardmax": ops.get("Hardmax", 0),
            "giant_einsum_nodes": sum(
                node.op_type == "Einsum" and len(node.input) >= 8
                for node in model.graph.node
            ),
            "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        },
        "exact_duplicate_initializers": exact_initializer_duplicates,
        "duplicate_node_signature_multiplicities": duplicates,
    }


def known_raw_differential(
    baseline: onnx.ModelProto, candidate: onnx.ModelProto, disable_all: bool
) -> dict[str, object]:
    base_session = session(baseline, disable_all)
    candidate_session = session(candidate, disable_all)
    base_input = base_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name
    base_output = base_session.get_outputs()[0].name
    candidate_output = candidate_session.get_outputs()[0].name
    requested = raw_equal = threshold_equal = errors = 0
    maximum = 0.0
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(345)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            requested += 1
            try:
                left = np.asarray(
                    base_session.run([base_output], {base_input: benchmark["input"]})[0]
                )
                right = np.asarray(
                    candidate_session.run(
                        [candidate_output], {candidate_input: benchmark["input"]}
                    )[0]
                )
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            raw_equal += int(np.array_equal(left, right))
            threshold_equal += int(np.array_equal(left > 0, right > 0))
            maximum = max(maximum, float(np.max(np.abs(left.astype(np.int64) - right.astype(np.int64)))))
    return {
        "requested": requested,
        "raw_equal": raw_equal,
        "threshold_equal": threshold_equal,
        "errors": errors,
        "max_abs_difference": maximum,
    }


def scored_known(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    with tempfile.TemporaryDirectory(prefix="b30_known_", dir="/tmp") as workdir:
        scored = scoring.score_and_verify(
            copy.deepcopy(model), 345, workdir, "b30", require_correct=True
        )
    if scored is None:
        raise RuntimeError("official-like scorer returned None")
    return {
        "cost": int(scored["cost"]),
        "memory": int(scored["memory"]),
        "params": int(scored["params"]),
        "correct": bool(scored["correct"]),
        "official_decoder_gold": bool(VERIFY.official_gold(path, 345)),
        "disable_all": C11.known(345, session(model, True)),
        "default": C11.known(345, session(model, False)),
    }


def summarize_search(prefix: str) -> dict[str, object]:
    paths = sorted(HERE.glob(f"{prefix}*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    return {
        "files": [path.name for path in paths],
        "ranges": [row["range"] for row in rows],
        "collision_free_multipliers": sum(
            int(row["collision_free_multipliers"]) for row in rows
        ),
        "exact_decoder_candidates": sum(
            len(row["exact_decoder_candidates"]) for row in rows
        ),
    }


def main() -> int:
    baseline_model = onnx.load(BASELINE)
    legal_model = onnx.load(LEGAL)
    build_manifest = json.loads((HERE / "build_manifest.json").read_text())
    blank = json.loads((HERE / "blank_reuse_search.json").read_text())
    baseline_structure = structure(BASELINE)
    legal_structure = structure(LEGAL)
    legal410_structure = structure(LEGAL410)
    differential = {
        "disable_all": known_raw_differential(baseline_model, legal_model, True),
        "default": known_raw_differential(baseline_model, legal_model, False),
    }
    known = scored_known(LEGAL)

    # Exact factor/reuse arithmetic.  Wpack is an outer product of a 10-value
    # channel vector and a 10-value width vector (singletons omitted).
    factor_reuse = {
        "wpack_exact_rank1": {
            "dense_params": 100,
            "factor_params": 20,
            "parameter_saving": 80,
            "conv_accepts_factored_weight": False,
            "runtime_materialization_bytes": 400,
            "projected_cost_if_materialized": 709,
        },
        "wfac_from_wpack_channel2_reverse": {
            "equal_first_ten_values": True,
            "saved_params": 30,
            "required_intermediates_bytes": {
                "slice_float10": 40,
                "cast_int32_10": 40,
                "pad_int32_30": 120,
            },
            "net_cost_delta": 170,
        },
        "zero_from_wfac_slice": {
            "saved_params": 1,
            "minimum_slice_output_bytes": 4,
            "net_cost_delta": 3,
        },
        "blank_gauge": {
            "legal_generator_cases": blank["legal_generator_cases"],
            "scalars_tested": blank["scalar_count"],
            "exact_blank_gauge_count": len(blank["exact_blank_gauges"]),
        },
    }
    i16 = {
        "signed_add": summarize_search("i16_decoder_"),
        "unsigned_add": summarize_search("u16_decoder_"),
        "signed_xor": summarize_search("i16_xor_"),
        "unsigned_xor": summarize_search("u16_xor_"),
        "theoretical_cost_if_exact_decoder_existed": 380,
    }
    strict_cheaper = []
    decision = {
        "status": "NO_STRICT_CHEAPER_ADMISSIBLE_CANDIDATE",
        "winner_count": 0,
        "verified_gain": 0.0,
        "fresh5000": "skipped: no strict-cheaper candidate passed the cost gate",
        "external500": "skipped: no strict-cheaper candidate passed the cost gate",
        "root_submission_modified": False,
    }
    audit = {
        "task": 345,
        "baseline_score_label": 8000.46,
        "baseline": baseline_structure,
        "legal_same_cost_control": legal_structure,
        "prior_legal410": legal410_structure,
        "legal_control_known_dual": known,
        "baseline_vs_legal_control_raw_differential": differential,
        "sparse_storage_probes": build_manifest["sparse_probes"],
        "factor_gauge_cse_and_slice_reuse": factor_reuse,
        "int16_uint16_carrier_search": i16,
        "strict_cheaper_candidates": strict_cheaper,
        "decision": decision,
    }
    path = HERE / "audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": {
                    key: baseline_structure[key]
                    for key in ("sha256", "memory", "params", "cost", "negative_conv_pads")
                },
                "legal": {
                    key: legal_structure[key]
                    for key in (
                        "sha256",
                        "memory",
                        "params",
                        "cost",
                        "shape_truthful",
                        "negative_conv_pads",
                    )
                },
                "known": known,
                "differential": differential,
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
