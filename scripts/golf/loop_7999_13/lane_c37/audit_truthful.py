#!/usr/bin/env python3
"""Strict structural, runtime-shape, and known audit for C37 task162."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def shared() -> Any:
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c37_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def runtime_shapes(path: Path, mode: str) -> dict[str, object]:
    model = onnx.load(path)
    declared = {value.name: shape(value) for value in model.graph.value_info}
    typed = {value.name: value for value in model.graph.value_info}
    traced = copy.deepcopy(model)
    graph_outputs = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name in typed and name not in graph_outputs and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(
            traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        example = scoring.convert_to_numpy(scoring.load_examples(162)["train"][0])
        assert example is not None
        arrays = session.run(names, {"input": example["input"]})
    except Exception as exc:  # rejection evidence
        return {
            "mode": mode,
            "session_and_run_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "requested": len(names),
            "mismatch_count": None,
            "mismatches": [],
        }
    mismatches = [
        {"name": name, "declared": declared[name], "runtime": list(np.asarray(array).shape)}
        for name, array in zip(names, arrays, strict=True)
        if declared[name] != list(np.asarray(array).shape)
    ]
    return {
        "mode": mode,
        "session_and_run_ok": True,
        "error": "",
        "requested": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    memory, params, total = cost_of(path)
    return {
        "sha256": digest(path),
        "cost": {"memory": memory, "params": params, "total": total},
        "checker_full": True,
        "strict_data_prop": True,
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        "standard_domains": all(
            item.domain in ("", "ai.onnx") for item in model.opset_import
        ) and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "static_positive_shapes": all(
            dim.HasField("dim_value") and dim.dim_value > 0
            for value in values
            for dim in value.type.tensor_type.shape.dim
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all()) for array in arrays
        ),
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in model.graph.initializer
        ),
        "no_functions_subgraphs_sparse": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attr in node.attribute
            )
        ),
        "lookup_ops": sorted(
            {node.op_type for node in model.graph.node if node.op_type in {"TfIdfVectorizer", "GatherND", "ScatterND"}}
        ),
        "conv_ub_findings": check_conv_bias(model),
        "ops": sorted({node.op_type for node in model.graph.node}),
    }


def main() -> int:
    helper = shared()
    source = HERE / "sources" / "task162_exact_cse.onnx"
    repaired = HERE / "candidates" / "task162_exact_cse_truthful.onnx"
    baseline = HERE / "baseline" / "task162.onnx"
    report = {
        "authority": {
            "task": 162,
            "baseline_sha256": digest(baseline),
            "baseline_cost": structural(baseline)["cost"],
            "generator_hash": "6cf79266",
        },
        "source_cost373": {
            "path": str(source.relative_to(ROOT)),
            "structure": structural(source),
            "runtime_shapes": [runtime_shapes(source, mode) for mode in ("disabled", "default")],
        },
        "truthful_repair": {
            "path": str(repaired.relative_to(ROOT)),
            "structure": structural(repaired),
            "runtime_shapes": [runtime_shapes(repaired, mode) for mode in ("disabled", "default")],
            "known_dual": helper.known_dual(copy.deepcopy(onnx.load(repaired)), 162),
        },
        "cost_gate": {
            "baseline": 451,
            "candidate": structural(repaired)["cost"]["total"],
            "strictly_cheaper": structural(repaired)["cost"]["total"] < 451,
            "first_truthful_tensor_lower_bound": {
                "tensor": "csp30f",
                "shape": [1, 30, 30, 30],
                "dtype": "float32",
                "elements": 27000,
                "bytes": 108000,
                "reason": "one required intermediate alone exceeds the 451 budget",
            },
        },
        "fresh5000": {
            "run": False,
            "reason": "truthful repair fails the cheaper-than-451 gate",
        },
        "external500": {
            "run": False,
            "reason": "truthful repair fails the cheaper-than-451 gate",
        },
        "winners": [],
        "projected_gain": 0.0,
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (HERE / "winner_manifest.json").write_text(
        json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_runtime": report["source_cost373"]["runtime_shapes"],
                "repair_cost": report["truthful_repair"]["structure"]["cost"],
                "repair_runtime": report["truthful_repair"]["runtime_shapes"],
                "known_dual": report["truthful_repair"]["known_dual"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
