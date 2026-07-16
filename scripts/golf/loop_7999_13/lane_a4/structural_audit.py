#!/usr/bin/env python3
"""Strict structural and algebra audit for the retained task324 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline" / "task324.onnx"
CANDIDATE = HERE / "candidates" / "task324_synth_quarter.onnx"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_model(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    bad_ops = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
    ]
    nested = [
        node.output[0]
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    conv_biases = []
    initializers = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        bias_index = 8 if node.op_type == "QLinearConv" else 2
        if node.op_type not in {"Conv", "ConvTranspose", "QLinearConv"}:
            continue
        if len(node.input) > bias_index and node.input[bias_index]:
            bias = initializers.get(node.input[bias_index])
            conv_biases.append(
                {
                    "node": node.output[0],
                    "bias": node.input[bias_index],
                    "initializer": bias is not None,
                    "elements": int(np.prod(bias.dims)) if bias is not None else None,
                }
            )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "checker_full": True,
        "strict_shape_inference": True,
        "inferred_value_info": len(inferred.graph.value_info),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "bad_ops": bad_ops,
        "nested_graphs": nested,
        "foreign_domains": [op.domain for op in model.opset_import if op.domain not in ("", "ai.onnx")],
        "conv_biases": conv_biases,
        "node_count": len(model.graph.node),
        "params": sum(int(np.prod(init.dims)) if init.dims else 1 for init in model.graph.initializer),
    }


def algebra_proof() -> dict[str, object]:
    model = onnx.load(BASE)
    arrays = {init.name: numpy_helper.to_array(init).astype(np.float64) for init in model.graph.initializer}
    base0 = arrays["base0"]
    hot = arrays["onehot_values"]
    ref = arrays["refdiff"]
    seed = arrays["seedsel"]
    emap = arrays["Emap"]
    quarter = float(np.einsum("iKjL,K,L,iMjN,M,N->ij", base0, hot, hot, base0, hot, hot)[0, 0])
    degree_two = np.einsum("ZA,A,ZB,eB->e", ref, hot, seed, emap)
    return {
        "selected_base0_product": quarter,
        "expected_quarter": 0.25,
        "degree_two_selector": degree_two.tolist(),
        "expected_degree_two_selector": [0.0, 0.0, 1.0],
        "exact": quarter == 0.25 and np.array_equal(degree_two, np.array([0.0, 0.0, 1.0])),
    }


def main() -> None:
    base = onnx.load(BASE)
    candidate = onnx.load(CANDIDATE)
    base_names = {init.name for init in base.graph.initializer}
    candidate_names = {init.name for init in candidate.graph.initializer}
    report = {
        "baseline": audit_model(BASE),
        "candidate": audit_model(CANDIDATE),
        "initializer_removed": sorted(base_names - candidate_names),
        "initializer_added": sorted(candidate_names - base_names),
        "algebra_proof": algebra_proof(),
    }
    report["pass"] = bool(
        report["candidate"]["checker_full"]
        and report["candidate"]["strict_shape_inference"]
        and not report["candidate"]["functions"]
        and not report["candidate"]["sparse_initializers"]
        and not report["candidate"]["bad_ops"]
        and not report["candidate"]["nested_graphs"]
        and not report["candidate"]["foreign_domains"]
        and report["algebra_proof"]["exact"]
        and report["initializer_removed"] == ["deg2"]
        and report["initializer_added"] == []
    )
    (HERE / "structural_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
