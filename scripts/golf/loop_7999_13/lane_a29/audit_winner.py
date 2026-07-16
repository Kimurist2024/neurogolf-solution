#!/usr/bin/env python3
"""Strict structural, algebraic, and dual-ORT known audit for A29 winner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

TASK = 275
BASE = HERE / "task275_base.onnx"
CANDIDATE = HERE / "task275_shared_gate_router.onnx"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def io_shape(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    nested = [
        node.output[0]
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    bad = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
    ]
    inferred_shapes = {
        item.name: io_shape(item)
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "checker_full": True,
        "strict_shape_inference": True,
        "input_shape": io_shape(model.graph.input[0]),
        "output_shape": io_shape(model.graph.output[0]),
        "inferred_shapes": inferred_shapes,
        "node_count": len(model.graph.node),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
        "einsum_operand_count": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0
        ),
        "params": sum(max(1, math.prod(item.dims)) for item in model.graph.initializer),
        "initializer_names": [item.name for item in model.graph.initializer],
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "foreign_domains": [item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")],
        "bad_ops": bad,
        "nested_graphs": nested,
        "conv_family": [node.op_type for node in model.graph.node if node.op_type in {"ConvTranspose", "QLinearConv"}],
    }


def router_proof() -> dict[str, object]:
    base = arrays(onnx.load(BASE))
    candidate = arrays(onnx.load(CANDIDATE))
    rows: list[dict[str, object]] = []
    exact = True
    for total in (18.0, 32.0):
        base_gate = base["GW"].reshape(2) * total + base["GB"]
        candidate_gate = candidate["GW"].reshape(2) * total + candidate["GB"]
        base_router = np.einsum("a,ap,aq->pq", base_gate, base["GU"], base["GV"])
        candidate_router = np.einsum(
            "a,ap,aq->pq", candidate_gate, candidate["GU"], candidate["GU"]
        )
        same = np.array_equal(base_router, candidate_router)
        exact &= same
        rows.append(
            {
                "total": total,
                "base_gate": base_gate.tolist(),
                "candidate_gate": candidate_gate.tolist(),
                "base_router": base_router.tolist(),
                "candidate_router": candidate_router.tolist(),
                "exact": same,
            }
        )
    return {
        "symbolic_identity": (
            "GV[0] = -GU[0], GV[1] = 7*GU[1]; therefore "
            "(total-25)*GU[0]xGV[0] + GU[1]xGV[1] = "
            "(25-total)*GU[0]xGU[0] + 7*GU[1]xGU[1]"
        ),
        "real_arithmetic_exact_for_all_totals": True,
        "generator_reachable_totals": [18, 32],
        "rows": rows,
        "exact": exact,
    }


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_audit(model: onnx.ModelProto, disabled: bool) -> dict[str, object]:
    sess = session(model, disabled)
    right = wrong = errors = nonfinite = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(TASK)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                nonfinite += int(raw.size - np.count_nonzero(np.isfinite(raw)))
                positive = raw[raw > 0.0]
                nonpositive = raw[raw <= 0.0]
                if positive.size:
                    minimum_positive = min(minimum_positive, float(np.min(positive)))
                if nonpositive.size:
                    maximum_nonpositive = max(maximum_nonpositive, float(np.max(nonpositive)))
                if np.array_equal(raw > 0.0, benchmark["output"] > 0.0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001 - explicitly counted audit evidence
                errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": right + wrong + errors,
        "nonfinite_values": nonfinite,
        "minimum_positive": minimum_positive,
        "maximum_nonpositive": maximum_nonpositive,
        "margin_stable": minimum_positive >= 0.25 and maximum_nonpositive <= 0.0,
        "perfect": right == 266 and wrong == 0 and errors == 0 and nonfinite == 0,
    }


def main() -> None:
    base_model = onnx.load(BASE)
    candidate_model = onnx.load(CANDIDATE)
    base_structure = structural(BASE)
    candidate_structure = structural(CANDIDATE)
    proof = router_proof()
    report = {
        "task": TASK,
        "base": base_structure,
        "candidate": candidate_structure,
        "removed_initializers": sorted(
            set(base_structure["initializer_names"]) - set(candidate_structure["initializer_names"])
        ),
        "added_initializers": sorted(
            set(candidate_structure["initializer_names"]) - set(base_structure["initializer_names"])
        ),
        "router_proof": proof,
        "known": {
            "candidate_disable_all": known_audit(candidate_model, True),
            "candidate_default": known_audit(candidate_model, False),
            "base_disable_all": known_audit(base_model, True),
            "base_default": known_audit(base_model, False),
        },
        "cost": {
            "base_memory": 12,
            "base_params": 420,
            "base_cost": 432,
            "candidate_memory": 12,
            "candidate_params": 416,
            "candidate_cost": 428,
            "score_gain": math.log(432 / 428),
        },
    }
    report["pass"] = bool(
        proof["exact"]
        and report["removed_initializers"] == ["GV"]
        and report["added_initializers"] == []
        and candidate_structure["input_shape"] == [1, 10, 30, 30]
        and candidate_structure["output_shape"] == [1, 10, 30, 30]
        and candidate_structure["einsum_operand_count"] == base_structure["einsum_operand_count"] == 41
        and not candidate_structure["functions"]
        and not candidate_structure["sparse_initializers"]
        and not candidate_structure["foreign_domains"]
        and not candidate_structure["bad_ops"]
        and not candidate_structure["nested_graphs"]
        and not candidate_structure["conv_family"]
        and all(item["perfect"] and item["margin_stable"] for item in report["known"].values())
    )
    (HERE / "winner_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
