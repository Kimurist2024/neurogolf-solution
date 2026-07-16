#!/usr/bin/env python3
"""Final structural/algebraic/raw-equivalence audit for the A31 winner."""

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

TASK = 306
BASE = HERE / "task306_base.onnx"
CANDIDATE = HERE / "task306_reuse_dp0_diag_for_s.onnx"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def shape(item: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    bad = [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED]
    nested = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "opsets": [[item.domain, item.version] for item in model.opset_import],
        "input_shape": shape(model.graph.input[0]),
        "output_shape": shape(model.graph.output[0]),
        "checker_full": True,
        "strict_shape_inference": True,
        "strict_value_info_count": len(inferred.graph.value_info),
        "nodes": len(model.graph.node),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
        "einsum_operand_count": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "initializer_names": [item.name for item in model.graph.initializer],
        "params": sum(math.prod(item.dims) for item in model.graph.initializer),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "foreign_domains": [
            item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")
        ],
        "bad_ops": bad,
    }


def algebra() -> dict[str, object]:
    base_model = onnx.load(BASE)
    candidate_model = onnx.load(CANDIDATE)
    base = arrays(base_model)
    candidate = arrays(candidate_model)
    scale = base["S"] / np.diag(base["Dp0"])
    inverse = 1.0 / scale
    d0_products_equal = np.array_equal(
        base["Dp0"][:, :, None] * base["X"][None, :, :],
        candidate["Dp0"][:, :, None] * candidate["X"][None, :, :],
    )
    d1_products_equal = np.array_equal(
        base["Dp1"][:, :, None] * base["X"][None, :, :],
        candidate["Dp1"][:, :, None] * candidate["X"][None, :, :],
    )
    diagonal_equal = np.array_equal(np.diag(candidate["Dp0"]), base["S"])

    base_node = base_model.graph.node[0]
    candidate_node = candidate_model.graph.node[0]
    base_eq = next(item.s.decode("ascii") for item in base_node.attribute if item.name == "equation")
    candidate_eq = next(
        item.s.decode("ascii") for item in candidate_node.attribute if item.name == "equation"
    )
    base_terms = base_eq.split("->")[0].split(",")
    candidate_terms = candidate_eq.split("->")[0].split(",")
    rewrites = []
    other_terms_equal = True
    for index, (base_name, candidate_name) in enumerate(zip(base_node.input, candidate_node.input)):
        if base_name == "S":
            rewrites.append(
                {
                    "operand": index,
                    "base": [base_name, base_terms[index]],
                    "candidate": [candidate_name, candidate_terms[index]],
                }
            )
            other_terms_equal &= bool(
                candidate_name == "Dp0" and candidate_terms[index] == base_terms[index] * 2
            )
        else:
            other_terms_equal &= bool(
                base_name == candidate_name and base_terms[index] == candidate_terms[index]
            )
    return {
        "scale": scale.tolist(),
        "inverse_scale": inverse.tolist(),
        "identity": (
            "Dp'[:,q]=Dp[:,q]*c[q] and X'[q,:]=X[q,:]/c[q], so every "
            "pointwise Dp-X factor is unchanged; diag(Dp0')=S permits S[q] -> Dp0'[q,q]."
        ),
        "float32_dyadic_scaling": True,
        "dp0_x_products_bit_equal": d0_products_equal,
        "dp1_x_products_bit_equal": d1_products_equal,
        "candidate_dp0_diagonal_equals_base_s": diagonal_equal,
        "s_operand_rewrites": rewrites,
        "s_rewrite_count": len(rewrites),
        "all_other_operands_and_terms_equal": other_terms_equal,
        "exact": bool(
            np.array_equal(scale, np.array([-1.0, -1.0, -2.0], dtype=np.float32))
            and d0_products_equal
            and d1_products_equal
            and diagonal_equal
            and len(rewrites) == 12
            and other_terms_equal
        ),
    }


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def raw_known(disabled: bool) -> dict[str, object]:
    base_session = session(onnx.load(BASE), disabled)
    candidate_session = session(onnx.load(CANDIDATE), disabled)
    total = raw_equal = threshold_equal = errors = nonfinite = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(TASK)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            total += 1
            try:
                base_raw = base_session.run(["output"], {"input": benchmark["input"]})[0]
                candidate_raw = candidate_session.run(["output"], {"input": benchmark["input"]})[0]
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            raw_equal += int(np.array_equal(base_raw, candidate_raw))
            threshold_equal += int(np.array_equal(base_raw > 0, candidate_raw > 0))
            nonfinite += int(candidate_raw.size - np.count_nonzero(np.isfinite(candidate_raw)))
            positive = candidate_raw[candidate_raw > 0]
            nonpositive = candidate_raw[candidate_raw <= 0]
            if positive.size:
                minimum_positive = min(minimum_positive, float(np.min(positive)))
            if nonpositive.size:
                maximum_nonpositive = max(maximum_nonpositive, float(np.max(nonpositive)))
    return {
        "total": total,
        "raw_equal": raw_equal,
        "threshold_equal": threshold_equal,
        "errors": errors,
        "nonfinite_values": nonfinite,
        "minimum_positive": minimum_positive,
        "maximum_nonpositive": maximum_nonpositive,
        "perfect_raw_equivalence": bool(
            total == 265
            and raw_equal == total
            and threshold_equal == total
            and errors == 0
            and nonfinite == 0
        ),
    }


def main() -> None:
    base = structure(BASE)
    candidate = structure(CANDIDATE)
    proof = algebra()
    known = {
        "disable_all": raw_known(True),
        "default": raw_known(False),
    }
    fresh = json.loads((HERE / "task306_fresh5000.json").read_text())[0]
    external = json.loads((HERE / "task306_external500.json").read_text())
    report = {
        "task": TASK,
        "base": base,
        "candidate": candidate,
        "removed_initializers": sorted(
            set(base["initializer_names"]) - set(candidate["initializer_names"])
        ),
        "added_initializers": sorted(
            set(candidate["initializer_names"]) - set(base["initializer_names"])
        ),
        "algebra": proof,
        "known_raw_equivalence": known,
        "fresh5000": fresh,
        "external500": external,
        "cost": {
            "base_memory": 0,
            "base_params": 131,
            "base_cost": 131,
            "candidate_memory": 0,
            "candidate_params": 128,
            "candidate_cost": 128,
            "projected_gain": math.log(131 / 128),
        },
        "conv_bias_ub_count_in_lane_zip": 0,
    }
    report["pass"] = bool(
        proof["exact"]
        and report["removed_initializers"] == ["S"]
        and report["added_initializers"] == []
        and base["input_shape"] == candidate["input_shape"] == [1, 10, 30, 30]
        and base["output_shape"] == candidate["output_shape"] == [1, 10, 30, 30]
        and base["einsum_operand_count"] == candidate["einsum_operand_count"] == 69
        and candidate["nodes"] == 1
        and not candidate["functions"]
        and not candidate["sparse_initializers"]
        and not candidate["nested_graphs"]
        and not candidate["foreign_domains"]
        and not candidate["bad_ops"]
        and all(item["perfect_raw_equivalence"] for item in known.values())
        and fresh["perfect"]
        and external["decision"]["verdict"] == "ACCEPT_STRICT"
        and external["differential"]["raw_equal"] == 500
        and external["differential"]["mismatches"] == 0
    )
    (HERE / "winner_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
