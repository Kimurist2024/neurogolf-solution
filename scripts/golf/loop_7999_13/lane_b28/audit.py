#!/usr/bin/env python3
"""B28 task347 exact-reuse and structural audit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.loop_7999_13.lane_b15 import audit_candidates as strict  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


ITEMS = {
    "baseline": HERE / "baseline_task347.onnx",
    "baseline_truthful_control": HERE / "baseline_task347_truthful_shapes.onnx",
    "history_cost51_shape_cloak": HERE / "history_cost51_shape_cloak.onnx",
    "history_cost143_shape_honest": HERE / "history_cost143_shape_honest.onnx",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initializer_inventory(model: onnx.ModelProto) -> dict[str, Any]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    rows = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        groups[(initializer.data_type, tuple(initializer.dims), array.tobytes())].append(initializer.name)
        rows.append(
            {
                "name": initializer.name,
                "dtype": int(initializer.data_type),
                "shape": list(initializer.dims),
                "params": int(math.prod(initializer.dims)),
                "uses": int(uses[initializer.name]),
                "values": array.reshape(-1).tolist(),
            }
        )
    duplicates = [names for names in groups.values() if len(names) > 1]
    return {"rows": rows, "exact_duplicate_groups": duplicates}


def node_cse_inventory(model: onnx.ModelProto) -> dict[str, Any]:
    signatures: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        attrs = tuple(sorted((attr.name, attr.SerializeToString()) for attr in node.attribute))
        signature = (node.domain, node.op_type, tuple(node.input), attrs)
        signatures[signature].append(node.name or f"node_{index}")
    duplicates = [names for names in signatures.values() if len(names) > 1]
    return {"exact_duplicate_node_groups": duplicates}


def audit_item(label: str, path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    structure = strict.structural(copy.deepcopy(model))
    runtime = strict.trace_runtime_shapes(copy.deepcopy(model), 347)
    memory, params, cost = (int(value) for value in cost_of(str(path)))
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "serialized_bytes": path.stat().st_size,
        "memory": memory,
        "params": params,
        "cost": cost,
        "structure": structure,
        "runtime_shape": runtime,
        "shape_truthful": runtime["shape_cloak"] is False,
        "conv_bias_findings": check_conv_bias(model),
        "known_dual": strict.known_dual(copy.deepcopy(model), 347),
    }


def main() -> int:
    rows = {label: audit_item(label, path) for label, path in ITEMS.items()}
    baseline_model = onnx.load(ITEMS["baseline"])
    initializer = initializer_inventory(baseline_model)
    cse = node_cse_inventory(baseline_model)
    algebraic_search = {
        "task": 347,
        "baseline_cost": 41,
        "allowed_families": {
            "duplicate_initializer": {
                "candidate_count": 0,
                "evidence": initializer["exact_duplicate_groups"],
                "reason": "no equal dtype/shape/value initializer pair exists",
            },
            "factor_or_gauge_reuse": {
                "candidate_count": 0,
                "reason": "s is already shared by GroupNormalization scale/bias and all four quantization scales; z is already shared by all four zero-point positions; no duplicated gauge factor remains",
            },
            "singleton_contraction": {
                "candidate_count": 0,
                "reason": "there is no Einsum, Gemm, or MatMul; final ConvInteger has the minimum ten output-channel coefficients required for channels 0 and 6, and the dynamic QLinearConv reuses x as both data and weight",
            },
            "common_subexpression": {
                "candidate_count": 0,
                "evidence": cse["exact_duplicate_node_groups"],
                "reason": "no duplicate node signature exists; g, x, s, z, W, and ss are already shared at every repeated use",
            },
        },
        "initializer_inventory": initializer,
        "node_cse_inventory": cse,
        "non_candidate_observation": {
            "slice": "qf reverses both spatial axes of the 3x3 q tensor; positive Conv strides cannot absorb that reversal exactly without another tensor/operator",
            "shape_floor": "truthfully declaring g and x exposes 45018 intermediate bytes before parameters",
        },
        "eligible_candidate_count": 0,
    }
    (HERE / "algebraic_search.json").write_text(json.dumps(algebraic_search, indent=2) + "\n")

    payload = {
        "task": 347,
        "baseline_score_label": 8000.46,
        "rows": rows,
        "decision": {
            "status": "NO_ELIGIBLE_WINNER",
            "winner_count": 0,
            "verified_gain": 0.0,
            "reasons": [
                "baseline claimed cost 41 relies on two shape-cloaked full-canvas intermediates",
                "truthful declaration of the same graph costs 45036",
                "the only shape-honest historical model costs 143, above baseline 41",
                "all permitted exact reuse families are already exhausted in the baseline graph",
                "fresh5000 and external validator500 were not run because no cheaper shape-honest candidate exists",
            ],
        },
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({label: {key: row[key] for key in ("sha256", "cost", "shape_truthful", "conv_bias_findings", "known_dual")} for label, row in rows.items()}, indent=2))
    print(json.dumps(payload["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
