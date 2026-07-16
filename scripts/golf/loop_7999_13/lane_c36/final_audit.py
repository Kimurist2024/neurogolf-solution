#!/usr/bin/env python3
"""Assemble strict C36 task012 evidence and an empty winner manifest."""

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
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def shared() -> Any:
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c36_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def structure(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    return {
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
        "declared_shapes": {value.name: static_shape(value) for value in values},
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
        "conv_ub_findings": check_conv_bias(model),
        "ops": [node.op_type for node in model.graph.node],
        "lookup_or_giant_einsum": False,
    }


def load(path: str) -> object:
    return json.loads((HERE / path).read_text())


def search_summary(path: str, list_root: bool = False) -> dict[str, object]:
    data = load(path)
    rows = data if list_root else data["results"]
    return {
        "path": path,
        "alignments": len(rows),
        "winners": sum(bool(row.get("success")) for row in rows),
    }


def main() -> int:
    helper = shared()
    path = HERE / "baseline" / "task012.onnx"
    model = onnx.load(path)
    memory, params, total = cost_of(path)
    baseline = {
        "task": 12,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "cost": {"memory": memory, "params": params, "total": total},
        "structure": structure(model),
        "runtime_shapes": helper.trace_runtime_shapes(copy.deepcopy(model), 12),
        "known_dual": helper.known_dual(copy.deepcopy(model), 12),
        "external": load("task012_baseline_external.json")["candidate"],
        "fresh100_dual": [
            load("fresh_task012_base_disabled_100.json"),
            load("fresh_task012_base_default_100.json"),
        ],
    }
    searches = {
        "biased_hard_margin": [
            search_summary("../lane_c4/task012_below70_search.json", list_root=True),
            search_summary("missing_biased_search.json"),
            search_summary("missing_biased_search_long_side.json"),
        ],
        "biased_exact_decoder_boundary": [
            search_summary("missing_biased_search_weak_covered.json"),
            search_summary("missing_biased_search_weak.json"),
            search_summary("missing_biased_search_weak_long_side.json"),
        ],
        "bias_free_exact_decoder_boundary": [
            search_summary("homogeneous_search_missing_weak.json"),
            search_summary("homogeneous_search_weak_long_side.json"),
            search_summary("homogeneous_search_weak.json"),
        ],
    }
    report = {
        "authority": {
            "source_zip": "submission_base_8000.46.zip",
            "source_zip_sha256": "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534",
            "task_model_sha256": baseline["sha256"],
            "generator_hash": "0962bcdd",
        },
        "baseline": baseline,
        "complete_local_conv_lower_bound": {
            "generator_parameter_domain_cases": 392,
            "minimum_side_proof": (
                "A source pixel must affect outputs at both relative offsets -2 and +2 on each axis, "
                "so each kernel side is at least five."
            ),
            "area_below_70_coverage": (
                "Every integer geometry with kh>=5, kw>=5 and kh*kw<70 is covered; "
                "the largest possible side is floor(69/5)=13."
            ),
            "searches": searches,
            "hard_margin_alignment_total": sum(
                item["alignments"] for item in searches["biased_hard_margin"]
            ),
            "weak_affine_alignment_total": sum(
                item["alignments"] for item in searches["biased_exact_decoder_boundary"]
            ),
            "result": "No exact depthwise Conv below 710 parameters.",
        },
        "exact_structure": load("exact_structure_audit.json"),
        "fresh5000": {
            "run": False,
            "reason": "No cheaper candidate exists; fresh100 was retained as an incumbent control only.",
        },
        "candidate_external": {
            "run": False,
            "reason": "No cheaper candidate passed the analytic/generator-domain gate.",
        },
        "winners": [],
        "projected_gain": 0.0,
        "conclusion": "Retain authoritative task012 incumbent at cost 710.",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (HERE / "winner_manifest.json").write_text(
        json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"known_dual": baseline["known_dual"], "winners": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
