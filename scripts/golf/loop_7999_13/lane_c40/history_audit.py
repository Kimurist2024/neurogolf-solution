#!/usr/bin/env python3
"""Inventory task391 baseline, sub-baseline history, and SOUND controls."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


BASELINE = HERE / "baseline" / "task391.onnx"
SUB_BASELINE = [
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task391_r01_static85.onnx",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task391_r02_static87.onnx",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task391_r03_static87.onnx",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task391_r04_static88.onnx",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task391_r05_static88.onnx",
    ROOT / "artifacts/quarantine/task391_7801rej_cost102_private0_soloprobe.onnx",
]
SOUND_CONTROLS = [
    ROOT / "others/highspeed/task391_cost139.onnx",
    ROOT / "scripts/golf/scratch_codex/task391/cand_argmax_xor.onnx",
    ROOT / "scripts/golf/scratch/task391/candidate_strided_topk.onnx",
]
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LIMIT = 1.44 * 1024 * 1024


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_shapes_static(model: onnx.ModelProto) -> bool:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    return all(
        value.type.tensor_type.HasField("shape")
        and all(
            dim.HasField("dim_value") and dim.dim_value > 0
            for dim in value.type.tensor_type.shape.dim
        )
        for value in values
    )


def audit(path: Path, category: str) -> dict[str, Any]:
    model = onnx.load(path)
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001 - rejection evidence
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        static = all_shapes_static(model)
    except Exception as exc:  # noqa: BLE001
        strict = static = False
        strict_error = f"{type(exc).__name__}: {exc}"
    ops = Counter(node.op_type for node in model.graph.node)
    lookup_nodes = [
        node.op_type
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax"}
    ]
    lookup_attribute_ints = sum(
        len(attribute.ints)
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax"}
        for attribute in node.attribute
    )
    nested = sum(
        attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    )
    external = sum(
        item.data_location == onnx.TensorProto.EXTERNAL or bool(item.external_data)
        for item in model.graph.initializer
    )
    finite = all(
        array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
        for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
    )
    banned = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
    ]
    giant = [
        {"node": node.name or node.output[0], "inputs": len(node.input)}
        for node in model.graph.node
        if node.op_type == "Einsum" and len(node.input) >= 8
    ]
    memory, params, total = cost_of(str(path))
    standard = all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(
        node.domain in ("", "ai.onnx") for node in model.graph.node
    )
    safe_candidate_structure = all(
        (
            checker,
            strict,
            static,
            standard,
            not model.functions,
            not model.graph.sparse_initializer,
            nested == 0,
            external == 0,
            finite,
            not banned,
            not giant,
            not check_conv_bias(model),
            not lookup_nodes,
        )
    )
    return {
        "category": category,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "file_bytes": path.stat().st_size,
        "file_limit_bytes": LIMIT,
        "file_margin_bytes": LIMIT - path.stat().st_size,
        "cost": {"memory": memory, "params": params, "total": total},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "ops": dict(ops),
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_inference": strict,
        "strict_error": strict_error,
        "all_shapes_static_positive": static,
        "standard_domains": standard,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graph_attributes": nested,
        "external_initializers": external,
        "finite_initializers": finite,
        "banned_ops": banned,
        "giant_einsum": giant,
        "conv_bias_findings": check_conv_bias(model),
        "lookup_nodes": lookup_nodes,
        "lookup_attribute_ints": lookup_attribute_ints,
        "safe_candidate_structure": safe_candidate_structure,
        "candidate_eligible_below_104": total < 104 and safe_candidate_structure,
    }


def main() -> int:
    rows = [audit(BASELINE, "authoritative_baseline")]
    rows.extend(audit(path, "sub_baseline_history") for path in SUB_BASELINE)
    rows.extend(audit(path, "sound_control") for path in SOUND_CONTROLS)
    baseline = rows[0]
    report = {
        "task": 391,
        "submission_base": {
            "zip": "submission_base_8000.46.zip",
            "zip_member": "task391.onnx",
            "member_sha256": baseline["sha256"],
            "member_cost": baseline["cost"],
            "member_file_bytes": baseline["file_bytes"],
        },
        "rows": rows,
        "sub_baseline_candidates": [
            row for row in rows if row["category"] == "sub_baseline_history"
        ],
        "sound_controls": [row for row in rows if row["category"] == "sound_control"],
        "eligible": [row for row in rows if row["candidate_eligible_below_104"]],
        "conclusion": (
            "Every discovered model below cost 104 contains TfIdfVectorizer lookup payloads. "
            "The smallest table-free generator-derived control costs 139, so no model is both "
            "strictly cheaper and admissible."
        ),
    }
    (HERE / "history_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": baseline["cost"],
                "below": [
                    {
                        "sha256": row["sha256"],
                        "cost": row["cost"]["total"],
                        "lookup": row["lookup_nodes"],
                    }
                    for row in report["sub_baseline_candidates"]
                ],
                "sound_controls": [
                    {"sha256": row["sha256"], "cost": row["cost"]["total"]}
                    for row in report["sound_controls"]
                ],
                "eligible": len(report["eligible"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
