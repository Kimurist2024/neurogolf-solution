#!/usr/bin/env python3
"""Structural, cost, runtime-shape, and known-example audit of lane baselines."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR_PATH = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
SPEC = importlib.util.spec_from_file_location("mid13_shared_auditor", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def static_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def explicit_shape_witnesses(model: onnx.ModelProto) -> list[dict[str, object]]:
    """Find guaranteed shape-preserving-op declaration conflicts."""
    values = {
        value.name: static_shape(value)
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }
    rows: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type not in {
            "Cast",
            "CastLike",
            "GroupNormalization",
            "Identity",
            "ScatterElements",
            "ScatterND",
        }:
            continue
        if not node.input or not node.output:
            continue
        source = values.get(node.input[0])
        declared = values.get(node.output[0])
        if source is None or declared is None or source == declared:
            continue
        rows.append(
            {
                "node": node.name or node.output[0],
                "op": node.op_type,
                "tensor": node.output[0],
                "source_shape": source,
                "declared_shape": declared,
                "reason": f"{node.op_type} preserves the first input's shape",
            }
        )
    return rows


def main() -> None:
    private_zero_catalog = {
        9, 15, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101, 102,
        112, 118, 133, 134, 145, 157, 158, 169, 170, 178, 185, 191,
        192, 196, 198, 202, 205, 208, 219, 222, 233, 246, 255, 264,
        277, 286, 302, 319, 325, 333, 343, 344, 361, 365, 366, 372,
        377, 391, 393, 396,
    }
    output: dict[str, object] = {}
    for task in (237, 238, 354, 378):
        path = HERE / "baseline" / f"task{task}.onnx"
        model = onnx.load(path, load_external_data=False)
        row = AUDITOR.audit(f"baseline_task{task}", task, path)
        row["explicit_shape_witnesses"] = explicit_shape_witnesses(model)
        trace = row.get("runtime_shape_trace") or {}
        trace_mismatches = trace.get("declared_actual_mismatches") or []
        row["truthful_static_runtime_shapes"] = not (
            row["explicit_shape_witnesses"]
            or trace_mismatches
            or trace.get("error")
        )
        row["private_status"] = {
            "catalogued_private_zero": task in private_zero_catalog,
            "source": "docs/golf/private_zero_tasks.md",
            "note": (
                "task354 appears in the documented isolated tail ordering and "
                "was previously white-probed; none of the four is listed as a "
                "private-zero task. This is provenance, not a new private probe."
            ),
        }
        output[str(task)] = row
        (HERE / "baseline_audit.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        score = row.get("official_like_score") or {}
        print(
            f"task={task} cost={score.get('cost')} "
            f"known_disable={row.get('known_disable_all', {}).get('total')} "
            f"known_default={row.get('known_default', {}).get('total')} "
            f"truthful_shapes={row['truthful_static_runtime_shapes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
