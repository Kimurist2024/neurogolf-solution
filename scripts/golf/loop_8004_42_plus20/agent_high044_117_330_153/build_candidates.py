#!/usr/bin/env python3
"""Build exact diagnostic rewrites that test the remaining local levers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_task330_constant_of_shape() -> tuple[onnx.ModelProto, dict[str, object]]:
    model = onnx.load(HERE / "baseline/task330.onnx")
    replacements: dict[str, np.ndarray] = {}
    kept = []
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape":
            kept.append(node)
            continue
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        value = attrs.get("value")
        if value is None or len(node.output) != 1:
            raise RuntimeError("unexpected ConstantOfShape form")
        fill = numpy_helper.to_array(value).reshape(-1)[0]
        replacements[node.output[0]] = np.asarray([fill], dtype=numpy_helper.to_array(value).dtype)
    if len(replacements) != 6:
        raise RuntimeError(f"expected six foldable nodes, got {len(replacements)}")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    for name, array in replacements.items():
        model.graph.initializer.append(numpy_helper.from_array(array, name))
    return model, {
        "rewrite": "ConstantOfShape([1], scalar-value-attribute) -> one-element initializer",
        "folded_outputs": sorted(replacements),
        "formal_equivalence": "Each removed node has the fixed input shape [1] and emits exactly one copy of its scalar value attribute.",
        "expected_nominal_delta": "remove six int64 scalar intermediates (48 bytes), add six parameter elements: -42 cost",
        "warning": "Diagnostic only: admission still requires truthful runtime shapes and both ORT modes.",
    }


def main() -> None:
    auditor = load_module(
        "lane153_auditor_candidate",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    candidates = HERE / "candidates"
    candidates.mkdir(exist_ok=True)
    model, proof = fold_task330_constant_of_shape()
    path = candidates / "task330_fold_constantofshape.onnx"
    onnx.save(model, path)
    row = {
        "task": 330,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "base_cost": 896,
        "proof": proof,
        "audit": auditor.audit("task330_fold_constantofshape", 330, path),
    }
    profile = row["audit"].get("official_like_score") or {}
    row["candidate_cost"] = profile.get("cost")
    row["strictly_lower"] = profile.get("cost") is not None and int(profile["cost"]) < 896
    trace = row["audit"].get("runtime_shape_trace") or {}
    row["truthful_runtime_shapes"] = not trace.get("error") and not trace.get("declared_actual_mismatches")
    (HERE / "candidate_audit.json").write_text(json.dumps({"task330_fold_constantofshape": row}, indent=2) + "\n")
    print(row["sha256"], row["candidate_cost"], row["truthful_runtime_shapes"])


if __name__ == "__main__":
    main()
