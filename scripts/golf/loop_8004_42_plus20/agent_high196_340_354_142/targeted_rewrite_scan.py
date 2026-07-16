#!/usr/bin/env python3
"""Audit requested reduction/attribute rewrites without admitting shape cloaks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = HERE / "current"
REJECTED = HERE / "rejected_probes"
HELPER_DIR = HERE.parent / "agent_8009_exact_B_116"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module("high142_scan_helper", HELPER_DIR / "scan_candidates.py")
AUDIT = load_module("high142_audit_helper", HELPER_DIR / "audit_candidates.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def castlike_to_cast(model: onnx.ModelProto, output: str, dtype: int) -> None:
    matches = [node for node in model.graph.node if output in node.output]
    if len(matches) != 1 or matches[0].op_type != "CastLike":
        raise RuntimeError(f"expected one CastLike producer for {output}")
    node = matches[0]
    node.op_type = "Cast"
    del node.input[1:]
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", dtype)])


def profile(task: int, label: str, model: onnx.ModelProto) -> dict:
    data = model.SerializeToString()
    path = REJECTED / f"task{task}_{label}.onnx"
    path.write_bytes(data)
    official = SCAN.official_cost(data, f"task{task}_{label}")
    structural = SCAN.structural(model)
    trace = AUDIT.direct_trace(task, data) if structural.get("pass") else None
    baseline_cost = {196: 1210, 340: 1173, 354: 536}[task]
    strict_lower = official["cost"] < baseline_cost
    truthful = bool(trace and trace.get("truthful"))
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "official_profile": official,
        "strict_lower": strict_lower,
        "structural": structural,
        "runtime_shape_trace": trace,
        "truthful": truthful,
        "eligible_for_known_gate": bool(strict_lower and structural.get("pass") and truthful),
        "decision": (
            "ADVANCE"
            if strict_lower and structural.get("pass") and truthful
            else "REJECT_NOT_STRICT_LOWER"
            if not strict_lower
            else "REJECT_STRUCTURE_OR_RUNTIME_SHAPE"
        ),
    }


def main() -> int:
    REJECTED.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, dict] = {}
    reduction_ops = {"ReduceL1", "ReduceSumSquare"}
    for task in (196, 340, 354):
        data = (CURRENT / f"task{task}.onnx").read_bytes()
        model = onnx.load_model_from_string(data)
        found = [
            {"index": index, "name": node.name, "op": node.op_type}
            for index, node in enumerate(model.graph.node)
            if node.op_type in reduction_ops
        ]
        tasks[str(task)] = {
            "baseline_sha256": digest(data),
            "reduction_rewrite": {
                "searched_ops": sorted(reduction_ops),
                "matches": found,
                "applicable": bool(found),
                "conclusion": (
                    "candidate required" if found else "no ReduceL1/ReduceSumSquare node exists"
                ),
            },
            "attribute_migration_probes": [],
        }

    base354 = onnx.load(CURRENT / "task354.onnx")
    for label, outputs in (
        ("shape12_cast", (("shape12_dyn", TensorProto.INT64),)),
        ("idx_i32_cast", (("idx_i32", TensorProto.INT32),)),
        (
            "both_casts",
            (("shape12_dyn", TensorProto.INT64), ("idx_i32", TensorProto.INT32)),
        ),
    ):
        candidate = copy.deepcopy(base354)
        for output, dtype in outputs:
            castlike_to_cast(candidate, output, dtype)
        tasks["354"]["attribute_migration_probes"].append(profile(354, label, candidate))

    prior = ROOT / "scripts/golf/loop_7999_13/lane_a42/A42_RESULT.json"
    prior_row = json.loads(prior.read_text(encoding="utf-8"))
    tasks["196"]["attribute_migration_probes"] = prior_row["exact_local_probes"]
    tasks["196"]["attribute_migration_evidence"] = str(prior.relative_to(ROOT))
    tasks["340"]["attribute_migration_conclusion"] = (
        "no type-template CastLike initializer or parameter-to-attribute migration exists"
    )

    result = {
        "scope": [196, 340, 354],
        "policy": (
            "Only official strict-lower, full/strict, UB0, runtime-shape-truthful payloads "
            "may advance to known/fresh gates. Existing LB-white shape-cloak status is SHA-only."
        ),
        "tasks": tasks,
        "stage_survivors": [
            row
            for task in tasks.values()
            for row in task["attribute_migration_probes"]
            if isinstance(row, dict) and row.get("eligible_for_known_gate")
        ],
    }
    (HERE / "targeted_rewrite_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage_survivors": len(result["stage_survivors"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
