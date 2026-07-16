#!/usr/bin/env python3
"""Try every available single ONNX optimizer pass on sound task225."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASK = 225
BASE_COST = 333
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def digest(model: onnx.ModelProto) -> str:
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def structural(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"checker_or_shape:{type(exc).__name__}:{exc}"
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return False, "noncanonical_io"
    if model.functions or model.graph.sparse_initializer:
        return False, "functions_or_sparse"
    if any(item.domain not in {"", "ai.onnx"} for item in model.opset_import):
        return False, "nonstandard_domain"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED or "Sequence" in node.op_type:
            return False, f"banned:{node.op_type}"
        if any(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for attr in node.attribute
        ):
            return False, "nested_graph"
    return True, "pass"


def main() -> None:
    model = onnx.load(HERE / "base" / f"task{TASK:03d}.onnx")
    seen = {digest(model)}
    rows: list[dict[str, object]] = []
    outdir = HERE / "optimizer_variants" / f"task{TASK:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    for pass_name in onnxoptimizer.get_available_passes():
        row: dict[str, object] = {"pass": pass_name}
        try:
            candidate = onnxoptimizer.optimize(model, [pass_name])
            sha = digest(candidate)
            row["sha256"] = sha
            if sha in seen:
                row["status"] = "duplicate_or_base"
                rows.append(row)
                continue
            seen.add(sha)
            ok, reason = structural(candidate)
            row["structure"] = reason
            if not ok:
                row["status"] = "structure_reject"
                rows.append(row)
                continue
            path = outdir / f"{pass_name}.onnx"
            onnx.save(candidate, path)
            row["path"] = str(path.relative_to(ROOT))
            with tempfile.TemporaryDirectory(prefix="c10_opt225_", dir="/tmp") as workdir:
                score = scoring.score_and_verify(
                    candidate, TASK, workdir, pass_name, require_correct=False
                )
            row["score"] = score
            if score is None:
                row["status"] = "unscorable"
            elif not score["correct"]:
                row["status"] = "known_wrong"
            elif score["cost"] < BASE_COST:
                row["status"] = "cheaper_known_correct"
            else:
                row["status"] = "not_cheaper"
        except Exception as exc:  # noqa: BLE001
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    result = {
        "task": TASK,
        "base_cost": BASE_COST,
        "passes": len(rows),
        "unique_outputs_including_base": len(seen),
        "rows": rows,
    }
    (HERE / "optimizer_sweep.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "passes", len(rows),
        "unique", len(seen),
        "cheaper_known_correct",
        sum(row.get("status") == "cheaper_known_correct" for row in rows),
    )


if __name__ == "__main__":
    main()
