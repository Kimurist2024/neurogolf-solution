#!/usr/bin/env python3
"""Extract and audit exact task280/task396 models from the 7999.13 ZIP."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_7999.13.zip"
TASKS = (280, 396)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(str(HERE.parent / "lane_b16"))

from audit_exact import known_dual, structure  # noqa: E402
from audit_candidates import runtime_shapes  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ort.set_default_logger_severity(4)
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            (baseline / name).write_bytes(archive.read(name))
    rows = {}
    for task in TASKS:
        path = baseline / f"task{task:03d}.onnx"
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(prefix=f"b17_{task:03d}_", dir="/tmp") as workdir:
            score = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label="exact", require_correct=False
            )
        try:
            trace = runtime_shapes(model, task)
        except Exception as exc:  # noqa: BLE001
            trace = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
        rows[str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "file_bytes": path.stat().st_size,
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "actual_score": score,
            "structure": structure(model),
            "runtime_shapes": trace,
            "known_dual": known_dual(model, task),
            "ops": [node.op_type for node in model.graph.node],
            "initializers_names": [item.name for item in model.graph.initializer],
            "max_einsum_inputs": max(
                (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
                default=0,
            ),
        }
        print(task, rows[str(task)]["sha256"], score, "cloak", trace.get("shape_cloak"), flush=True)
    report = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha256(BASE_ZIP),
        "models": rows,
    }
    (HERE / "exact_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
