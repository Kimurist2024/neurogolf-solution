#!/usr/bin/env python3
"""Read-only score scan of byte-distinct historical candidates for B7 tasks."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (30, 132, 175, 199, 212, 240, 304)
BASE_COST = {30: 162, 132: 316, 175: 166, 199: 261, 212: 240, 240: 172, 304: 180}


def static_cost(model: onnx.ModelProto) -> int | None:
    """Cheap lower-bound-like cost used only to reject already dominated models."""
    try:
        params = sum(int(np.prod(item.dims)) for item in model.graph.initializer)
        graph = shape_inference.infer_shapes(model, strict_mode=False).graph
        initializer_names = {item.name for item in graph.initializer}
        memory = 0
        for value in list(graph.value_info) + list(graph.output):
            if value.name in initializer_names or value.name in {"input", "output"}:
                continue
            tensor = value.type.tensor_type
            if not tensor.HasField("shape"):
                continue
            elements = 1
            for dim in tensor.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    return None
                elements *= dim.dim_value
            memory += elements * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        return int(params + memory)
    except Exception:
        return None


def main() -> None:
    onnxruntime.set_default_logger_severity(3)
    rows: list[dict[str, object]] = []
    candidates: list[tuple[int, Path]] = []
    for path in ROOT.rglob("*.onnx"):
        text = path.name.lower()
        for task in TASKS:
            if f"task{task:03d}" in text:
                candidates.append((task, path))
                break

    seen: set[tuple[int, str]] = set()
    for task, path in sorted(candidates, key=lambda item: str(item[1])):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        key = (task, digest)
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
        }
        try:
            model = onnx.load(path)
            estimated = static_cost(model)
            record["static_cost"] = estimated
            if estimated is not None and estimated >= BASE_COST[task]:
                record.update(status="dominated_static", cheaper=False)
                rows.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix=f"b7_scan_{task:03d}_") as workdir:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = scoring.score_and_verify(
                        model, task, workdir, label="b7", require_correct=False
                    )
            if result is None:
                record["status"] = "unscorable"
            else:
                record.update(
                    status="ok",
                    cost=int(result["cost"]),
                    memory=int(result["memory"]),
                    params=int(result["params"]),
                    correct=bool(result["correct"]),
                    cheaper=int(result["cost"]) < BASE_COST[task],
                )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    output = HERE / "existing_scan.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    winners = [row for row in rows if row.get("correct") and row.get("cheaper")]
    print("WINNERS", json.dumps(winners, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
