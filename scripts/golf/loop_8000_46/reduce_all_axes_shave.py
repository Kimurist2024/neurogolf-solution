#!/usr/bin/env python3
"""Remove explicit constant Reduce axes when they enumerate every input axis."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(__file__).resolve().parent / "submission_8000.46_wave4_safe_meta.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lane_reduce_all_axes"


def shape_rank(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result: dict[str, int] = {}
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if item.type.HasField("tensor_type") and item.type.tensor_type.HasField("shape"):
            result[item.name] = len(item.type.tensor_type.shape.dim)
    return result


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]], int]:
    model = copy.deepcopy(model)
    ranks = shape_rank(model)
    initializers = {item.name: item for item in model.graph.initializer}
    conversions: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if not node.op_type.startswith("Reduce") or len(node.input) < 2 or not node.input[1]:
            continue
        axes_init = initializers.get(node.input[1])
        rank = ranks.get(node.input[0])
        if axes_init is None or rank is None or rank == 0:
            continue
        axes = [int(value) for value in np.asarray(numpy_helper.to_array(axes_init)).reshape(-1)]
        normalized = sorted({axis % rank for axis in axes})
        if normalized != list(range(rank)) or len(axes) != rank:
            continue
        axes_name = node.input[1]
        del node.input[1:]
        conversions.append(
            {"node_index": index, "op": node.op_type, "axes": axes, "initializer": axes_name, "rank": rank}
        )
    if not conversions:
        raise RuntimeError("no explicit reduce-all axes")

    uses = Counter(name for node in model.graph.node for name in node.input if name)
    uses.update(item.name for item in model.graph.output)
    saving = 0
    kept = []
    for item in model.graph.initializer:
        if uses[item.name] == 0:
            saving += int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1
        else:
            kept.append(item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if saving <= 0:
        raise RuntimeError("converted axes are still shared; no parameter saving")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, conversions, saving


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.source) as archive:
        for task in range(1, 401):
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            try:
                candidate, conversions, saving = build(model)
            except RuntimeError as exc:
                if str(exc) not in {"no explicit reduce-all axes", "converted axes are still shared; no parameter saving"}:
                    errors.append({"task": task, "error": repr(exc)})
                continue
            except Exception as exc:
                errors.append({"task": task, "error": repr(exc)})
                continue
            path = output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, path)
            rows.append(
                {
                    "task": task,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "parameter_saving": saving,
                    "conversions": conversions,
                }
            )
    report = {"source": str(args.source), "candidate_count": len(rows), "rows": rows, "errors": errors}
    manifest = output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "errors": len(errors), "manifest": str(manifest.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
