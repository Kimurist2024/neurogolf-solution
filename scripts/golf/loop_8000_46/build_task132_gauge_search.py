#!/usr/bin/env python3
"""Enumerate small exact task132 gauges that expose A as Q's diagonal view."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "submission_base_8000.46.zip"
OUTPUT_DIR = Path(__file__).resolve().parent / "lane_task132_gauge_search"
TARGET = np.array([[0.5, 1.0], [-1.0, 0.0]], dtype=np.float64)


def attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto:
    return next(item for item in node.attribute if item.name == name)


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    initializer = next(item for item in model.graph.initializer if item.name == name)
    initializer.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name=name))


def solve_c(q: np.ndarray, f_gauge: np.ndarray) -> np.ndarray | None:
    c_gauge = np.empty((2, 2), dtype=np.float64)
    for t in range(2):
        matrix = np.empty((2, 2), dtype=np.float64)
        for m in range(2):
            matrix[m] = np.einsum("f,fc->c", f_gauge[m], q[:, :, m, t])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            return None
        c_gauge[t] = np.linalg.solve(matrix, TARGET[:, t])
    if abs(float(np.linalg.det(c_gauge))) < 1e-9:
        return None
    return c_gauge


def enumerate_gauges(q: np.ndarray, limit: int = 48, mode: str = "condition") -> list[dict[str, object]]:
    values = np.arange(-2.0, 2.0001, 0.25, dtype=np.float64)
    rows: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for a in values:
        for b in values:
            for c in values:
                for d in values:
                    f_gauge = np.array([[a, b], [c, d]], dtype=np.float64)
                    if abs(float(np.linalg.det(f_gauge))) < 1e-9:
                        continue
                    c_gauge = solve_c(q, f_gauge)
                    if c_gauge is None:
                        continue
                    q_new = np.einsum("if,jc,fcqv->ijqv", f_gauge, c_gauge, q)
                    if not np.array_equal(
                        np.einsum("mtmt->mt", q_new).astype(np.float32),
                        TARGET.astype(np.float32),
                    ):
                        continue
                    key = q_new.astype(np.float32).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    condition_score = (
                        max(float(np.linalg.cond(f_gauge)), float(np.linalg.cond(c_gauge))),
                        float(np.max(np.abs(q_new))),
                        float(np.max(np.abs(f_gauge))) + float(np.max(np.abs(c_gauge))),
                    )
                    joined = np.concatenate((f_gauge.ravel(), c_gauge.ravel(), q_new.ravel()))
                    dyadic_error = float(np.max(np.abs(joined * 4.0 - np.rint(joined * 4.0))))
                    simple_score = (
                        dyadic_error,
                        int(np.count_nonzero(joined)),
                        len(np.unique(joined.astype(np.float32))),
                        float(np.max(np.abs(joined))),
                        float(np.sum(np.abs(f_gauge - np.array([[0.5, 0.0], [-1.0, 1.0]])))),
                    )
                    rows.append(
                        {
                            "f": f_gauge.tolist(),
                            "c": c_gauge.tolist(),
                            "score": list(condition_score),
                            "simple_score": list(simple_score),
                        }
                    )
    if mode == "near":
        rows.sort(key=lambda row: (row["simple_score"][4], *row["simple_score"][:4]))
    else:
        score_key = "simple_score" if mode == "simple" else "score"
        rows.sort(key=lambda row: tuple(row[score_key]))

    # Keep numerically strong but structurally diverse Q tensors.
    selected: list[dict[str, object]] = []
    for row in rows:
        f_gauge = np.asarray(row["f"], dtype=np.float64)
        c_gauge = np.asarray(row["c"], dtype=np.float64)
        q_new = np.einsum("if,jc,fcqv->ijqv", f_gauge, c_gauge, q)
        if any(np.array_equal(q_new.astype(np.float32), np.asarray(item["q"], dtype=np.float32)) for item in selected):
            continue
        row["q"] = q_new.astype(np.float32).tolist()
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def build(source: onnx.ModelProto, f_values: object, c_values: object) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    arrays = {item.name: numpy_helper.to_array(item).astype(np.float64) for item in model.graph.initializer}
    f_gauge = np.asarray(f_values, dtype=np.float64)
    c_gauge = np.asarray(c_values, dtype=np.float64)
    q_new = np.einsum("if,jc,fcqv->ijqv", f_gauge, c_gauge, arrays["Q"])
    pc_new = np.einsum("if,fpu->ipu", np.linalg.inv(f_gauge).T, arrays["PC"])
    l_new = np.einsum("jc,csw->jsw", np.linalg.inv(c_gauge).T, arrays["L"])
    h_new = arrays["H"] * 100_000.0

    if not np.array_equal(np.einsum("mtmt->mt", q_new).astype(np.float32), TARGET.astype(np.float32)):
        raise RuntimeError("repeated Q view is not the exact target")
    replace_initializer(model, "PC", pc_new)
    replace_initializer(model, "Q", q_new)
    replace_initializer(model, "L", l_new)
    replace_initializer(model, "H", h_new)
    kept = [item for item in model.graph.initializer if item.name != "A"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    node = model.graph.node[0]
    lhs, rhs = attribute(node, "equation").s.decode().split("->")
    terms = lhs.split(",")
    names = list(node.input)
    for index, name in enumerate(names):
        if name == "A":
            names[index] = "Q"
            terms[index] = {"mt": "mtmt", "lR": "lRlR"}[terms[index]]
    del node.input[:]
    node.input.extend(names)
    attribute(node, "equation").s = (",".join(terms) + "->" + rhs).encode()
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("condition", "simple", "near"), default="condition")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASELINE) as archive:
        source = onnx.load_model_from_string(archive.read("task132.onnx"))
    q = numpy_helper.to_array(next(item for item in source.graph.initializer if item.name == "Q")).astype(np.float64)
    gauges = enumerate_gauges(q, limit=args.limit, mode=args.mode)
    rows = []
    for index, gauge in enumerate(gauges):
        try:
            model = build(source, gauge["f"], gauge["c"])
            path = output_dir / f"task132_g{index:02d}.onnx"
            onnx.save(model, path)
            rows.append(
                {
                    "index": index,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "f": gauge["f"],
                    "c": gauge["c"],
                    "score": gauge["score"],
                    "simple_score": gauge["simple_score"],
                }
            )
        except Exception as exc:
            rows.append({"index": index, "error": repr(exc), **gauge})
    manifest = output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"built": sum("path" in row for row in rows), "errors": sum("error" in row for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
