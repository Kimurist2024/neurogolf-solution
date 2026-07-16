#!/usr/bin/env python3
"""Find 2-D Einsum initializer operands with parameter-saving low rank."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode("ascii")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/einsum_low_rank_audit.json"),
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            payload = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model(io.BytesIO(payload))
            initializers = {item.name: item for item in model.graph.initializer}
            for node_index, node in enumerate(model.graph.node):
                if node.op_type != "Einsum":
                    continue
                eq = equation(node)
                if not eq or "->" not in eq:
                    continue
                operands = eq.split("->", 1)[0].split(",")
                if len(operands) != len(node.input):
                    continue
                for input_index, (name, subscripts) in enumerate(zip(node.input, operands)):
                    initializer = initializers.get(name)
                    if initializer is None or len(initializer.dims) != 2:
                        continue
                    if len(subscripts) != 2 or subscripts[0] == subscripts[1]:
                        continue
                    array = np.asarray(numpy_helper.to_array(initializer), dtype=np.float64)
                    if min(array.shape) < 2:
                        continue
                    u, singular, vh = np.linalg.svd(array, full_matrices=False)
                    scale = singular[0] if singular.size and singular[0] else 1.0
                    rank = int(np.sum(singular > scale * args.relative_tolerance))
                    original = int(array.size)
                    factored = int(rank * (array.shape[0] + array.shape[1]))
                    if rank and factored < original:
                        reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank, :]
                        abs_error = float(np.max(np.abs(reconstructed - array)))
                        rows.append(
                            {
                                "task": task,
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "node_index": node_index,
                                "input_index": input_index,
                                "initializer": name,
                                "subscripts": subscripts,
                                "shape": list(array.shape),
                                "rank": rank,
                                "original_params": original,
                                "factored_params": factored,
                                "saving": original - factored,
                                "max_abs_reconstruction_error_f64": abs_error,
                                "singular_values": singular[: min(8, len(singular))].tolist(),
                            }
                        )
    rows.sort(key=lambda row: (-int(row["saving"]), int(row["task"])))
    document = {"source_zip": str(args.zip), "candidate_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
