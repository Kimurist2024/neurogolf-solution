#!/usr/bin/env python3
"""Exhaustively prove the task315 candidate on all 3^9 generator grids."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
MODEL = HERE / "task315_tied_color_factor.onnx"
OUT = HERE / "task315_complete_generator_proof.json"


def main() -> None:
    model = onnx.load(MODEL)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item), dtype=np.float64)
        for item in model.graph.initializer
    }
    feature = arrays["colorF"][:, :3]
    common = feature.T @ arrays["L1"] @ feature
    mode = np.einsum("am,tab,bc->tmc", feature, arrays["L2"], feature)
    color = common[None, :, :] * mode

    position = arrays["V"][:, :9]
    source_position = position[:, :3]
    routing = np.ones((2, 9, 3), dtype=np.float64)
    for name in ("S0", "S1", "S2", "S3"):
        factor = np.einsum("ah,ab,bp->hp", position, arrays[name], source_position)
        routing *= factor[None, :, :]
    for name in ("D0", "D1"):
        factor = np.einsum("ah,tab,bp->thp", position, arrays[name], source_position)
        routing *= factor

    # W[m,p,q,c,h,w] is the exact real-arithmetic linear map represented by
    # the factor graph over the generator's nonzero input support.
    weight = np.einsum("tmc,thp,tkq->mpqchk", color, routing, routing)
    right = wrong = 0
    minimum_positive = float("inf")
    maximum_nonpositive = -float("inf")
    first_failure = None
    chunk_size = 729
    all_grids = itertools.product(range(3), repeat=9)
    for chunk_index in range(27):
        flat = np.asarray(list(itertools.islice(all_grids, chunk_size)), dtype=np.int64)
        grids = flat.reshape(-1, 3, 3)
        logits = np.zeros((len(grids), 3, 9, 9), dtype=np.float64)
        for row in range(3):
            for col in range(3):
                logits += weight[grids[:, row, col], row, col]
        expected = np.zeros_like(logits, dtype=bool)
        for gate_row in range(3):
            for source_row in range(3):
                for gate_col in range(3):
                    for source_col in range(3):
                        output_row = 3 * gate_row + source_row
                        output_col = 3 * gate_col + source_col
                        values = grids[:, source_row, source_col] * (
                            grids[:, gate_row, gate_col] > 1
                        )
                        expected[
                            np.arange(len(grids)), values, output_row, output_col
                        ] = True
        predicted = logits > 0.0
        matches = np.all(predicted == expected, axis=(1, 2, 3))
        right += int(np.count_nonzero(matches))
        wrong += int(np.count_nonzero(~matches))
        minimum_positive = min(minimum_positive, float(logits[expected].min()))
        maximum_nonpositive = max(maximum_nonpositive, float(logits[~expected].max()))
        if first_failure is None and not np.all(matches):
            index = int(np.flatnonzero(~matches)[0])
            first_failure = {
                "global_index": chunk_index * chunk_size + index,
                "grid": grids[index].tolist(),
                "different_cells": int(np.count_nonzero(predicted[index] != expected[index])),
            }
    result = {
        "task": 315,
        "domain": "all 3^9 grids over generator colors {0,1,2}",
        "total": 3**9,
        "right": right,
        "wrong": wrong,
        "first_failure": first_failure,
        "minimum_positive_real_arithmetic": minimum_positive,
        "maximum_nonpositive_real_arithmetic": maximum_nonpositive,
        "padding_argument": (
            "colorF columns 3..9 and V columns 9..29 are exactly zero, so "
            "channels 3..9 and spatial padding are exactly non-positive zero"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
