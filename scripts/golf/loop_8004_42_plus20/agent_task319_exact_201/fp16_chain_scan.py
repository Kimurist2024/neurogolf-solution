#!/usr/bin/env python3
"""Exhaustively test folding task319's fp16 Floor/Add/Sub chain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    bits = np.arange(1 << 16, dtype=np.uint16)
    values = bits.view(np.float16)
    values = values[np.isfinite(values) & (values >= np.float16(0.5))]

    # Mirror the authority's fp16 Log -> Mul(1.442) -> Add(.005) -> Floor -> Sub(7).
    log2_like = (
        np.log(values).astype(np.float16) * np.float16(1.442)
    ).astype(np.float16)
    authority = (
        np.floor((log2_like + np.float16(0.005)).astype(np.float16)).astype(np.float16)
        - np.float16(7.0)
    ).astype(np.float16)

    all_constants = bits.view(np.float16)
    constants = all_constants[
        np.isfinite(all_constants)
        & (all_constants >= np.float16(-8.0))
        & (all_constants <= np.float16(-6.0))
    ]
    best: tuple[int, float, int] | None = None
    exact_constants: list[float] = []
    for constant in constants:
        folded = np.floor((log2_like + constant).astype(np.float16)).astype(np.float16)
        differences = int(np.count_nonzero(folded != authority))
        candidate = (differences, float(constant), int(constant.view(np.uint16)))
        if best is None or candidate < best:
            best = candidate
        if differences == 0:
            exact_constants.append(float(constant))

    if best is None:
        raise RuntimeError("constant scan unexpectedly empty")
    report = {
        "domain": "all finite fp16 x >= 0.5",
        "domain_size": int(values.size),
        "constant_interval": [-8.0, -6.0],
        "constants_tested": int(constants.size),
        "exact_constants": exact_constants,
        "best": {
            "constant": best[1],
            "fp16_bits": best[2],
            "different_values": best[0],
        },
        "decision": "REJECT_NO_EXACT_FOLD" if not exact_constants else "EXACT_FOLD_EXISTS",
    }
    (HERE / "fp16_chain_scan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
