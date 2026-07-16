#!/usr/bin/env python3
"""Finite proof of the task343 exact-178 classifier on generator support.

The generator's correlations depend only on equality between colors, not on
their numeric identities.  Consequently every possible draw is represented
by a restricted-growth string using at most three canonical nonzero colors.
This script enumerates all such strings, all length vectors, both flip modes,
and every allowed visible width.  It then evaluates the four dynamic-Conv
correlations and the period-6/period-8 output rule used by exact178.onnx.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def restricted_growth_strings(size: int) -> np.ndarray:
    """Return canonical equality patterns of length ``size`` with <=3 labels."""
    rows: list[tuple[int, ...]] = []
    work = [0] * size

    def visit(index: int, maximum: int) -> None:
        if index == size:
            rows.append(tuple(work))
            return
        for value in range(min(maximum + 1, 2) + 1):
            work[index] = value
            visit(index + 1, max(maximum, value))

    # A nonempty restricted-growth string always starts with label zero.
    work[0] = 0
    visit(1, 0)
    return np.asarray(rows, dtype=np.uint8)


def affine_pairs(
    height: int, width: int, spec: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int, int], ...]:
    dilation_h, pad_top, dilation_w, pad_left = spec
    return tuple(
        (row, col, dilation_h * row - pad_top, dilation_w * col - pad_left)
        for row in range(height)
        if 0 <= dilation_h * row - pad_top < height
        for col in range(width)
        if 0 <= dilation_w * col - pad_left < width
    )


def correlation(grid: np.ndarray, spec: tuple[int, int, int, int]) -> np.ndarray:
    """Simulate a scalar Conv(input,input) on the truthful one-hot support."""
    value = np.zeros(grid.shape[0], dtype=np.int16)
    for row, col, other_row, other_col in affine_pairs(5, 15, spec):
        value += grid[:, row, col] == grid[:, other_row, other_col]
    return value


def main() -> None:
    specs = {
        "q4": (1, 0, 1, 4),
        "q11": (1, 0, 1, 11),
        "z7": (30, -4, 30, -7),
        "z11": (30, -4, 30, -11),
    }
    pattern_counts = {
        str(size): int(restricted_growth_strings(size).shape[0])
        for size in range(3, 13)
    }
    parameter_states = 0
    accepted_generator_states = 0
    rejected_input_equals_output = 0
    use6_states = 0
    use8_states = 0
    mismatches = 0
    first_mismatch = None

    for period in (3, 4):
        for lengths in itertools.product((1, 2, 3), repeat=period):
            patterns = restricted_growth_strings(sum(lengths))
            batch = patterns.shape[0]
            base = np.zeros((batch, 5, period), dtype=np.uint8)
            offset = 0
            for col, length in enumerate(lengths):
                for down in range(length):
                    # Add one so canonical labels remain nonzero ARC colors.
                    base[:, 4 - down, col] = patterns[:, offset + down] + 1
                offset += length

            for flip in (0, 1):
                source = np.arange(15) % period
                if flip:
                    blocks = np.arange(15) // period
                    source = np.where(blocks % 2, period - 1 - source, source)
                expected = base[:, :, source]

                for tail in range(period):
                    visible = (flip + 2) * period + tail
                    grid = expected.copy()
                    grid[:, :, visible:] = 0
                    parameter_states += batch

                    # generate() resamples these states, because draw() returns
                    # false when no cell changes.  Keep an explicit census.
                    changed = np.any(grid != expected, axis=(1, 2))
                    accepted_generator_states += int(changed.sum())
                    rejected_input_equals_output += int((~changed).sum())

                    q4 = correlation(grid, specs["q4"])
                    q11 = correlation(grid, specs["q11"])
                    z7 = correlation(grid, specs["z7"])
                    z11 = correlation(grid, specs["z11"])
                    use6 = q11 + 2 * z7 + 35 * z11 > q4
                    use6_states += int(use6.sum())
                    use8_states += int((~use6).sum())

                    selected_period = np.where(use6, 6, 8)
                    candidate = np.empty_like(expected)
                    for index in range(batch):
                        candidate[index] = grid[index][
                            :, np.arange(15) % int(selected_period[index])
                        ]
                    bad = np.any(candidate != expected, axis=(1, 2))
                    mismatches += int(bad.sum())
                    if first_mismatch is None and np.any(bad):
                        index = int(np.flatnonzero(bad)[0])
                        first_mismatch = {
                            "period": period,
                            "lengths": list(lengths),
                            "canonical_colors": patterns[index].tolist(),
                            "flip": flip,
                            "visible": visible,
                            "q4": int(q4[index]),
                            "q11": int(q11[index]),
                            "z7": int(z7[index]),
                            "z11": int(z11[index]),
                            "selected_period": int(selected_period[index]),
                        }

    model_path = HERE / "controls/exact178.onnx"
    result = {
        "scope": {
            "periods": [3, 4],
            "length_values": [1, 2, 3],
            "canonical_nonzero_color_labels_at_most": 3,
            "flip_values": [0, 1],
            "visible_formula": "(flip + 2) * period + tail, 0 <= tail < period",
            "reason_color_canonicalization_is_complete": (
                "all four probes are self-correlations, so their values depend "
                "only on color equality; the gather route preserves identities"
            ),
        },
        "classifier": "q11 + 2*z7 + 35*z11 > q4 selects period 6, else 8",
        "specs": {name: list(spec) for name, spec in specs.items()},
        "restricted_growth_counts": pattern_counts,
        "parameter_states": parameter_states,
        "accepted_generator_states": accepted_generator_states,
        "rejected_input_equals_output_states": rejected_input_equals_output,
        "use6_states": use6_states,
        "use8_states": use8_states,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "exact178_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "verdict": "universal_on_generator_support" if mismatches == 0 else "failed",
    }
    output = HERE / "finite_rule_proof.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
