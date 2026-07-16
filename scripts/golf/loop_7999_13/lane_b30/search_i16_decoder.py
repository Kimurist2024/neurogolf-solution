#!/usr/bin/env python3
"""Exact modular-decoder search for a task345 int16 row carrier.

The current model stores each output row as ``R + 2047*G`` and uses int32
overflow in the output Einsum to recover the black/red/gray channels.  This
probe asks whether the row can instead be stored as the signed int16 value of
``R + K*G``.  A successful K must first be information preserving on every
legal generator row.  For each output color, an exact interval intersection
then solves for a uint32 multiplier C whose top ten product bits are precisely
the complement of that color mask.  Those are the bits read by the existing
reverse-power-of-two width factor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODULUS = 1 << 32
TOP_SHIFT = 22
HERE = Path(__file__).resolve().parent


def enumerate_rows() -> dict[tuple[int, int], tuple[int, ...]]:
    parameter_cases: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def recurse(
        start: int,
        starts: tuple[int, ...],
        rows: tuple[int, ...],
        cols: tuple[int, ...],
    ) -> None:
        if start + 1 >= 10:
            parameter_cases.append((rows, cols, starts))
            return
        next_starts = starts + (start,)
        recurse(start + 2, next_starts, rows, cols)
        for row in range(2, 7):
            for offset in (-1, 0, 1):
                for step in (3, 4):
                    recurse(
                        start + step,
                        next_starts,
                        rows + (row,),
                        cols + (start + offset,),
                    )

    recurse(1, (), (), ())
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    legal_count = 0
    for rows, cols, starts in parameter_cases:
        output = [[0] * 10 for _ in range(10)]
        for row, col in zip(rows, cols):
            output[row][col] = 5
        legal = False
        for start in starts:
            row, col = 9, start
            output[row][col] = 2
            while row > 0:
                if output[row - 1][col] == 5:
                    col += 1
                    legal = True
                else:
                    row -= 1
                output[row][col] = 2
        if not legal:
            continue
        legal_count += 1
        for row in output:
            red = sum((value == 2) << col for col, value in enumerate(row))
            gray = sum((value == 5) << col for col, value in enumerate(row))
            key = (red, gray)
            labels = tuple(row)
            prior = result.get(key)
            if prior is not None and prior != labels:
                raise AssertionError(f"non-unique row state: {key}")
            result[key] = labels
    if legal_count != 16540 or len(result) != 666:
        raise AssertionError((legal_count, len(result)))
    return result


def signed_i16(value: int) -> int:
    return ((value + (1 << 15)) % (1 << 16)) - (1 << 15)


def encoded_rows(
    rows: dict[tuple[int, int], tuple[int, ...]],
    multiplier: int,
    wrap: bool,
    unsigned: bool,
    combine: str,
) -> dict[int, tuple[int, ...]] | None:
    encoded: dict[int, tuple[int, ...]] = {}
    for (red, gray), labels in rows.items():
        product = multiplier * gray
        if combine == "add":
            raw = red + product
        elif combine == "xor":
            raw = (red ^ product) & 0xFFFF
        else:
            raise ValueError(combine)
        if not wrap and not -(1 << 15) <= raw < (1 << 15):
            return None
        code = raw % (1 << 16) if unsigned else signed_i16(raw)
        prior = encoded.get(code)
        if prior is not None and prior != labels:
            return None
        encoded[code] = labels
    return encoded


def merge_intervals(
    intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for low, high in intervals[1:]:
        old_low, old_high = merged[-1]
        if low <= old_high + 1:
            merged[-1] = (old_low, max(old_high, high))
        else:
            merged.append((low, high))
    return merged


def positive_preimage(
    factor: int, low: int, high: int
) -> list[tuple[int, int]]:
    """Return all C in uint32 space with factor*C mod 2^32 in [low, high]."""
    if not 0 <= low <= high < MODULUS:
        raise ValueError((low, high))
    if factor == 0:
        return [(0, MODULUS - 1)] if low == 0 else []
    intervals: list[tuple[int, int]] = []
    for quotient in range(factor):
        base = quotient * MODULUS
        first = (base + low + factor - 1) // factor
        last = (base + high) // factor
        if first <= last and first < MODULUS and last >= 0:
            intervals.append((max(0, first), min(MODULUS - 1, last)))
    return intervals


def multiplier_preimage(
    signed_factor: int, top_bits: int
) -> list[tuple[int, int]]:
    low = top_bits << TOP_SHIFT
    high = ((top_bits + 1) << TOP_SHIFT) - 1
    if signed_factor > 0:
        return positive_preimage(signed_factor, low, high)
    factor = -signed_factor
    # -x mod M maps [low, high] to [M-high, M-low], except the interval that
    # includes zero, which wraps into two pieces.
    if low == 0:
        return merge_intervals(
            positive_preimage(factor, 0, 0)
            + positive_preimage(factor, MODULUS - high, MODULUS - 1)
        )
    return positive_preimage(factor, MODULUS - high, MODULUS - low)


def intersect(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        low = max(left[i][0], right[j][0])
        high = min(left[i][1], right[j][1])
        if low <= high:
            result.append((low, high))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def constrain_positive_factor(
    feasible: list[tuple[int, int]],
    factor: int,
    low: int,
    high: int,
) -> list[tuple[int, int]]:
    """Intersect without materializing all ``factor`` modular preimages."""
    result: list[tuple[int, int]] = []
    for current_low, current_high in feasible:
        first_quotient = (factor * current_low) // MODULUS
        last_quotient = (factor * current_high) // MODULUS
        for quotient in range(first_quotient, last_quotient + 1):
            base = quotient * MODULUS
            allowed_low = (base + low + factor - 1) // factor
            allowed_high = (base + high) // factor
            clipped_low = max(current_low, allowed_low)
            clipped_high = min(current_high, allowed_high)
            if clipped_low <= clipped_high:
                result.append((clipped_low, clipped_high))
    return merge_intervals(result)


def constrain(
    feasible: list[tuple[int, int]], signed_factor: int, top_bits: int
) -> list[tuple[int, int]]:
    low = top_bits << TOP_SHIFT
    high = ((top_bits + 1) << TOP_SHIFT) - 1
    if signed_factor > 0:
        return constrain_positive_factor(feasible, signed_factor, low, high)
    factor = -signed_factor
    if low == 0:
        return merge_intervals(
            constrain_positive_factor(feasible, factor, 0, 0)
            + constrain_positive_factor(
                feasible, factor, MODULUS - high, MODULUS - 1
            )
        )
    return constrain_positive_factor(
        feasible, factor, MODULUS - high, MODULUS - low
    )


def target_bits(labels: tuple[int, ...], color: int) -> int:
    # The existing width weights are [2^9, ..., 2^0].  After multiplication,
    # output column j reads bit 22+j as its sign bit.  Off cells therefore need
    # a one in that bit; on cells need zero.
    return sum((value != color) << col for col, value in enumerate(labels))


def solve_color(
    encoded: dict[int, tuple[int, ...]], color: int
) -> tuple[int | None, int, list[dict[str, int]]]:
    constraints = [
        (code, target_bits(labels, color))
        for code, labels in encoded.items()
        if code != 0
    ]
    constraints.sort(key=lambda item: (abs(item[0]), item[0], item[1]))
    feasible = [(0, MODULUS - 1)]
    trace: list[dict[str, int]] = []
    for ordinal, (code, bits) in enumerate(constraints, 1):
        feasible = constrain(feasible, code, bits)
        trace.append(
            {
                "ordinal": ordinal,
                "code": code,
                "target_top_bits": bits,
                "feasible_interval_count": len(feasible),
                "feasible_integer_count": sum(high - low + 1 for low, high in feasible),
            }
        )
        if not feasible:
            return None, ordinal, trace
    return feasible[0][0], len(constraints), trace


def classify(code: int, coefficient: int, col: int) -> bool:
    width = 1 << (9 - col)
    value = ((code & 0xFFFFFFFF) * coefficient * width) & 0xFFFFFFFF
    return 0 < value < 0x80000000


def verify_solution(
    encoded: dict[int, tuple[int, ...]], coefficients: dict[int, int]
) -> bool:
    for code, labels in encoded.items():
        for col, label in enumerate(labels):
            for color in (0, 2, 5):
                if classify(code, coefficients[color], col) != (label == color):
                    return False
    return True


def scan(args: argparse.Namespace) -> dict[str, object]:
    rows = enumerate_rows()
    baseline_encoded = {
        red + 2047 * gray: labels for (red, gray), labels in rows.items()
    }
    baseline_coefficients = {0: 4198404, 2: (-4196355) & 0xFFFFFFFF, 5: (-2050) & 0xFFFFFFFF}
    baseline_exact = verify_solution(baseline_encoded, baseline_coefficients)

    candidates: list[dict[str, object]] = []
    collision_free = 0
    for multiplier in range(args.minimum, args.maximum + 1):
        encoded = encoded_rows(rows, multiplier, args.wrap, args.unsigned, args.combine)
        if encoded is None:
            continue
        collision_free += 1
        coefficients: dict[int, int] = {}
        failure: dict[str, int] | None = None
        traces: dict[str, list[dict[str, int]]] = {}
        for color in (0, 2, 5):
            coefficient, constraints_used, trace = solve_color(encoded, color)
            traces[str(color)] = trace
            if coefficient is None:
                failure = {"color": color, "constraints_used": constraints_used}
                break
            coefficients[color] = coefficient
        if failure is not None:
            continue
        if not verify_solution(encoded, coefficients):
            raise AssertionError((multiplier, coefficients))
        candidates.append(
            {
                "gray_multiplier": multiplier,
                "encoded_row_count": len(encoded),
                "coefficients_uint32": {str(k): v for k, v in coefficients.items()},
                "coefficients_int32": {
                    str(k): v if v < (1 << 31) else v - MODULUS
                    for k, v in coefficients.items()
                },
                "trace": traces,
            }
        )
        if len(candidates) >= args.limit:
            break

    result: dict[str, object] = {
        "legal_generator_cases": 16540,
        "unique_row_states": len(rows),
        "baseline_decoder_exact": baseline_exact,
        "range": [args.minimum, args.maximum],
        "wrap": args.wrap,
        "unsigned": args.unsigned,
        "combine": args.combine,
        "collision_free_multipliers": collision_free,
        "exact_decoder_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", type=int, default=-4096)
    parser.add_argument("--maximum", type=int, default=4096)
    parser.add_argument("--wrap", action="store_true")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--combine", choices=("add", "xor"), default="add")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "i16_decoder_search.json")
    args = parser.parse_args()
    result = scan(args)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "exact_decoder_candidates"
            }
            | {"exact_decoder_candidate_count": len(result["exact_decoder_candidates"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
