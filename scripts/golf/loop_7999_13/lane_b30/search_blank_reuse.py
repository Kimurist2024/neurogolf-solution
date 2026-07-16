#!/usr/bin/env python3
"""Search an exact decoder gauge that makes an existing scalar a blank row.

Replacing the dedicated one-parameter zero initializer would turn the legal,
pre-scaled swapped-Conv graph from cost 389 into cost 388.  The existing
decoder gauge cannot reuse any scalar, but the multiplier solution is not
unique.  This script enumerates all 16,540 legal generator cases, collects the
complete value set of every int32 scalar, and solves the exact modular decoder
again with that value set constrained to decode to all-off.
"""

from __future__ import annotations

import json
from pathlib import Path

from search_i16_decoder import classify, solve_color, verify_solution


HERE = Path(__file__).resolve().parent


def parameter_cases() -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    result: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def recurse(
        start: int,
        starts: tuple[int, ...],
        rows: tuple[int, ...],
        cols: tuple[int, ...],
    ) -> None:
        if start + 1 >= 10:
            result.append((rows, cols, starts))
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
    return result


def evidence() -> tuple[
    dict[int, tuple[int, ...]], dict[str, set[int]], int
]:
    output_codes: dict[int, tuple[int, ...]] = {}
    scalar_values: dict[str, set[int]] = {}
    legal_count = 0

    def record(name: str, value: int) -> int:
        scalar_values.setdefault(name, set()).add(value)
        return value

    for rows, cols, starts in parameter_cases():
        output = [[0] * 10 for _ in range(10)]
        gray_masks = {row: 0 for row in range(2, 7)}
        for row, col in zip(rows, cols):
            output[row][col] = 5
            gray_masks[row] |= 1 << col
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

        red9 = sum((value == 2) << col for col, value in enumerate(output[9]))
        r9 = record("r9", red9)
        gray = {
            row: record(f"g{row}", -2047 * gray_masks[row])
            for row in range(2, 7)
        }
        hit = {
            row: record(f"h{row}", r9 & gray[row])
            for row in range(2, 7)
        }
        qa6 = record("qa6", r9 + hit[6])
        q7 = record("q7", qa6 + hit[6])
        qa5 = record("qa5", qa6 + hit[5])
        q6 = record("q6", qa5 + hit[5])
        qa4 = record("qa4", qa5 + hit[4])
        q5 = record("q5", qa4 + hit[4])
        qa3 = record("qa3", qa4 + hit[3])
        q4 = record("q4", qa3 + hit[3])
        q0 = record("q0", qa3 + hit[2])
        q3 = record("q3", q0 + hit[2])
        packed = {
            2: record("p2", q0 - gray[2]),
            3: record("p3", q3 - gray[3]),
            4: record("p4", q4 - gray[4]),
            5: record("p5", q5 - gray[5]),
            6: record("p6", q6 - gray[6]),
        }
        expected_packed = {
            row: sum((value == 2) << col for col, value in enumerate(output[row]))
            + 2047 * sum((value == 5) << col for col, value in enumerate(output[row]))
            for row in range(10)
        }
        actual_packed = {
            0: q0,
            1: q0,
            **packed,
            7: q7,
            8: r9,
            9: r9,
        }
        if actual_packed != expected_packed:
            raise AssertionError((rows, cols, starts, actual_packed, expected_packed))
        for row, code in expected_packed.items():
            labels = tuple(output[row])
            prior = output_codes.get(code)
            if prior is not None and prior != labels:
                raise AssertionError((code, prior, labels))
            output_codes[code] = labels

    if legal_count != 16540 or len(output_codes) != 666:
        raise AssertionError((legal_count, len(output_codes)))
    return output_codes, scalar_values, legal_count


def main() -> int:
    output_codes, scalar_values, legal_count = evidence()
    off_labels = (-1,) * 10
    rows: list[dict[str, object]] = []
    for name, values in sorted(scalar_values.items()):
        augmented = dict(output_codes)
        conflicts: list[int] = []
        for value in values:
            if value == 0:
                continue
            prior = augmented.get(value)
            if prior is not None and prior != off_labels:
                conflicts.append(value)
            else:
                augmented[value] = off_labels
        row: dict[str, object] = {
            "scalar": name,
            "unique_values": len(values),
            "nonzero_values": sum(value != 0 for value in values),
            "direct_output_code_conflicts": len(conflicts),
            "first_conflicts": sorted(conflicts)[:12],
        }
        if conflicts:
            row["status"] = "semantic_conflict"
            rows.append(row)
            continue
        coefficients: dict[int, int] = {}
        failure: dict[str, int] | None = None
        for color in (0, 2, 5):
            coefficient, constraints_used, _ = solve_color(augmented, color)
            if coefficient is None:
                failure = {"color": color, "constraints_used": constraints_used}
                break
            coefficients[color] = coefficient
        if failure is not None:
            row.update(status="no_exact_gauge", failure=failure)
            rows.append(row)
            continue
        if not verify_solution(output_codes, coefficients):
            raise AssertionError((name, coefficients))
        blank_ok = all(
            not classify(value, coefficient, col)
            for value in values
            for coefficient in coefficients.values()
            for col in range(10)
        )
        if not blank_ok:
            raise AssertionError((name, coefficients))
        row.update(
            status="exact_blank_gauge",
            coefficients_uint32={str(k): v for k, v in coefficients.items()},
            coefficients_int32={
                str(k): v if v < (1 << 31) else v - (1 << 32)
                for k, v in coefficients.items()
            },
        )
        rows.append(row)

    result = {
        "legal_generator_cases": legal_count,
        "unique_output_row_codes": len(output_codes),
        "scalar_count": len(rows),
        "exact_blank_gauges": [row for row in rows if row["status"] == "exact_blank_gauge"],
        "rows": rows,
    }
    path = HERE / "blank_reuse_search.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "legal_generator_cases": legal_count,
                "unique_output_row_codes": len(output_codes),
                "scalar_count": len(rows),
                "exact_blank_gauge_count": len(result["exact_blank_gauges"]),
                "output": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
