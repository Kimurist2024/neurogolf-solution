#!/usr/bin/env python3
"""Search compact task034 direction recodings allowed by the generator."""

from __future__ import annotations

import itertools


def main() -> None:
    for coefficient in range(1, 32):
        weights = [1, 1 + coefficient, 2 + 2 * coefficient, 2]
        for shift in range(8):
            if sum(weights) << shift > 255:
                continue
            codes = {
                subset: sum(weights[index] for index in subset) << shift
                for count in (1, 2, 3)
                for subset in itertools.combinations(range(4), count)
            }
            masks: list[int] = []
            valid = True
            for direction in range(4):
                forbidden_bits = 0
                for subset, code in codes.items():
                    if direction not in subset:
                        forbidden_bits |= code
                mask = (~forbidden_bits) & 255
                if not all(
                    code & mask
                    for subset, code in codes.items()
                    if direction in subset
                ):
                    valid = False
                masks.append(mask)
            if valid:
                print(
                    f"coefficient={coefficient} shift={shift} "
                    f"weights={weights} masks={masks} max={max(codes.values())}"
                )

    print("-- generator-existing dynamic coefficient families --")
    families = {
        "k": lambda k: k,
        "4-k": lambda k: 4 - k,
        "k+1": lambda k: k + 1,
        "2k": lambda k: 2 * k,
        "2k-1": lambda k: 2 * k - 1,
        "2(4-k)": lambda k: 2 * (4 - k),
    }
    subsets = [
        subset
        for count in (1, 2, 3)
        for subset in itertools.combinations(range(4), count)
    ]
    for name, coefficient_for_k in families.items():
        for shift in range(8):
            codes: dict[tuple[int, ...], int] = {}
            for subset in subsets:
                d = [int(index in subset) for index in range(4)]
                k = len(subset)
                a = d[2] + d[3]
                bc = d[1] + 2 * d[2]
                codes[subset] = (k + a + coefficient_for_k(k) * bc) << shift
            if max(codes.values()) > 255:
                continue
            masks = []
            valid = True
            for direction in range(4):
                forbidden_bits = 0
                for subset, code in codes.items():
                    if direction not in subset:
                        forbidden_bits |= code
                mask = (~forbidden_bits) & 255
                if not all(
                    code & mask
                    for subset, code in codes.items()
                    if direction in subset
                ):
                    valid = False
                masks.append(mask)
            if valid:
                print(
                    f"family={name} shift={shift} masks={masks} "
                    f"max={max(codes.values())} codes={codes}"
                )

    print("-- arbitrary coefficient by direction-count (first 30) --")
    found = 0
    for coefficients in itertools.product(range(16), repeat=3):
        for shift in (0, 1, 4, 6):
            codes = {}
            for subset in subsets:
                d = [int(index in subset) for index in range(4)]
                k = len(subset)
                a = d[2] + d[3]
                bc = d[1] + 2 * d[2]
                codes[subset] = (k + a + coefficients[k - 1] * bc) << shift
            if max(codes.values()) > 255:
                continue
            masks = []
            valid = True
            for direction in range(4):
                forbidden_bits = 0
                for subset, code in codes.items():
                    if direction not in subset:
                        forbidden_bits |= code
                mask = (~forbidden_bits) & 255
                if not all(
                    code & mask
                    for subset, code in codes.items()
                    if direction in subset
                ):
                    valid = False
                masks.append(mask)
            if valid:
                print(
                    f"coefficients={coefficients} shift={shift} "
                    f"masks={masks} max={max(codes.values())}"
                )
                found += 1
                if found == 30:
                    break
        if found == 30:
            break

    print("-- constant-free formulas over x=k+A, y=B+d2 --")
    formulas = {
        "x+y": lambda x, y: x + y,
        "x*y": lambda x, y: x * y,
        "x<<y": lambda x, y: x << y,
        "y<<x": lambda x, y: y << x,
        "x+(y<<x)": lambda x, y: x + (y << x),
        "x+(y<<y)": lambda x, y: x + (y << y),
        "(x<<y)+y": lambda x, y: (x << y) + y,
        "(y<<x)+y": lambda x, y: (y << x) + y,
        "(x+y)<<x": lambda x, y: (x + y) << x,
        "(x+y)<<y": lambda x, y: (x + y) << y,
        "(x*y)<<x": lambda x, y: (x * y) << x,
        "(x*y)<<y": lambda x, y: (x * y) << y,
        "x|(y<<x)": lambda x, y: x | (y << x),
        "x|(y<<y)": lambda x, y: x | (y << y),
        "(x<<y)|y": lambda x, y: (x << y) | y,
        "(y<<x)|y": lambda x, y: (y << x) | y,
    }
    for name, formula in formulas.items():
        codes = {}
        for subset in subsets:
            d = [int(index in subset) for index in range(4)]
            k = len(subset)
            x = k + d[2] + d[3]
            y = d[1] + 2 * d[2]
            codes[subset] = formula(x, y)
        if max(codes.values()) > 255:
            continue
        masks = []
        valid = True
        for direction in range(4):
            forbidden_bits = 0
            for subset, code in codes.items():
                if direction not in subset:
                    forbidden_bits |= code
            mask = (~forbidden_bits) & 255
            if not all(
                (code & mask) > 6
                for subset, code in codes.items()
                if direction in subset
            ):
                valid = False
            masks.append(mask)
        if not valid:
            # Also report formulas that become usable with zero point one.
            valid = True
            masks = []
            for direction in range(4):
                forbidden_bits = 0
                for subset, code in codes.items():
                    if direction not in subset:
                        forbidden_bits |= code
                mask = (~forbidden_bits) & 255
                if not all(
                    (code & mask) > 1
                    for subset, code in codes.items()
                    if direction in subset
                ):
                    valid = False
                masks.append(mask)
            if valid and min(codes.values()) > 1:
                print(f"formula-zp1={name} masks={masks} codes={codes}")
        elif min(codes.values()) > 6:
            print(f"formula={name} masks={masks} codes={codes}")


if __name__ == "__main__":
    main()
