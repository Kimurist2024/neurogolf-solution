#!/usr/bin/env python3
"""Synthesize a <=3-op task034 direction code from existing scalar inputs."""

from __future__ import annotations

import itertools


SUBSETS = [
    subset
    for count in (1, 2, 3)
    for subset in itertools.combinations(range(4), count)
]


def is_separable(values: tuple[int, ...], zero_point: int = 6) -> tuple[int, ...] | None:
    if min(values) <= zero_point:
        return None
    masks: list[int] = []
    for direction in range(4):
        forbidden = 0
        for subset, value in zip(SUBSETS, values, strict=True):
            if direction not in subset:
                forbidden |= value
        mask = (~forbidden) & 255
        if not all(
            (value & mask) > zero_point
            for subset, value in zip(SUBSETS, values, strict=True)
            if direction in subset
        ):
            return None
        masks.append(mask)
    return tuple(masks)


def operations(name_a: str, a: tuple[int, ...], name_b: str, b: tuple[int, ...]):
    yield f"({name_a}+{name_b})", tuple((x + y) & 255 for x, y in zip(a, b, strict=True))
    yield f"({name_a}*{name_b})", tuple((x * y) & 255 for x, y in zip(a, b, strict=True))
    yield f"({name_a}|{name_b})", tuple(x | y for x, y in zip(a, b, strict=True))
    yield f"({name_a}^{name_b})", tuple(x ^ y for x, y in zip(a, b, strict=True))
    yield f"({name_a}-{name_b})", tuple((x - y) & 255 for x, y in zip(a, b, strict=True))
    if max(b) <= 7:
        yield f"({name_a}<<{name_b})", tuple((x << y) & 255 for x, y in zip(a, b, strict=True))


def main() -> None:
    x_values = []
    y_values = []
    for subset in SUBSETS:
        d = [int(index in subset) for index in range(4)]
        x_values.append(len(subset) + d[2] + d[3])
        y_values.append(d[1] + 2 * d[2])
    base = {
        "x": tuple(x_values),
        "y": tuple(y_values),
        "one": (1,) * len(SUBSETS),
        "four": (4,) * len(SUBSETS),
        "six": (6,) * len(SUBSETS),
    }
    levels: list[dict[tuple[int, ...], str]] = [
        {values: name for name, values in base.items()}
    ]
    seen = dict(levels[0])
    for cost in range(1, 4):
        level: dict[tuple[int, ...], str] = {}
        for left_cost in range(cost):
            right_cost = cost - 1 - left_cost
            for a, name_a in levels[left_cost].items():
                for b, name_b in levels[right_cost].items():
                    for name, values in operations(name_a, a, name_b, b):
                        if values in seen or values in level:
                            continue
                        level[values] = name
                        masks = is_separable(values)
                        if masks is not None:
                            print(
                                f"cost={cost} expression={name} masks={masks} "
                                f"values={values}"
                            )
                            return
        levels.append(level)
        seen.update(level)
        print(f"cost={cost} unique={len(level)} total={len(seen)}")
    print("no <=3-op expression")


if __name__ == "__main__":
    main()
