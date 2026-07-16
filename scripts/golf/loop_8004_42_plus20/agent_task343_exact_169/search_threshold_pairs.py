#!/usr/bin/env python3
"""Search exact two-Conv threshold formulas for task343.

The previous exhaustive search compared affine dynamic-Conv probes to only
four reusable Conv constants.  A scalar initializer costs one parameter, so
two arbitrary thresholds still fit below the cost-173 authority.  This scan
enumerates every affine scalar probe and every non-dominated threshold/equality
predicate, then solves exact AND/OR pairs with bitsets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOUND77 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_task343_sound77"
sys.path.insert(0, str(SOUND77))
from search_full_thresholds import build_features, examples  # noqa: E402


@dataclass(frozen=True)
class Predicate:
    feature_index: int
    spec: tuple[int, int, int, int]
    op: str
    constant: int
    not_count: int = 0


def bits(values: np.ndarray) -> int:
    return int.from_bytes(np.packbits(values).tobytes())


def predicate(values: np.ndarray, op: str, constant: int) -> np.ndarray:
    if op == "gt":
        return values > constant
    if op == "lt":
        return values < constant
    if op == "eq":
        return values == constant
    if op == "ne":
        return values != constant
    raise ValueError(op)


def complexity(item: Predicate) -> tuple[int, int, int, str]:
    return item.not_count, abs(item.constant), item.feature_index, item.op


def retain(
    table: dict[int, Predicate], mask: int, item: Predicate,
) -> None:
    old = table.get(mask)
    if old is None or complexity(item) < complexity(old):
        table[mask] = item


def disjoint_pair(table: dict[int, Predicate], bit_count: int):
    rows = sorted(table.items(), key=lambda row: (row[0].bit_count(), complexity(row[1])))
    if not rows:
        return None
    all_candidates = (1 << len(rows)) - 1
    zero_at = [0] * bit_count
    for index, (mask, _) in enumerate(rows):
        flag = 1 << index
        for bit in range(bit_count):
            if not ((mask >> bit) & 1):
                zero_at[bit] |= flag
    for index, (mask, item) in enumerate(rows):
        allowed = all_candidates & ~(1 << index)
        pending = mask
        while pending and allowed:
            low = pending & -pending
            bit = low.bit_length() - 1
            allowed &= zero_at[bit]
            pending ^= low
        if allowed:
            other_index = (allowed & -allowed).bit_length() - 1
            return item, rows[other_index][1], mask, rows[other_index][0]
    return None


def main() -> None:
    sample_size = 3000
    seed = 343_169_001
    x, req = examples(sample_size, seed)
    only6 = req == 1
    only8 = req == -1
    fm, specs = build_features(x)
    universal: dict[int, Predicate] = {}
    zero8: dict[int, Predicate] = {}

    for index, spec in enumerate(specs):
        f6 = fm[only6, index]
        f8 = fm[only8, index]
        # Non-dominated monotone predicates universal on the period-6-only side.
        universal_candidates = [
            ("gt", int(f6.min()) - 1),
            ("lt", int(f6.max()) + 1),
        ]
        if np.all(f6 == f6[0]):
            universal_candidates.append(("eq", int(f6[0])))
        set6 = set(map(int, np.unique(f6)))
        universal_candidates.extend(
            ("ne", int(value)) for value in np.unique(f8) if int(value) not in set6
        )
        for op, constant in universal_candidates:
            p8 = predicate(f8, op, constant)
            item = Predicate(index, specs[index], op, constant, int(op == "ne"))
            retain(universal, bits(p8), item)

        # Non-dominated predicates false on every period-8-only example.
        zero_candidates = [
            ("gt", int(f8.max())),
            ("lt", int(f8.min())),
        ]
        set8 = set(map(int, np.unique(f8)))
        zero_candidates.extend(
            ("eq", int(value)) for value in np.unique(f6) if int(value) not in set8
        )
        if np.all(f8 == f8[0]):
            zero_candidates.append(("ne", int(f8[0])))
        for op, constant in zero_candidates:
            p6 = predicate(f6, op, constant)
            # OR succeeds unless both predicates miss a period-6-only case.
            miss6 = ~p6
            item = Predicate(index, specs[index], op, constant, int(op == "ne"))
            retain(zero8, bits(miss6), item)

    and_pair = disjoint_pair(universal, int(only8.sum()))
    or_pair = disjoint_pair(zero8, int(only6.sum()))
    found = None
    if and_pair is not None:
        found = {
            "combine": "And",
            "predicates": [asdict(and_pair[0]), asdict(and_pair[1])],
            "error_masks": [and_pair[2], and_pair[3]],
        }
    elif or_pair is not None:
        found = {
            "combine": "Or",
            "predicates": [asdict(or_pair[0]), asdict(or_pair[1])],
            "error_masks": [or_pair[2], or_pair[3]],
        }

    result = {
        "sample_size": sample_size,
        "seed": seed,
        "only6": int(only6.sum()),
        "only8": int(only8.sum()),
        "both": int((req == 0).sum()),
        "features": int(fm.shape[1]),
        "universal_unique_masks": len(universal),
        "zero8_unique_masks": len(zero8),
        "found": found,
    }
    (HERE / "threshold_search.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
