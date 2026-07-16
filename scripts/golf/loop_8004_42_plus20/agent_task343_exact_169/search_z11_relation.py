#!/usr/bin/env python3
"""Exhaust sampled Cast(z11) AND affine-relation classifiers.

The exact rule always chooses period 8 when visible width is at least 12.
The z11 dynamic-Conv probe is nonzero exactly below that boundary.  Thus a
cost-172 graph exists if two affine probes A/B satisfy one Equal/Greater/Less
relation on every hard case with width <= 11.  This script deduplicates all
56,074 probes to their complete hard-case value vectors and exhausts that
relation family, rather than checking only ranked anchors.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOUND77 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_task343_sound77"
sys.path.insert(0, str(SOUND77))
from search_full_thresholds import build_features, examples  # noqa: E402


def pairing(spec: tuple[int, int, int, int]):
    dh, top, dw, left = spec
    return (
        tuple((r, dh * r - top) for r in range(5) if 0 <= dh * r - top < 5),
        tuple((c, dw * c - left) for c in range(15) if 0 <= dw * c - left < 15),
    )


def exhaustive_greater(
    a: np.ndarray,
    b: np.ndarray,
    specs: list[tuple[int, int, int, int]],
):
    """Find i,j with A_i>A_j for every six case and B_i<=B_j for every eight."""
    min_a, max_a, sum_a = a.min(0), a.max(0), a.sum(0)
    min_b, max_b, sum_b = b.min(0), b.max(0), b.sum(0)
    for i in range(a.shape[1]):
        candidates = np.flatnonzero(
            (min_a[i] > min_a)
            & (max_a[i] > max_a)
            & (sum_a[i] > sum_a)
            & (min_b[i] <= min_b)
            & (max_b[i] <= max_b)
            & (sum_b[i] <= sum_b)
        )
        if not candidates.size:
            continue
        # Apply the tightest constraints first.  This is exact: rows are only
        # reordered, never sampled or dropped.
        order_a = np.argsort(a[:, i], kind="stable")
        order_b = np.argsort(-b[:, i], kind="stable")
        for row in order_a:
            candidates = candidates[a[row, candidates] < a[row, i]]
            if not candidates.size:
                break
        if not candidates.size:
            continue
        for row in order_b:
            candidates = candidates[b[row, candidates] >= b[row, i]]
            if not candidates.size:
                break
        if candidates.size:
            j = int(candidates[0])
            return {
                "op": "Greater",
                "left_index": i,
                "right_index": j,
                "left_spec": specs[i],
                "right_spec": specs[j],
            }
    return None


def exhaustive_equal(
    a: np.ndarray,
    b: np.ndarray,
    specs: list[tuple[int, int, int, int]],
):
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index in range(a.shape[1]):
        groups[a[:, index].tobytes()].append(index)
    checked = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        values = b[:, group]
        for pos, i in enumerate(group):
            different = np.all(values[:, pos, None] != values[:, pos + 1 :], axis=0)
            checked += len(group) - pos - 1
            if np.any(different):
                offset = int(np.flatnonzero(different)[0])
                j = group[pos + 1 + offset]
                return {
                    "op": "Equal",
                    "left_index": i,
                    "right_index": j,
                    "left_spec": specs[i],
                    "right_spec": specs[j],
                    "pairs_checked_before_hit": checked,
                }
    return None


def main() -> None:
    sample_size = 3000
    seed = 343_169_001
    x, req = examples(sample_size, seed)
    fm, all_specs = build_features(x)
    canonical = {pairing(spec): index for index, spec in enumerate(all_specs)}
    z11_index = canonical[pairing((30, -4, 30, -11))]
    z11 = fm[:, z11_index] != 0
    six = req == 1
    ambiguous_eight = (req == -1) & z11
    hard = six | ambiguous_eight

    unique: dict[bytes, int] = {}
    for index in range(fm.shape[1]):
        unique.setdefault(fm[hard, index].tobytes(), index)
    representatives = np.asarray(list(unique.values()), dtype=np.int32)
    reduced_specs = [all_specs[int(index)] for index in representatives]
    a = fm[six][:, representatives].astype(np.int16)
    b = fm[ambiguous_eight][:, representatives].astype(np.int16)

    found = exhaustive_equal(a, b, reduced_specs)
    if found is None:
        found = exhaustive_greater(a, b, reduced_specs)
    result = {
        "sample_size": sample_size,
        "seed": seed,
        "all_features": int(fm.shape[1]),
        "unique_hard_value_vectors": int(representatives.size),
        "only6": int(six.sum()),
        "ambiguous_only8_visible_le_11": int(ambiguous_eight.sum()),
        "other_only8_filtered_by_z11": int(((req == -1) & ~z11).sum()),
        "z11_original_spec": [30, -4, 30, -11],
        "z11_canonical_spec": all_specs[z11_index],
        "found": found,
    }
    (HERE / "z11_relation_search.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
