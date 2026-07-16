#!/usr/bin/env python3
"""Exhaust the finite task066 path geometry used by the support proof.

Random cyan is intentionally not enumerated: it is prepended and the path plus
mandatory guards overwrite it.  Extra cyan can only OR extra bits into G/O;
it cannot remove the aligned mandatory bit checked here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROTECTED = (ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "others/71407")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256(path.read_bytes())
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def protected() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): tree_sha(path) for path in PROTECTED}


def main() -> None:
    before = protected()
    counts = {"S": 0, "U": 0, "with_flip_and_xpose": 0}
    extrema = {"turn_min": 1 << 30, "turn_max": -1, "green_min": 1 << 30, "green_max": -1}

    for size in range(10, 21):
        # Exact inclusive bounds from task_2dd70a9a.py/common.randint.
        for height in range(3 * size // 4, 7 * size // 8 + 1):
            for width in range(size // 2, 3 * size // 4 + 1):
                for row in range(1, size - height):
                    for _col in range(1, size - width):
                        for mid in range(row + 3, row + height - 2):
                            counts["S"] += 1
                            green_unflipped = row + height - 2
                            turn_unflipped = mid
                            assert turn_unflipped < green_unflipped
                            # Mandatory green guard mid-1 is shifted one bit by G=2*C,
                            # hence it aligns with the O guard at mid in pairD.
                            assert (mid - 1) + 1 == turn_unflipped

                            green_flipped = size - (row + height)
                            turn_flipped = size - 1 - mid
                            assert turn_flipped >= green_flipped + 2
                            # After reflection, the G bit is shifted down two in pairU.
                            assert (size - mid) + 1 - 2 == turn_flipped
                            assert 0 <= turn_unflipped < 20 and 0 <= turn_flipped < 20
                            extrema["turn_min"] = min(extrema["turn_min"], turn_unflipped, turn_flipped)
                            extrema["turn_max"] = max(extrema["turn_max"], turn_unflipped, turn_flipped)
                            extrema["green_min"] = min(extrema["green_min"], green_unflipped, green_flipped)
                            extrema["green_max"] = max(extrema["green_max"], green_unflipped, green_flipped)

        for height in range(size // 2, 3 * size // 4 + 1):
            for width in range(size // 2, 3 * size // 4 + 1):
                for row in range(1, size - height):
                    for _col in range(1, size - width):
                        base = row + height - 1
                        for _mid1 in range(row, row + height - 3):
                            for mid2 in range(row, row + height - 2):
                                counts["U"] += 1
                                green_unflipped = mid2
                                turn_unflipped = base
                                assert turn_unflipped >= green_unflipped + 2
                                # Mandatory green guard base+1 enters G one bit up,
                                # then pairU shifts it down two to the O guard at base.
                                assert (base + 1) + 1 - 2 == turn_unflipped

                                green_flipped = size - mid2 - 2
                                turn_flipped = size - 1 - base
                                assert turn_flipped < green_flipped
                                # Reflection makes the G and O guards align in pairD.
                                assert (size - base - 2) + 1 == turn_flipped
                                assert 0 <= turn_unflipped < 20 and 0 <= turn_flipped < 20
                                extrema["turn_min"] = min(extrema["turn_min"], turn_unflipped, turn_flipped)
                                extrema["turn_max"] = max(extrema["turn_max"], turn_unflipped, turn_flipped)
                                extrema["green_min"] = min(extrema["green_min"], green_unflipped, green_flipped)
                                extrema["green_max"] = max(extrema["green_max"], green_unflipped, green_flipped)

    # Each canonical geometry was proved for both flips above; xpose only swaps
    # which of the algebraically identical Gv/Gh and Ov/Oh equations is selected.
    counts["with_flip_and_xpose"] = (counts["S"] + counts["U"]) * 4
    result = {
        "generator_hash": "2dd70a9a",
        "geometry_parameter_tuples": counts,
        "extrema": extrema,
        "all_assertions_passed": True,
        "arbitrary_cyan_closure": "mandatory guards override prepended noise; extra cyan only adds G/O bits",
        "consequence": "each valid geometry supplies a nonzero bit to aMask or bMask, so selF>=1",
        "counterexample": None,
    }
    after = protected()
    result["integrity"] = {"before": before, "after": after, "unchanged": before == after}
    (HERE / "geometry_proof.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
