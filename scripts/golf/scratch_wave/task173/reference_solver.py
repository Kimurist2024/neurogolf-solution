#!/usr/bin/env python3
"""Reference numpy solver for task173 (ARC 72322fa7), archetype overlaps.

Rule (generator task_72322fa7.py):
- 1..3 sprite types. Each type = 3x3 stamp, 2 distinct colors:
    color0 = arm cells, color1 = center cell (offset (1,1)).
  Shapes (offsets present): 0=X {(0,0),(0,2),(1,1),(2,0),(2,2)},
    1=plus {(0,1),(1,0),(1,1),(1,2),(2,1)}, 2=horiz {(1,0),(1,1),(1,2)},
    3=vert {(0,1),(1,1),(2,1)}.
- All 2..6 colors across types are distinct. So a color identifies a unique
  (type, role).
- OUTPUT: every stamp drawn fully. INPUT: some cells blacked:
    ms==0 full, ms==1 only center (arms blacked), ms==2 only arms (center black).
  First copy of each type is ms==0 (full), so each type's full template is
  observable in the input.

Reconstruction from input alone:
  1. Find all fully-shown 3x3 stamps -> learn, per type, the template
     (shape id, arm color, center color).
  2. For each location of a sprite (anchored top-left), stamp the full template.
     A location is detected from ANY visible cell of that type. We anchor via
     the center pixel when shown, else via arm geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / "inputs/neurogolf-2026/task173.json"

SHAPES = {
    0: [(0, 0), (0, 2), (2, 0), (2, 2)],  # X arms
    1: [(0, 1), (1, 0), (1, 2), (2, 1)],  # plus arms
    2: [(1, 0), (1, 2)],  # horiz arms
    3: [(0, 1), (2, 1)],  # vert arms
}


def learn_types(inp: np.ndarray):
    """Return dict color -> (role, sid, arm_color, center_color).

    Discovered from fully-shown (ms==0) stamps in the input.
    """
    H, W = inp.shape
    info = {}  # color -> (role 'a'|'c', sid, arm, center)
    for r in range(H - 2):
        for c in range(W - 2):
            block = inp[r:r + 3, c:c + 3]
            for sid, arms in SHAPES.items():
                want_offsets = set(arms) | {(1, 1)}
                # all wanted nonzero, all others zero
                ok = True
                for i in range(3):
                    for j in range(3):
                        v = block[i, j]
                        if (i, j) in want_offsets:
                            if v == 0:
                                ok = False
                        else:
                            if v != 0:
                                ok = False
                if not ok:
                    continue
                arm_vals = {block[o] for o in arms}
                cen = block[1, 1]
                if len(arm_vals) != 1:
                    continue
                arm = next(iter(arm_vals))
                if arm == cen:
                    continue
                info[arm] = ("a", sid, int(arm), int(cen))
                info[cen] = ("c", sid, int(arm), int(cen))
    return info


def solve(inp: np.ndarray) -> np.ndarray:
    H, W = inp.shape
    info = learn_types(inp)
    out = np.zeros_like(inp)

    # Collect anchors. Each anchor carries its own (sid, arm, cen) so two types
    # that share a shape are never confused. Center pixels first (unambiguous),
    # then arm-only stamps (ms==2).
    placed_anchors = set()  # (sid, arm, cen, ar, ac)
    occupied = set()  # (sid, ar, ac) to skip re-detection

    # Pass A: center pixels
    for r in range(H):
        for c in range(W):
            v = inp[r, c]
            if v == 0 or v not in info:
                continue
            role, sid, arm, cen = info[v]
            if role == "c":
                ar, ac = r - 1, c - 1
                placed_anchors.add((sid, arm, cen, ar, ac))
                occupied.add((sid, ar, ac))

    # Pass B: arm-only stamps. For each arm color, scan windows; if all arm
    # offsets carry the arm color, it's an ms==2 stamp.
    for col in [c for c in info if info[c][0] == "a"]:
        role, sid, arm, cen = info[col]
        arms = SHAPES[sid]
        for r in range(H - 2):
            for c in range(W - 2):
                if (sid, r, c) in occupied:
                    continue
                block = inp[r:r + 3, c:c + 3]
                if all(block[o] == arm for o in arms):
                    placed_anchors.add((sid, arm, cen, r, c))
                    occupied.add((sid, r, c))

    # Stamp all anchors
    for sid, arm, cen, ar, ac in placed_anchors:
        role_arm = SHAPES[sid]
        for o in role_arm:
            rr, cc = ar + o[0], ac + o[1]
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = arm
        cr, cc = ar + 1, ac + 1
        if 0 <= cr < H and 0 <= cc < W:
            out[cr, cc] = cen
    return out


def main() -> None:
    data = json.loads(TASK.read_text())
    total = bad = 0
    for subset in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data.get(subset, [])):
            inp = np.array(ex["input"], dtype=np.int64)
            gold = np.array(ex["output"], dtype=np.int64)
            pred = solve(inp)
            total += 1
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH {subset}[{i}] dims={inp.shape}")
                    diff = np.argwhere(pred != gold)
                    print("  ndiff", len(diff), "first", diff[:5].tolist())
    print(f"total={total} bad={bad}")
    assert bad == 0, "reference solver does not match all pairs"
    print("OK: reference solver exact on all pairs")


if __name__ == "__main__":
    main()
