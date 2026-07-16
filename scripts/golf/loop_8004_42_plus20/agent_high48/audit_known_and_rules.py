#!/usr/bin/env python3
"""Audit known pairs and decoded Sakana rules for the high48 lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_new_low45" / "audit_known_and_rules.py"
SPEC = importlib.util.spec_from_file_location("high48_shared_known", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)

shared.HERE = HERE
shared.ROOT = HERE.parents[3]
shared.TARGETS = (8, 275, 134, 112, 168, 109, 160, 170)
shared.RULES = {
    8: (
        "Find the row containing the 2x2 color-8 anchor. Rotate the grid four "
        "times, each time stable-sorting the rows above the anchor by emptiness; "
        "equivalently translate the color-2 object until its bounding box touches "
        "the anchor's row and column while preserving its pattern."
    ),
    275: (
        "Split the rectangular input into two equal square patterns A and B along "
        "its long axis, then output the Kronecker-style substitution: every active "
        "cell of A stamps a copy of B colored by that A cell."
    ),
    134: (
        "Identify the structured nonzero color D and the sparse noise color A; "
        "partition the D-component's repeated blocks and emit the 3x3 occupancy "
        "mask recolored with A."
    ),
    112: (
        "Locate the 2x2 separator/pivot block of color 3, replace the asymmetric "
        "motif color with the separator color, and mirror the motif across both "
        "pivot axes to make four reflected copies."
    ),
    168: (
        "For each L-shaped three-cell corner in the nonzero color, extend a "
        "diagonal ray away from the missing corner until the fixed 10x10 boundary."
    ),
    109: (
        "Remove the central cross divider, recolor the source quadrant by the "
        "divider color, and reflect it horizontally and vertically into the four "
        "quadrants, yielding an even square one cell smaller than the input."
    ),
    160: (
        "In the fixed 10x10 grid, recognize every exact 3x3 plus-shaped five-cell "
        "object of color 1 and recolor only those plus objects to color 2."
    ),
    170: (
        "Decode the large monochrome block mosaic as a binary mask and apply that "
        "mask to the nearby small color matrix: keep entries at occupied mosaic "
        "cells and output zero at empty cells."
    ),
}


if __name__ == "__main__":
    shared.main()
