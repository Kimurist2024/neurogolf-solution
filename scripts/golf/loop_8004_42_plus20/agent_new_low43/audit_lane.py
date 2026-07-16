#!/usr/bin/env python3
"""Run the fail-closed low43 rule, baseline, history, and lower-lead audit."""

from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parent / "agent_new_low39" / "audit_lane.py"
SPEC = importlib.util.spec_from_file_location("low39_audit_lane", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE.HERE = HERE
MODULE.TARGETS = (6, 334, 244, 249, 347, 386, 146, 291)
MODULE.RULES = {
    6: "For each fixed 3x7 row, output color 2 where the corresponding left/right binary cells are both one.",
    334: "Map the single nonzero input color 1, 2, or 3 to its fixed color-5 3x3 glyph.",
    244: "Decode the separator period, sample one representative from each constant-color block, and emit the block-color matrix.",
    249: "Duplicate every input row horizontally.",
    347: "For each fixed 3x6 row, output color 6 where either corresponding half-cell is nonzero.",
    386: "For each fixed 4x7 row, output color 3 where both corresponding half-cells are zero.",
    146: "Split the fixed 9x3 input into three 3x3 blocks and return the first block that is not symmetric under transpose.",
    291: "Return the color of the hollow object, recognized by two same-color cells separated by one or more zeros in a row-major encoding.",
}
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400"
MODULE.LOWER_LEADS = {
    6: [
        ARCHIVE / "task006_r01_static30.onnx",
        ARCHIVE / "task006_r02_static38.onnx",
        ARCHIVE / "task006_r03_static40.onnx",
        ARCHIVE / "task006_r04_static40.onnx",
    ],
    146: [
        ARCHIVE / "task146_r01_static38.onnx",
        ARCHIVE / "task146_r02_static38.onnx",
    ],
    291: [
        ROOT / "scripts/golf/scratch_codex/task291/no_channel_constants_edge30.onnx",
    ],
}


if __name__ == "__main__":
    MODULE.main()
