#!/usr/bin/env python3
"""Rank baseline tasks 150..400 and select only sound Type A/B rebuilds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COSTS = ROOT / "scripts/golf/loop_8003_40/agent_exact_scanners/base_costs.json"
MODELS = ROOT / "scripts/golf/loop_8003_40/base_models"
BASELINE = ROOT / "submission_base_8003.40.zip"

PRIVATE_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    191, 192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246,
    255, 264, 277, 285, 286, 302, 319, 325, 333, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
CONTAMINATED = {182, 204}
OTHER_LANES = {73, 111, 122, 168, 192, 260, 271, 285, 289, 343, 344, 359}

# Manual spec classification after reading the generators and sound-rebuild
# guidance.  These are deliberately explicit rather than guessed from ops.
MANUAL_REJECT = {
    349: ("TYPE_C", "variable component radius/size must be recovered per object"),
    367: ("SHAPE_CLOAK", "24 CenterCropPad nodes in baseline lineage"),
    340: ("LOOKUP_OR_PACKED", "ScatterElements and repeated packed Einsum path"),
    370: ("SHAPE_CLOAK", "24 CenterCropPad nodes"),
    330: ("SHAPE_CLOAK", "31 CenterCropPad nodes"),
    280: ("LOOKUP_OR_CLOAK", "ScatterElements plus 20 CenterCropPad nodes"),
    382: ("SHAPE_CLOAK", "CenterCropPad with 13-input Einsum path"),
    201: ("TYPE_D_LOOKUP", "TfIdfVectorizer/ScatterElements global geometry"),
    251: ("SHAPE_CLOAK", "55 CenterCropPad nodes"),
    364: ("SHAPE_CLOAK", "196 CenterCropPad nodes"),
    270: ("LOOKUP_OR_CLOAK", "ScatterElements and CenterCropPad"),
    165: ("LOOKUP_OR_CLOAK", "ArgMax/Gather with 80 CenterCropPad nodes"),
    310: ("LOOKUP", "TfIdfVectorizer with 23-input Einsum"),
    238: ("TYPE_C_LOOKUP", "ScatterND/Resize for data-dependent geometry"),
    328: ("GIANT_EINSUM", "58-input Einsum"),
    368: ("TYPE_C", "copy the unique colored sprite into every gray component"),
}

SELECTED = {
    156: {
        "type": "A",
        "rule": "classify two solid rectangles by their unequal extent; recolor interiors 1/2",
    },
    237: {
        "type": "B",
        "rule": "extend each marker rightward and carry the latest marker down the last column",
    },
    345: {
        "type": "B",
        "rule": "unroll at most nine upward steps, turning right when gray blocks the next cell",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(model: onnx.ModelProto) -> dict[str, object]:
    ops = [node.op_type for node in model.graph.node]
    return {
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max(
            [len(node.input) for node in model.graph.node if node.op_type == "Einsum"] or [0]
        ),
        "lookup_or_cloak_ops": sorted(
            set(ops)
            & {
                "TfIdfVectorizer", "Hardmax", "ScatterElements", "ScatterND",
                "GatherND", "CenterCropPad", "AffineGrid", "Resize",
            }
        ),
    }


def main() -> None:
    source = json.loads(COSTS.read_text())
    ranked = [row for row in source["ranked"] if 150 <= row["task"] <= 400]
    decisions = []
    for position, row in enumerate(ranked, 1):
        task = row["task"]
        model_path = MODELS / f"task{task:03d}.onnx"
        model = onnx.load(model_path, load_external_data=False)
        if task in OTHER_LANES:
            decision, reason = "EXCLUDE_OTHER_LANE", "owned by a separate optimization lane"
        elif task in PRIVATE_OR_UNSOUND:
            decision, reason = "EXCLUDE_RISK", "private-zero/unsound/monitored catalog"
        elif task in CONTAMINATED:
            decision, reason = "EXCLUDE_CONTAMINATION", "known downstream contamination task"
        elif task in MANUAL_REJECT:
            kind, reason = MANUAL_REJECT[task]
            decision = f"EXCLUDE_{kind}"
        elif task in SELECTED:
            decision, reason = "SELECT", SELECTED[task]["rule"]
        else:
            decision, reason = "LOWER_PRIORITY", "below the first three sound Type A/B selections"
        decisions.append(
            {
                "rank_150_400": position,
                **row,
                "model": str(model_path.relative_to(ROOT)),
                "model_sha256": sha(model_path),
                "graph": flags(model),
                "decision": decision,
                "reason": reason,
                "classification": SELECTED.get(task, {}).get("type"),
            }
        )

    selected = [next(row for row in decisions if row["task"] == task) for task in SELECTED]
    report = {
        "baseline": "submission_base_8003.40.zip",
        "baseline_sha256": sha(BASELINE),
        "cost_source": str(COSTS.relative_to(ROOT)),
        "range": [150, 400],
        "ranked_count": len(ranked),
        "selected": selected,
        "decisions": decisions,
    }
    output = HERE / "ranking_selection.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
