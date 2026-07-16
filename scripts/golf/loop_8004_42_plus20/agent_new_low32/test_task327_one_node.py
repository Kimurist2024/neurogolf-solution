#!/usr/bin/env python3
"""Prove the archived cost-46 one-node ConvTranspose proposal infeasible."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.golf.scratch_agents.task327.build_candidate import features, placements, targets


HERE = Path(__file__).resolve().parent


def main() -> None:
    constraints: dict[tuple[int, bytes], dict[str, object]] = {}
    conflict = None
    processed = 0
    for placement_index, points in enumerate(placements()):
        background_input = np.zeros((30, 30), dtype=np.uint8)
        background_input[:3, :3] = 1
        for row, col in points:
            background_input[row, col] = 0
        background_target, rays = targets(points)
        cases = [(1, features(background_input), background_target, "background")]
        for point, ray in zip(points, rays):
            foreground = np.zeros((30, 30), dtype=np.uint8)
            foreground[point] = 1
            cases.append((0, features(foreground), ray, f"foreground_{point}"))
        for bias_kind, matrix, target, case_name in cases:
            labels = np.zeros((30, 30), dtype=np.uint8)
            labels[: target.shape[0], : target.shape[1]] = target
            for output_index, (row, label) in enumerate(zip(matrix, labels.reshape(-1))):
                key = (bias_kind, np.packbits(row.astype(np.uint8)).tobytes())
                new = {
                    "label": int(label),
                    "placement_index": placement_index,
                    "points": [list(point) for point in points],
                    "case": case_name,
                    "output_index": output_index,
                    "output_rc": [output_index // 30, output_index % 30],
                }
                old = constraints.get(key)
                if old is not None and old["label"] != new["label"]:
                    conflict = {"prior": old, "new": new, "bias_kind": bias_kind}
                    break
                constraints[key] = new
            if conflict is not None:
                break
        processed += 1
        if conflict is not None:
            break
    result = {
        "task": 327,
        "proposal": "one-node dynamic-weight ConvTranspose, 6x6 kernel plus ten biases",
        "claimed_cost": 46,
        "all_generator_placements": len(placements()),
        "placements_processed_before_contradiction": processed,
        "unique_local_constraints_before_contradiction": len(constraints),
        "feasible": False,
        "conflict": conflict,
        "proof": "The same local binary patch and the same channel-bias kind require opposite output labels. Therefore no affine ConvTranspose response, regardless of learned weights, can implement the full generator domain.",
        "candidate_written": False,
    }
    (HERE / "evidence" / "task327_one_node_infeasible.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
