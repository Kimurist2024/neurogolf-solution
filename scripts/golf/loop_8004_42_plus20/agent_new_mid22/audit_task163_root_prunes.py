#!/usr/bin/env python3
"""Independent gate for root_sweep29 task163 latent-prune candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
from golf.rank_dir import cost_of  # noqa: E402
from audit_lane import known_dual, structure  # noqa: E402


PATHS = [
    ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent/task163_r001.onnx",
    ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent/task163_r002.onnx",
]


def main() -> None:
    rows = []
    for path in PATHS:
        data = path.read_bytes()
        model = onnx.load_from_string(data)
        memory, params, cost = cost_of(str(path))
        known = known_dual(model, 163)
        known_perfect = all(mode.get("perfect") for mode in known.values())
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "cost": {"memory": memory, "params": params, "cost": cost},
            "gain_if_correct": 0.06317890162153361,
            "known_dual": known,
            "known_perfect": known_perfect,
            "fresh_gate_run": False,
            "fresh_gate_reason": (
                "not_applicable_known_failure" if not known_perfect else "pending"
            ),
            "structure": structure(model, 163),
            "decision": "REJECT" if not known_perfect else "PENDING_FRESH",
        }
        rows.append(row)
        print(
            f"{path.name} sha={row['sha256'][:12]} cost={cost} "
            f"disabled={known['disable_all']['right']}/{known['disable_all']['total']} "
            f"default={known['default']['right']}/{known['default']['total']} "
            f"decision={row['decision']}",
            flush=True,
        )
    (HERE / "task163_root_prunes_audit.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
