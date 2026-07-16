#!/usr/bin/env python3
"""Package wave2 plus feedback-loop POLICY90 admissions for individual probes."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_BUILDER = ROOT / "scripts/golf/root_nonblack_checkpoint_wave2_415/build.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("wave2_checkpoint_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(BASE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = load_builder()
    builder.OUT = ROOT / "others/71407/nonblack_policy90_8012_15_wave3"
    extra = (
        {
            "task": 110,
            "source": ROOT / "others/71407/continuous_30m/cycle_0005_20260715T022451Z_cost21_50/repair_3w/candidates/task110_POLICY90_cost10_2d108b240cff.onnx",
            "sha256": "2d108b240cffa7224457911c2ec94460236bf45ecdfee66c208e17eddf1fda76",
            "authority_cost": 24,
            "candidate_cost": 10,
            "classification": "NONBLACK_POLICY90_FEEDBACK_REPAIR",
            "minimum_accuracy": 0.9849624060150376,
            "risk": "feedback repair worker; fresh 99.95% and 100.00%",
        },
        {
            "task": 188,
            "source": ROOT / "others/71407/continuous_30m/cycle_0005_20260715T022451Z_cost21_50/repair_3w/candidates/task188_POLICY90_cost39_4e2813d73ce8.onnx",
            "sha256": "4e2813d73ce85252560e777483deae9176b41688ed9a13553f3b7c11e4d00a83",
            "authority_cost": 46,
            "candidate_cost": 39,
            "classification": "NONBLACK_POLICY90_FEEDBACK_REPAIR",
            "minimum_accuracy": 0.9405,
            "risk": "feedback repair worker; fresh minimum 94.05%",
        },
    )
    builder.CANDIDATES = tuple(builder.CANDIDATES) + extra
    builder.main()

    out = builder.OUT
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_dir = out / "evidence"
    for worker_id in (1, 2):
        source = ROOT / "others/71407/continuous_30m/cycle_0005_20260715T022451Z_cost21_50/repair_3w" / f"worker_{worker_id}.json"
        shutil.copy2(source, evidence_dir / f"feedback_worker_{worker_id}.json")
    extra_gain = sum(math.log(row["authority_cost"] / row["candidate_cost"]) for row in extra)
    manifest["feedback_loop"] = {
        "cycle": 5,
        "source": "others/71407/continuous_30m/cycle_0005_20260715T022451Z_cost21_50/repair_3w",
        "new_tasks": [110, 188],
        "new_gain": extra_gain,
        "fresh_and_structure_audited": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("wave 2", "wave 3")
    text += (
        "\n\nFeedback repair admissions added in cycle 5:\n"
        "- task110: 24→10, +0.875469 (fresh 99.95% / 100.00%)\n"
        "- task188: 46→39, +0.165080 (fresh minimum 94.05%)\n"
        "\nThese remain individual-probe candidates; no LB guarantee is claimed.\n"
    )
    readme.write_text(text, encoding="utf-8")
    print(json.dumps({
        "output": str(out.relative_to(ROOT)),
        "new_tasks": [110, 188],
        "new_gain": extra_gain,
        "projected_lb": builder.BASE_LB + float(manifest["projected_gain"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
