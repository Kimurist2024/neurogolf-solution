#!/usr/bin/env python3
"""Package the exact-authority-equivalent task023 reduction for probing."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "others/71407/exact_authority_equivalent_8012_15_wave1"
CANDIDATE = HERE / "task023_cost1317_exact.onnx"
CANDIDATE_SHA = "c9725627bc4aaa49494c4da4ff6c06849e38cd4fffd0c0fd3c64afdf5ce1472c"

sys.path.insert(0, str(ROOT / "scripts/golf/root_nonblack_checkpoint_410"))
import build as checkpoint  # noqa: E402


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint: {OUT}")
    evidence = json.loads((HERE / "evidence.json").read_text(encoding="utf-8"))
    if not evidence.get("admitted") or not evidence.get("raw_equivalent_to_verified_authority"):
        raise RuntimeError("task023 exact-equivalence audit is not admitted")
    data = CANDIDATE.read_bytes()
    if checkpoint.sha256(data) != CANDIDATE_SHA:
        raise RuntimeError("task023 candidate SHA drift")
    if checkpoint.sha256(checkpoint.BASE.read_bytes()) != checkpoint.BASE_SHA256:
        raise RuntimeError("authority SHA drift")

    (OUT / "candidates").mkdir(parents=True)
    (OUT / "evidence").mkdir()
    shutil.copy2(CANDIDATE, OUT / "candidates/task023.onnx")
    shutil.copy2(HERE / "evidence.json", OUT / "evidence/task023.json")
    zip_row = checkpoint.write_zip(
        OUT / "submission_task023_EXACT_AUTHORITY_EQUIVALENT_ONLY.zip",
        {"task023.onnx": data},
    )
    gain = math.log(1321 / 1317)
    manifest = {
        "authority": {
            "path": str(checkpoint.BASE.relative_to(ROOT)),
            "sha256": checkpoint.BASE_SHA256,
            "lb": checkpoint.BASE_LB,
        },
        "candidate": {
            "task": 23,
            "authority_cost": 1321,
            "candidate_cost": 1317,
            "sha256": CANDIDATE_SHA,
            "gain": gain,
            "classification": "EXACT_AUTHORITY_EQUIVALENT",
            "raw_mismatch_across_known_and_8000_fresh_config_runs": 0,
            "proof": (
                "two same-shape duplicate initializers plus two scalar broadcasts; "
                "all replacement constants have identical dtype/value"
            ),
        },
        "known_black_exact_exclusions": [70, 134, 202, 343],
        "known_black_candidates_present": False,
        "projected_lb": checkpoint.BASE_LB + gain,
        "zip": zip_row,
        "root_authority_modified": False,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# task023 exact-authority-equivalent checkpoint\n\n"
        "task023 is reduced from **1321 to 1317** (`+0.003032603`). The graph's "
        "raw output is identical to the verified 8012.15 authority in all four "
        "ORT configurations over known cases and two fresh streams. This is an "
        "exact rewrite, not a POLICY90 approximation.\n\n"
        "The candidate removes two byte-identical initializers and reuses two "
        "same-dtype/value scalar constants under standard broadcasting. The four "
        "latest black candidates 070/134/202/343 are absent. Root files are unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
