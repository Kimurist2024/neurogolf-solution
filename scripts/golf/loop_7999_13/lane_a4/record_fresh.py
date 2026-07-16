#!/usr/bin/env python3
"""Record the repository's strict task324 fresh-5000 verdict as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_fix  # noqa: E402


def main() -> None:
    candidate = HERE / "candidates" / "task324_synth_quarter.onnx"
    result = verify_fix.verify_one(324, candidate, 5000, 1.0)
    result["runtime_errors"] = 0 if result.get("fresh_fails") == 0 else "included_in_fresh_fails"
    (HERE / "fresh5000.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("decision") != "ADOPT" or result.get("fresh_fails") != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
