#!/usr/bin/env python3
"""Strictly score one representative of every task367 optimizer output."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime

onnxruntime.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402

# Import the same static gate used by the pool scan.  Importing does not run its
# main routine, and keeps the banned-op/static-shape/Conv-bias policy identical.
import scan_existing as scan  # noqa: E402


def main() -> int:
    variants = sorted((HERE / "task367_optimizer_variants").glob("*.onnx"))
    by_hash: dict[str, list[Path]] = {}
    for path in variants:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path)

    rows = []
    for digest, paths in by_hash.items():
        representative = paths[0]
        data = representative.read_bytes()
        ok, reason, model = scan.static_check(data)
        row = {
            "sha256": digest,
            "representative": str(representative.relative_to(ROOT)),
            "passes": [path.stem for path in paths],
            "static": reason,
        }
        if ok and model is not None:
            floor, memory_floor, param_floor = scan.scanner.static_cost_floor(model)
            row.update(
                static_cost_floor=floor,
                static_memory_floor=memory_floor,
                param_floor=param_floor,
            )
            with tempfile.TemporaryDirectory(prefix="c2_task367_") as workdir:
                result = scoring.score_and_verify(
                    model,
                    367,
                    workdir,
                    label="optimizer",
                    require_correct=False,
                )
            row["score"] = result
        rows.append(row)

    output = {"base_cost": 2229, "unique_variants": len(rows), "rows": rows}
    (HERE / "task367_optimizer_scores.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
