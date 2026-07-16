#!/usr/bin/env python3
"""Audit the existing task216 rule-derived rebuild outside archive history."""

from __future__ import annotations

import json

from audit_history import HERE, ROOT, audit


def main() -> None:
    path = ROOT / "scripts/golf/scratch_wave/task216/cand.onnx"
    row = audit(
        216,
        "task216_rule_rebuild_existing",
        path,
        None,
        ["scripts/golf/scratch_wave/task216/cand.onnx"],
    )
    (HERE / "extra_audit.json").write_text(json.dumps(row, indent=2) + "\n")
    print(row.get("official_like_score"), row["pre_fresh_reasons"])


if __name__ == "__main__":
    main()
