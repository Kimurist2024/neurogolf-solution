#!/usr/bin/env python3
"""Audit the strongest prior task216 rule-derived controls."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_b16"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_b17"))

from audit_exact import known_dual, structure  # noqa: E402
from audit_candidates import runtime_shapes  # noqa: E402
from lib import scoring  # noqa: E402


CONTROLS: list[tuple[str, str]] = [
    (
        "old_spec_derived_peeling",
        "scripts/golf/scratch/task216/cand3.onnx",
    ),
    (
        "qlinear_true_rule_rebuild",
        "scripts/golf/scratch_wave/task216/cand.onnx",
    ),
    (
        "stacked_edge_repair",
        "scripts/golf/scratch_codex/task216/candidate_low34_from1577.onnx",
    ),
    (
        "all_rows_repair",
        "scripts/golf/scratch_codex/task216/candidate_low34_allrows.onnx",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows: list[dict[str, Any]] = []
    for label, relative in CONTROLS:
        path = ROOT / relative
        if not path.exists():
            rows.append({"label": label, "path": relative, "missing": True})
            continue
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(
            prefix=f"sound103_control_{label}_", dir="/tmp"
        ) as workdir:
            score = scoring.score_and_verify(
                copy.deepcopy(model),
                216,
                workdir,
                label=label,
                require_correct=False,
            )
        try:
            trace = runtime_shapes(model, 216)
        except Exception as exc:  # noqa: BLE001
            trace = {
                "shape_cloak": None,
                "trace_error": f"{type(exc).__name__}: {exc}",
            }
        row = {
            "label": label,
            "path": relative,
            "sha256": sha256(path),
            "score": score,
            "structure": structure(model),
            "runtime_shapes": trace,
            "known_dual": known_dual(model, 216),
        }
        rows.append(row)
        print(
            label,
            score,
            "shape_cloak",
            trace.get("shape_cloak"),
            "mismatches",
            len(trace.get("mismatches", [])),
            "known",
            row["known_dual"],
            flush=True,
        )
    (HERE / "control_audit.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
