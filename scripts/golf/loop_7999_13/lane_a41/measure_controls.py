#!/usr/bin/env python3
"""Measure authority and exact-control models with the official local scorer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import onnx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.scoring import score_and_verify  # noqa: E402


LANE = Path(__file__).resolve().parent
MODELS = {
    "authority": LANE / "baseline_task366.onnx",
    "identity_bypass": LANE / "probe_identity.onnx",
    "truthful_single_trace_control": LANE / "truthful_annotation_control.onnx",
    "truthful_identity_bypass": LANE / "truthful_identity_bypass.onnx",
}


def main() -> None:
    output = LANE / "control_costs.json"
    results: dict[str, object] = (
        json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    )
    for name, path in MODELS.items():
        if name in results and "score_and_verify" in results[name]:
            continue
        try:
            measured = score_and_verify(
                onnx.load(path),
                366,
                str(LANE / "score_work"),
                label=name,
                require_correct=False,
            )
            results[name] = {
                "path": str(path.relative_to(ROOT)),
                "score_and_verify": measured,
            }
        except Exception as exc:  # preserve diagnostic evidence
            results[name] = {
                "path": str(path.relative_to(ROOT)),
                "exception": f"{type(exc).__name__}: {exc}",
            }
    output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
