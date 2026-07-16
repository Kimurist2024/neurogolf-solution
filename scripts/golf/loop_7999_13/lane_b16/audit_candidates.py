#!/usr/bin/env python3
"""Audit every plausible below-baseline B16 candidate; never promote."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from audit_exact import known_dual, runtime_shapes, structure  # noqa: E402
from lib import scoring  # noqa: E402


BASE_COST = {157: 853, 319: 1023}
CANDIDATES = {
    157: [
        ROOT / "scripts/golf/scratch_codex/task157/header_pack64_codex.onnx",
        ROOT / "scripts/golf/scratch_codex/task157/nokeys_probe.onnx",
        HERE / "candidate_task157_no_lookup.onnx",
    ],
    319: [
        ROOT / f"scripts/golf/loop_7999_13/lane_archive_all400/task319_r{rank:02d}_static{cost}.onnx"
        for rank, cost in ((1, 719), (2, 721), (3, 748), (4, 793), (5, 794))
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_rejections(task: int, model: onnx.ModelProto) -> list[str]:
    initializers = {item.name for item in model.graph.initializer}
    reasons: list[str] = []
    if any(name.startswith("fixk_") for name in initializers):
        reasons.append("visible_fixture_key_lookup")
    if any("corr_pattern" in name for name in initializers):
        reasons.append("fixed_pattern_correction_lookup")
    if task == 319:
        reasons.append("generator_proven_non_injective")
    return reasons


def main() -> int:
    ort.set_default_logger_severity(4)
    rows: list[dict[str, Any]] = []
    for task, paths in CANDIDATES.items():
        seen: set[str] = set()
        for path in paths:
            if not path.exists():
                continue
            digest = sha256(path)
            if digest in seen:
                continue
            seen.add(digest)
            model = onnx.load(path)
            with tempfile.TemporaryDirectory(
                prefix=f"b16_{task:03d}_{digest[:8]}_", dir="/tmp"
            ) as workdir:
                score = scoring.score_and_verify(
                    copy.deepcopy(model),
                    task,
                    workdir,
                    label=digest[:8],
                    require_correct=False,
                )
            try:
                trace = runtime_shapes(model, task)
            except Exception as exc:  # noqa: BLE001
                trace = {
                    "shape_cloak": None,
                    "trace_error": f"{type(exc).__name__}: {exc}",
                }
            structural = structure(model)
            dual = known_dual(model, task)
            semantic = semantic_rejections(task, model)
            known_perfect = all(
                row.get("wrong") == 0 and row.get("errors") == 0 for row in dual
            )
            eligible = (
                score is not None
                and score["cost"] < BASE_COST[task]
                and structural["pass"]
                and trace.get("shape_cloak") is False
                and known_perfect
                and not semantic
            )
            row = {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "actual_score": score,
                "structure": structural,
                "runtime_shapes": trace,
                "known_dual": dual,
                "semantic_rejections": semantic,
                "eligible_for_fresh5000": eligible,
            }
            rows.append(row)
            print(
                task,
                path.name,
                score,
                "shape_cloak=",
                trace.get("shape_cloak"),
                "known=",
                [(item.get("right"), item.get("wrong"), item.get("errors")) for item in dual],
                "semantic=",
                semantic,
                "eligible=",
                eligible,
                flush=True,
            )
    report = {
        "base_cost": BASE_COST,
        "rows": rows,
        "eligible_for_fresh5000": [
            row for row in rows if row["eligible_for_fresh5000"]
        ],
    }
    (HERE / "candidate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
