#!/usr/bin/env python3
"""Audit B18 baselines and plausible candidates with the full safety pre-gate."""

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
sys.path.insert(0, str(HERE.parent / "lane_b16"))
sys.path.insert(0, str(HERE.parent / "lane_b17"))

from audit_exact import known_dual, structure  # noqa: E402
from audit_candidates import runtime_shapes  # type: ignore[no-redef]  # noqa: E402
from lib import scoring  # noqa: E402


BASE_COST = {89: 1361, 255: 1336}
CANDIDATES = {
    89: [
        HERE / "baseline/task089.onnx",
        ROOT / "scripts/golf/loop_7999_13/wave12_exact_shave/task089.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task089_r01_static1184.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task089_r02_static1184.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task089_r03_static1298.onnx",
        # Ground-up, generator-derived controls. These are not expected to beat
        # the shape-cloaked baseline, but prove the truthful cost scale.
        ROOT / "scripts/golf/scratch/task089/cand_u8.onnx",
        ROOT / "scripts/golf/scratch_codex/task089/candidate_rebuild_v11.onnx",
        ROOT / "scripts/golf/scratch_codex/task089/candidate_red_safe3_rebuilt.onnx",
        ROOT / "scripts/golf/scratch_claude/task089/rebuild.onnx",
    ],
    255: [
        HERE / "baseline/task255.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task255_r01_static814.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task255_r02_static814.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task255_r03_static814.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task255_r04_static878.onnx",
        ROOT / "scripts/golf/scratch_claude/task255/recon.onnx",
        ROOT / "scripts/golf/scratch_codex/task255/ambiguity_proof.py",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(task: int, path: Path, model: onnx.ModelProto) -> list[str]:
    reasons: list[str] = []
    if task == 255:
        reasons.append("generator_non_functional_same_input_different_output")
    if path.name.startswith("task089_r"):
        reasons.append("archive_candidate_requires_runtime_shape_audit")
    if path.name.startswith("task255_r"):
        reasons.append("archive_candidate_public_rule_guess")
    max_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    if max_inputs >= 15:
        reasons.append(f"giant_einsum_{max_inputs}_inputs")
    return reasons


def main() -> int:
    ort.set_default_logger_severity(4)
    rows: list[dict[str, Any]] = []
    for task, paths in CANDIDATES.items():
        seen: set[str] = set()
        for path in paths:
            if not path.exists() or path.suffix != ".onnx":
                continue
            digest = sha256(path)
            if digest in seen:
                continue
            seen.add(digest)
            model = onnx.load(path)
            with tempfile.TemporaryDirectory(
                prefix=f"b18_{task}_{digest[:8]}_", dir="/tmp"
            ) as workdir:
                score = scoring.score_and_verify(
                    copy.deepcopy(model),
                    task,
                    workdir,
                    label=digest[:8],
                    require_correct=False,
                )
            structural = structure(model)
            try:
                trace = runtime_shapes(model, task)
            except Exception as exc:  # noqa: BLE001
                trace = {
                    "shape_cloak": None,
                    "trace_error": f"{type(exc).__name__}: {exc}",
                }
            dual = known_dual(model, task)
            semantic_rejections = semantic(task, path, model)
            known_ok = all(
                row.get("wrong") == 0 and row.get("errors") == 0 for row in dual
            )
            pre_fresh = (
                score is not None
                and score["cost"] < BASE_COST[task]
                and structural["pass"]
                and trace.get("shape_cloak") is False
                and known_ok
                and not semantic_rejections
            )
            row = {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "actual_score": score,
                "structure": structural,
                "runtime_shapes": trace,
                "known_dual": dual,
                "semantic_rejections": semantic_rejections,
                "eligible_for_fresh5000": pre_fresh,
            }
            rows.append(row)
            print(
                task,
                path.name,
                score,
                "cloak",
                trace.get("shape_cloak"),
                "known",
                [
                    (item.get("right"), item.get("wrong"), item.get("errors"))
                    for item in dual
                ],
                "semantic",
                semantic_rejections,
                "pre_fresh",
                pre_fresh,
                flush=True,
            )
    report = {
        "base_cost_recomputed_from_exact_zip": BASE_COST,
        "assignment_cost_mismatch": {
            "task255_assignment": 1162,
            "task255_exact_zip_recomputed": 1336,
        },
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
