#!/usr/bin/env python3
"""Strict B17 candidate gate before any high-K or external validation."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE.parent / "lane_b16"))

from audit_exact import known_dual, shape, structure  # noqa: E402
from lib import scoring  # noqa: E402


BASE_COST = {280: 828, 396: 1019}
CANDIDATES = {
    280: [
        *[
            ROOT / f"scripts/golf/loop_7999_13/lane_archive_all400/task280_r{rank:02d}_static{cost}.onnx"
            for rank, cost in ((1, 648), (2, 650), (3, 650), (4, 651), (5, 676))
        ],
        ROOT / "scripts/golf/scratch_codex/task280/ground_up_det_round884.onnx",
        ROOT / "scripts/golf/scratch_codex/task280/cand_pad20.onnx",
        HERE / "candidate_task280_truthful.onnx",
    ],
    396: [
        *[
            ROOT / f"scripts/golf/loop_7999_13/lane_archive_all400/task396_r{rank:02d}_static{cost}.onnx"
            for rank, cost in ((1, 947), (2, 961), (3, 964), (4, 965), (5, 965))
        ],
        ROOT / "scripts/golf/scratch_codex/task396/cand_rule_k2.onnx",
        ROOT / "scripts/golf/scratch_codex/task396/cand_rule_k3.onnx",
        ROOT / "scripts/golf/scratch_codex/task396/cand_rule_k4.onnx",
        ROOT / "scripts/golf/scratch_codex/task396/cand_rule_k4_occupancy.onnx",
        ROOT / "scripts/golf/scratch_codex/task396/agent_corner_micro.onnx",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    declared = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    traced = copy.deepcopy(model)
    for index, node in enumerate(traced.graph.node):
        node.name = f"trace_{index}_{node.output[0] if node.output else 'empty'}"
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options)
    sample = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert sample is not None
    outputs = session.run(names, {"input": sample["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
    mismatches = [
        {"tensor": name, "declared": declared_shape, "actual": actual[name]}
        for name, declared_shape in declared.items()
        if name in actual and declared_shape != actual[name]
    ]
    return {"shape_cloak": bool(mismatches), "mismatches": mismatches}


def semantic(task: int, path: Path, model: onnx.ModelProto) -> list[str]:
    reasons: list[str] = []
    max_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    if max_inputs >= 15:
        reasons.append(f"giant_einsum_{max_inputs}_inputs")
    if task == 280 and path.name.startswith("task280_r"):
        reasons.append("archive_known_only_no_generator_proof")
    if task == 396 and path.name.startswith("task396_r"):
        reasons.append("archive_known_only_no_generator_proof")
    if task == 396 and path.name.startswith("cand_rule_"):
        reasons.append("compact_shortcut_not_generator_exact")
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
                prefix=f"b17_{task}_{digest[:8]}_", dir="/tmp"
            ) as workdir:
                score = scoring.score_and_verify(
                    copy.deepcopy(model), task, workdir, label=digest[:8], require_correct=False
                )
            structural = structure(model)
            try:
                trace = runtime_shapes(model, task)
            except Exception as exc:  # noqa: BLE001
                trace = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
            dual = known_dual(model, task)
            semantic_rejections = semantic(task, path, model)
            known_ok = all(row.get("wrong") == 0 and row.get("errors") == 0 for row in dual)
            pre_fresh = (
                score is not None
                and score["cost"] < BASE_COST[task]
                and structural["pass"]
                and trace.get("shape_cloak") is False
                and known_ok
                and not any(reason.startswith("giant_einsum") for reason in semantic_rejections)
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
            print(task, path.name, score, "cloak", trace.get("shape_cloak"), "known", [(x.get("right"), x.get("wrong"), x.get("errors")) for x in dual], "semantic", semantic_rejections, "pre_fresh", pre_fresh, flush=True)
    report = {
        "base_cost": BASE_COST,
        "rows": rows,
        "eligible_for_fresh5000": [row for row in rows if row["eligible_for_fresh5000"]],
    }
    (HERE / "candidate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
