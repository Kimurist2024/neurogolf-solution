#!/usr/bin/env python3
"""Discover genuinely cheaper ORT-basic rewrites from the immutable baseline.

The ORT optimizer is used only as a candidate generator.  Every saved model
must remain standard ONNX, pass the checker, match all known examples, and be
cheaper under the same full-example profiling path used by the local scorer.
Fresh/domain verification is deliberately left to the downstream strict gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def standard_and_safe(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    allowed_domains = {"", "ai.onnx", "ai.onnx.ml"}
    banned = {"Loop", "Scan", "NonZero", "Unique", "Compress"}
    for node in model.graph.node:
        if node.domain not in allowed_domains:
            raise ValueError(f"nonstandard domain {node.domain!r}")
        if node.op_type in banned or "Sequence" in node.op_type:
            raise ValueError(f"banned op {node.op_type}")
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("functions/sparse initializers are forbidden")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.base_costs.read_text())["costs"]
    ort.set_default_logger_severity(3)
    winners: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            entry = costs.get(str(task))
            if entry is None or not bool(entry.get("correct", True)):
                continue
            baseline_cost = int(entry["cost"] if isinstance(entry, dict) else entry)
            try:
                original = onnx.load_model_from_string(
                    archive.read(f"task{task:03d}.onnx")
                )
                with tempfile.TemporaryDirectory(prefix=f"ort_basic_{task:03d}_") as td:
                    source = Path(td) / "source.onnx"
                    optimized = Path(td) / "optimized.onnx"
                    onnx.save(original, source)
                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    options.optimized_model_filepath = str(optimized)
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    ort.InferenceSession(str(source), options)
                    candidate = onnx.load(optimized)
                    standard_and_safe(candidate)
                    with tempfile.TemporaryDirectory(prefix=f"ort_score_{task:03d}_") as wd:
                        score = scoring.score_and_verify(
                            candidate, task, wd, label="ort_basic", require_correct=True
                        )
                if not score or int(score["cost"]) >= baseline_cost:
                    continue
                path = args.out_dir / f"task{task:03d}.onnx"
                onnx.save(candidate, path)
                candidate_cost = int(score["cost"])
                item = {
                    "task": task,
                    "path": str(path),
                    "baseline_cost": baseline_cost,
                    "candidate_cost": candidate_cost,
                    "projected_gain": math.log(baseline_cost / candidate_cost),
                    "known_correct": bool(score["correct"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                winners.append(item)
                print(f"task{task:03d}: {baseline_cost}->{candidate_cost}")
            except Exception as exc:  # candidate generation is best-effort
                failures.append({"task": task, "error": repr(exc)})

    payload = {
        "baseline": str(args.baseline),
        "winners": winners,
        "projected_gain": sum(float(x["projected_gain"]) for x in winners),
        "failures": failures,
    }
    manifest = args.out_dir / "manifest_known_verified.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "winners": len(winners),
        "projected_gain": payload["projected_gain"],
        "failures": len(failures),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
