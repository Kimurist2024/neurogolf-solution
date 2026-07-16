#!/usr/bin/env python3
"""Strict dual-ORT and runtime-shape audit of the exact 8008.14 members."""

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
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_b16"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_b17"))

from audit_exact import known_dual, structure  # noqa: E402
from audit_candidates import runtime_shapes  # noqa: E402
from lib import scoring  # noqa: E402


TASKS = (216, 255)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_known(model: onnx.ModelProto, task: int) -> list[dict[str, Any]]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    examples = scoring.load_examples(task)
    rows: list[dict[str, Any]] = []
    for mode in ("disabled", "default"):
        row: dict[str, Any] = {
            "mode": mode,
            "right": 0,
            "wrong": 0,
            "errors": 0,
            "nonfinite": 0,
            "mid_margin": 0,
            "min_positive": None,
            "first_failure": None,
        }
        if sanitized is None:
            row["errors"] = 1
            row["first_failure"] = {"phase": "sanitize"}
            rows.append(row)
            continue
        options = ort.SessionOptions()
        if mode == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        try:
            session = ort.InferenceSession(sanitized.SerializeToString(), options)
        except Exception as exc:  # noqa: BLE001
            row["errors"] = 1
            row["first_failure"] = {
                "phase": "session",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                sample = scoring.convert_to_numpy(example)
                if sample is None:
                    continue
                try:
                    raw = np.asarray(
                        session.run(["output"], {"input": sample["input"]})[0]
                    )
                    finite = np.isfinite(raw)
                    row["nonfinite"] += int(np.count_nonzero(~finite))
                    row["mid_margin"] += int(
                        np.count_nonzero(finite & (raw > 0.0) & (raw < 0.25))
                    )
                    positive = raw[finite & (raw > 0.0)]
                    if positive.size:
                        current = float(positive.min())
                        prior = row["min_positive"]
                        row["min_positive"] = current if prior is None else min(prior, current)
                    if np.array_equal(raw > 0.0, sample["output"].astype(bool)):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "subset": subset,
                            "index": index,
                        }
                except Exception as exc:  # noqa: BLE001
                    row["errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "subset": subset,
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        rows.append(row)
    return rows


def main() -> int:
    ort.set_default_logger_severity(4)
    result: dict[str, Any] = {"tasks": {}}
    for task in TASKS:
        path = HERE / f"task{task:03d}.onnx"
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(
            prefix=f"sound103_{task}_", dir="/tmp"
        ) as workdir:
            score = scoring.score_and_verify(
                copy.deepcopy(model),
                task,
                workdir,
                label="authority",
                require_correct=False,
            )
        try:
            trace = runtime_shapes(model, task)
        except Exception as exc:  # noqa: BLE001
            trace = {
                "shape_cloak": None,
                "trace_error": f"{type(exc).__name__}: {exc}",
            }
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "score": score,
            "structure": structure(model),
            "runtime_shapes": trace,
            "known_dual_summary": known_dual(model, task),
            "known_dual_raw": raw_known(model, task),
        }
        result["tasks"][str(task)] = row
        print(
            task,
            score,
            "shape_cloak",
            trace.get("shape_cloak"),
            "mismatches",
            len(trace.get("mismatches", [])),
            "known",
            row["known_dual_summary"],
            flush=True,
        )
    (HERE / "authority_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
