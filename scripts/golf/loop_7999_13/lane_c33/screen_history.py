#!/usr/bin/env python3
"""Dual-ORT fail-fast screen for all below-baseline C33 history models."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def screen(path: Path, task: int) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    examples = scoring.load_examples(task)
    modes = []
    for mode in ("disabled", "default"):
        right = wrong = errors = 0
        first = None
        try:
            session = make_session(model, mode)
        except Exception as exc:  # noqa: BLE001
            modes.append({"mode": mode, "right": 0, "wrong": 0, "errors": 1, "first_failure": repr(exc)})
            continue
        stopped = False
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = session.run(["output"], {"input": benchmark["input"]})[0]
                    if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                        right += 1
                    else:
                        wrong += 1
                        first = {"subset": subset, "index": index}
                        stopped = True
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    first = {"subset": subset, "index": index, "error": repr(exc)}
                    stopped = True
                if stopped:
                    break
            if stopped:
                break
        modes.append({"mode": mode, "right": right, "wrong": wrong, "errors": errors, "first_failure": first})
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ops": sorted({node.op_type for node in model.graph.node}),
        "prohibited_lookup_or_vectorizer": any(
            node.op_type in {"TfIdfVectorizer", "ScatterElements", "GatherND", "ScatterND"}
            for node in model.graph.node
        ),
        "known_dual_fail_fast": modes,
    }


def main() -> int:
    inventory = json.loads((HERE / "history_inventory.json").read_text())
    candidates = []
    seen = set()
    for task in (143, 301):
        base = inventory["tasks"][str(task)]["baseline_cost"]
        for row in inventory["tasks"][str(task)]["models"]:
            if row["cost"] >= base:
                continue
            path = ROOT / row["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append((path, task))
    rows = []
    for path, task in candidates:
        row = screen(path, task)
        rows.append(row)
        print(task, path.name, row["known_dual_fail_fast"], flush=True)
    (HERE / "history_screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
