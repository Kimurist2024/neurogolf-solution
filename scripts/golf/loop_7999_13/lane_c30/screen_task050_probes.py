#!/usr/bin/env python3
"""Fail-fast dual-ORT screen for the task050 common-transition probes."""

from __future__ import annotations

import copy
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


def session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> int:
    examples = scoring.load_examples(50)
    rows = []
    for path in sorted(HERE.glob("task050_common_k_*.onnx")):
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        modes = []
        for mode in ("disabled", "default"):
            sess = session(model, mode)
            right = wrong = errors = 0
            first_failure = None
            stop = False
            for subset in ("train", "test", "arc-gen"):
                for index, example in enumerate(examples[subset]):
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    try:
                        raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                        if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                            right += 1
                        else:
                            wrong += 1
                            first_failure = {"subset": subset, "index": index}
                            stop = True
                            break
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        first_failure = {
                            "subset": subset,
                            "index": index,
                            "error": repr(exc),
                        }
                        stop = True
                        break
                if stop:
                    break
            modes.append(
                {
                    "mode": mode,
                    "right_before_stop": right,
                    "wrong": wrong,
                    "errors": errors,
                    "first_failure": first_failure,
                }
            )
        rows.append({"path": path.name, "modes": modes})
        print(path.name, modes, flush=True)
    (HERE / "task050_probe_screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
