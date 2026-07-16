#!/usr/bin/env python3
"""Run unsafe optional-output probes in isolated child processes.

An ORT native crash must not kill the lane controller.  A signal exit is a
terminal runtime-gate rejection and is recorded verbatim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODELS = [
    HERE / "current/task286.onnx",
    HERE / "exact_probes/task286_omit_V_12.onnx",
    HERE / "exact_probes/task286_omit_S_12.onnx",
    HERE / "exact_probes/task286_optional_outputs_a60274805e9d.onnx",
]


def child(code: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONFAULTHANDLER"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "signal": -result.returncode if result.returncode < 0 else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pass": result.returncode == 0,
    }


def profile(path: Path) -> dict[str, object]:
    code = (
        "from scripts.golf.rank_dir import cost_of; "
        f"print(cost_of({str(path)!r}))"
    )
    return child(code)


def zero_run(path: Path, disable: bool) -> dict[str, object]:
    level = "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL"
    code = f"""
import copy
import numpy as np
import onnx
import onnxruntime as ort
from scripts.lib import scoring
m = scoring.sanitize_model(copy.deepcopy(onnx.load({str(path)!r})))
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.{level}
o.intra_op_num_threads = 1
o.inter_op_num_threads = 1
s = ort.InferenceSession(m.SerializeToString(), o, providers=["CPUExecutionProvider"])
y = s.run(["output"], {{"input": np.zeros((1,10,30,30), dtype=np.float32)}})[0]
print(tuple(y.shape))
"""
    return child(code)


def main() -> int:
    rows = []
    for path in MODELS:
        row = {
            "path": str(path.relative_to(ROOT)),
            "official_profile_process": profile(path),
            "zero_disable_all": zero_run(path, True),
            "zero_default": zero_run(path, False),
        }
        row["runtime_gate_pass"] = all(
            row[key]["pass"]
            for key in ("official_profile_process", "zero_disable_all", "zero_default")
        )
        rows.append(row)
        print(
            path.name,
            row["official_profile_process"]["returncode"],
            row["zero_disable_all"]["returncode"],
            row["zero_default"]["returncode"],
            flush=True,
        )
    payload = {
        "purpose": "isolated native-runtime gate for unused Split-output exact probes",
        "rows": rows,
        "decision": "reject every probe with a nonzero or signal child exit",
    }
    (HERE / "optional_probe_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
