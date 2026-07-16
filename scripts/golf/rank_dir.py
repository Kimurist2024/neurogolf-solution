#!/usr/bin/env python3
"""Compute per-task cost (memory + params) for every taskXXX.onnx in a dir.

Cost-only: profiles each model once under ORT_DISABLE_ALL with a zero input
(shapes are static, so a zero [1,10,30,30] input yields the official trace
shapes) and adds calculate_params. Writes a cost-ranked JSON. No gold check,
so this is fast and works on an arbitrary baseline directory (e.g. an
extracted submission zip).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from lib import scoring  # noqa: E402

_ZERO = np.zeros((1, 10, 30, 30), dtype=np.float32)
_TASK_RE = re.compile(r"task(\d+)\.onnx$")


def cost_of(path: str) -> tuple[int, int, int]:
    """Return (memory, params, cost) for one onnx file."""
    with tempfile.TemporaryDirectory(prefix="rank_") as wd:
        sanitized = scoring.sanitize_model(onnx.load(path))
        if sanitized is None:
            return (-1, -1, -1)
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.profile_file_prefix = os.path.join(wd, f"p_{uuid.uuid4().hex[:8]}")
        sess = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
        try:
            sess.run(["output"], {"input": _ZERO})
        except Exception:
            # data-independent shapes should never need real data; if a model
            # rejects the zero input, fall back to a single train example.
            pass
        trace = sess.end_profiling()
        memory, params = scoring.score_network(sanitized, trace)
        if memory is None or params is None:
            return (-1, -1, -1)
        return (int(memory), int(params), int(memory) + int(params))


def _job(item: tuple[int, str]) -> tuple[int, int, int, int]:
    task, path = item
    mem, par, cost = cost_of(path)
    return (task, mem, par, cost)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: rank_dir.py <model_dir> <out.json>", file=sys.stderr)
        return 2
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    items: list[tuple[int, str]] = []
    for p in sorted(model_dir.glob("task*.onnx")):
        m = _TASK_RE.search(p.name)
        if m:
            items.append((int(m.group(1)), str(p)))

    results: dict[int, dict[str, int]] = {}
    workers = max(1, (os.cpu_count() or 4) - 1)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = ex.map(_job, items)
            for task, mem, par, cost in rows:
                results[task] = {"memory": mem, "params": par, "cost": cost}
    except PermissionError:
        # Some managed sandboxes deny the sysconf call used by
        # ProcessPoolExecutor.  Cost profiling is still valid sequentially.
        for item in items:
            task, mem, par, cost = _job(item)
            results[task] = {"memory": mem, "params": par, "cost": cost}

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["cost"])
    payload = {
        "model_dir": str(model_dir),
        "n": len(results),
        "ranked": [{"task": t, **v} for t, v in ranked],
        "costs": {str(t): v["cost"] for t, v in results.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} tasks)")
    print("TOP 20 by cost:")
    for entry in payload["ranked"][:20]:
        print(f"  task{entry['task']:03d}  cost={entry['cost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
