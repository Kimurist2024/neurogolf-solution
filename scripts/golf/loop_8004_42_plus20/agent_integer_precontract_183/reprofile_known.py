#!/usr/bin/env python3
"""Reprofile every candidate on a valid known input with the competition scorer."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE / "base"

sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402


def first_known(task: int):
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                return converted["input"]
    raise RuntimeError(task)


def profile(path: Path, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"precontract183_{task}_") as directory:
        model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
        if model is None:
            raise RuntimeError("sanitizer rejected")
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.profile_file_prefix = os.path.join(directory, f"trace_{uuid.uuid4().hex[:8]}")
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        session.run(["output"], {"input": first_known(task)})
        trace = session.end_profiling()
        memory, params = scoring.score_network(model, trace)
        if memory is None or params is None:
            raise RuntimeError((memory, params))
        return {"memory": int(memory), "params": int(params), "cost": int(memory) + int(params)}


def main() -> None:
    source = json.loads((HERE / "profile_results.json").read_text())
    baselines = {str(task): profile(BASE / f"task{task:03d}.onnx", task) for task in (74, 200, 211)}
    rows = []
    for index, row in enumerate(source["rows"], 1):
        item = dict(row)
        task = int(item["task"])
        actual = profile(REPO / item["path"], task)
        item["zero_screen_profile"] = item.pop("actual_profile")
        item["actual_profile"] = actual
        item["baseline_profile"] = baselines[str(task)]
        item["cost_delta"] = actual["cost"] - baselines[str(task)]["cost"]
        item["strict_lower"] = item["cost_delta"] < 0
        rows.append(item)
        if index % 20 == 0:
            print(f"profiled {index}/{len(source['rows'])}", flush=True)
    lower = [row for row in rows if row["strict_lower"]]
    result = dict(source)
    result["profiling_input"] = "first convertible known example per task"
    result["profiling_runtime"] = "ORT_DISABLE_ALL_threads1"
    result["baseline_profiles"] = baselines
    result["rows"] = rows
    result["strict_lower_count"] = len(lower)
    result["deep_policy"] = {
        "full_checker": "COMPETITION_PROFILER_IMPLICIT_PASS_ALL; LOWER_ONLY_STANDALONE_NOT_RUN" if not lower else "REQUIRED",
        "strict_data_prop": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
        "truthful_trace": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
        "known4": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
        "fresh_each_seed_10000": "LOWER_ONLY_NOT_RUN_NO_LOWER" if not lower else "REQUIRED",
    }
    (HERE / "actual_profile_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baselines": baselines, "strict_lower_count": len(lower)}, indent=2))
    if lower:
        (HERE / "strict_lower_requires_deep.json").write_text(json.dumps(lower, indent=2) + "\n")
        raise RuntimeError("strict-lower candidate requires lower-only deep audit")


if __name__ == "__main__":
    main()
