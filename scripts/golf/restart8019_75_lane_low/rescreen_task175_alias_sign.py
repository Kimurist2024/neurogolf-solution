#!/usr/bin/env python3
"""Rescreen historical task175 alias near-misses by sign before margin repair."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "scripts/golf/restart8018_91_lane_low"
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


spec = importlib.util.spec_from_file_location("old_alias", OLD / "search_task175_full_alias_gauge.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import historical alias builder")
old_alias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_alias)


def scaled_model(alias_from: str, alias_to: str, mask: int, scale: float):
    model = old_alias.make_model(alias_from, alias_to, mask)
    replacements = []
    for item in model.graph.initializer:
        if item.name == "TA":
            array = np.asarray(numpy_helper.to_array(item), dtype=np.float32) * np.float32(scale)
            replacements.append(numpy_helper.from_array(array, item.name))
        else:
            replacements.append(item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(replacements)
    return model


def make_session(model):
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


examples = scoring.load_examples(175)
cases = []
for subset in ("train", "test", "arc-gen"):
    for example in examples.get(subset, []):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is not None:
            cases.append(benchmark)


def screen(entry: dict[str, object]) -> dict[str, object]:
    row = {"alias_from": entry["alias_from"], "alias_to": entry["alias_to"],
           "mask": entry["mask"]}
    try:
        model = scaled_model(str(entry["alias_from"]), str(entry["alias_to"]),
                             int(entry["mask"]), 2.0)
        sess = make_session(model)
        min_positive = float("inf")
        max_false = -float("inf")
        small = 0
        for index, benchmark in enumerate(cases):
            raw = sess.run(["output"], {"input": benchmark["input"]})[0]
            target = benchmark["output"] > 0.0
            if not np.array_equal(raw > 0.0, target):
                row.update({"sign_exact": False, "first_wrong": index})
                return row
            if target.any():
                min_positive = min(min_positive, float(raw[target].min()))
            if (~target).any():
                max_false = max(max_false, float(raw[~target].max()))
            small += int(((raw > 0.0) & (raw < 0.25)).sum())
        row.update({"sign_exact": True, "min_positive": min_positive,
                    "max_false": max_false, "small_positive": small})
        return row
    except Exception as exc:
        row.update({"sign_exact": False, "error": f"{type(exc).__name__}: {exc}"})
        return row


def main() -> int:
    history = json.loads((OLD / "task175_full_alias_gauge_search.json").read_text())
    entries = [row for row in history["rows"]
               if str(row.get("reject", "")).startswith("small_positive")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(screen, entries))
    winners = [row for row in rows if row.get("sign_exact")]
    result = {"screened": len(rows), "sign_exact_count": len(winners),
              "sign_exact": winners, "rows": rows}
    (HERE / "task175_alias_sign_rescreen.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in
                      ("screened", "sign_exact_count", "sign_exact")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
