#!/usr/bin/env python3
"""Independent dual-ORT known/fresh validation of Wave3's task109 member."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WAVE3 = HERE.parent / "submission_8003.40_wave3_safe_meta.zip"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
from lib import scoring  # noqa: E402


def task109_model() -> onnx.ModelProto:
    with zipfile.ZipFile(WAVE3) as archive:
        names = [name for name in archive.namelist() if Path(name).name.lower() == "task109.onnx"]
        if len(names) != 1:
            raise RuntimeError(f"expected one task109 member, got {names}")
        return onnx.load_model_from_string(archive.read(names[0]))


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected task109")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def validate(session: ort.InferenceSession, examples: list[dict[str, object]]) -> dict[str, object]:
    right = mismatches = runtime_errors = invalid_examples = 0
    for example in examples:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            invalid_examples += 1
            continue
        try:
            raw = session.run(["output"], {"input": benchmark["input"]})[0]
        except Exception:  # noqa: BLE001
            runtime_errors += 1
            continue
        if np.array_equal(raw > 0, benchmark["output"] > 0):
            right += 1
        else:
            mismatches += 1
    return {
        "requested": len(examples),
        "right": right,
        "mismatches": mismatches,
        "runtime_errors": runtime_errors,
        "invalid_examples": invalid_examples,
        "perfect": right == len(examples) and mismatches == runtime_errors == invalid_examples == 0,
    }


def main() -> None:
    ort.set_default_logger_severity(3)
    model = task109_model()
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    known_data = json.loads((ROOT / "inputs/neurogolf-2026/task109.json").read_text())
    known = [*known_data["train"], *known_data["test"], *known_data["arc-gen"]]

    generator = importlib.import_module("tasks.task_47c1f68c")
    random.seed(109_803_341)
    fresh: list[dict[str, object]] = []
    generation_errors = 0
    for _ in range(5000):
        try:
            example = generator.generate()
            if isinstance(example, dict) and "input" in example and "output" in example:
                fresh.append(example)
            else:
                generation_errors += 1
        except Exception:  # noqa: BLE001
            generation_errors += 1

    modes = {}
    for disable_all, label in ((True, "disable_all"), (False, "default")):
        session = make_session(model, disable_all)
        modes[label] = {
            "known": validate(session, known),
            "fresh": validate(session, fresh),
        }

    report = {
        "task": 109,
        "wave3": str(WAVE3.relative_to(ROOT)),
        "seed": 109_803_341,
        "known_count": len(known),
        "fresh_requested": 5000,
        "fresh_generated": len(fresh),
        "generation_errors": generation_errors,
        "modes": modes,
        "perfect": bool(
            generation_errors == 0
            and len(fresh) == 5000
            and all(
                details[subset]["perfect"]
                for details in modes.values()
                for subset in ("known", "fresh")
            )
        ),
    }
    (HERE / "fresh_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
