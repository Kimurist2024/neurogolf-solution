#!/usr/bin/env python3
"""Explain the four known failures of the isolated task175 POLICY90 lead."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = HERE / "baseline/task175.onnx"
CANDIDATE = HERE.parent / "root_sweep29/prune_latent/task175_r001.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def session(path: Path) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitization failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> int:
    authority = session(AUTHORITY)
    candidate = session(CANDIDATE)
    examples = scoring.load_examples(175)
    rows: list[dict[str, Any]] = []
    global_index = 0
    mask_equal = 0
    raw_equal = 0
    for split in ("train", "test", "arc-gen"):
        for split_index, example in enumerate(examples[split]):
            converted = scoring.convert_to_numpy(example)
            if converted is None:
                continue
            feeds = {"input": converted["input"]}
            ref = np.asarray(authority.run(["output"], feeds)[0])
            got = np.asarray(candidate.run(["output"], feeds)[0])
            expected = converted["output"].astype(bool)
            same_mask = np.array_equal(got > 0.0, ref > 0.0)
            same_raw = np.array_equal(got, ref)
            mask_equal += int(same_mask)
            raw_equal += int(same_raw)
            if np.array_equal(got > 0.0, expected):
                global_index += 1
                continue
            differences = np.argwhere((got > 0.0) != expected)
            cells = []
            for batch, channel, row, col in differences[:30]:
                cells.append(
                    {
                        "channel": int(channel),
                        "row": int(row),
                        "col": int(col),
                        "candidate_raw": float(got[batch, channel, row, col]),
                        "authority_raw": float(ref[batch, channel, row, col]),
                        "expected_on": bool(expected[batch, channel, row, col]),
                    }
                )
            rows.append(
                {
                    "global_index": global_index,
                    "split": split,
                    "split_index": split_index,
                    "grid_shape": [len(example["input"]), len(example["input"][0])],
                    "threshold_difference_count": int(len(differences)),
                    "candidate_mask_equal_authority": same_mask,
                    "candidate_raw_equal_authority": same_raw,
                    "first_differences": cells,
                }
            )
            global_index += 1
    base = onnx.load(AUTHORITY)
    cand = onnx.load(CANDIDATE)
    base_shapes = {item.name: list(item.dims) for item in base.graph.initializer}
    cand_shapes = {item.name: list(item.dims) for item in cand.graph.initializer}
    changed = {
        name: {"authority": shape, "candidate": cand_shapes.get(name)}
        for name, shape in base_shapes.items()
        if cand_shapes.get(name) != shape
    }
    result = {
        "task": 175,
        "cause": (
            "The candidate removes latent index L=1 by shrinking the shared "
            "C0/G1 L axis from 2 to 1.  That component is not algebraically zero."
        ),
        "changed_initializer_shapes": changed,
        "known_cases": global_index,
        "candidate_mask_equal_authority": mask_equal,
        "candidate_raw_equal_authority": raw_equal,
        "known_failure_count": len(rows),
        "failures": rows,
    }
    output = HERE / "evidence/policy90_failure_details.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "known_cases", "candidate_mask_equal_authority",
        "candidate_raw_equal_authority", "known_failure_count"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
