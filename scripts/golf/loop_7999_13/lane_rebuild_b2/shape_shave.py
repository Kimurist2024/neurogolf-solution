#!/usr/bin/env python3
"""Create semantics-identical value_info-only shaves of exact 7999.13 members."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import score_and_verify  # noqa: E402


TASKS = (23, 80, 138, 187, 204, 216, 379)
BASE_ZIP = ROOT / "submission_base_7999.13.zip"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clear_value_info(model: onnx.ModelProto) -> bytes:
    clone = copy.deepcopy(model)
    clone.graph.ClearField("value_info")
    return clone.SerializeToString(deterministic=True)


def describe_value_info(model: onnx.ModelProto) -> dict[str, list[int]]:
    return {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.value_info
        if item.type.HasField("tensor_type")
        and item.type.tensor_type.HasField("shape")
    }


def make_all_ones(base: onnx.ModelProto) -> onnx.ModelProto:
    candidate = copy.deepcopy(base)
    for item in candidate.graph.value_info:
        if not item.type.HasField("tensor_type"):
            continue
        shape = item.type.tensor_type.shape
        for dim in shape.dim:
            dim.ClearField("dim_param")
            dim.dim_value = 1
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", type=int, default=list(TASKS))
    args = parser.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in args.tasks:
            data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(data)
            base_path = HERE / f"baseline_task{task:03d}.onnx"
            base_path.write_bytes(data)

            candidate = make_all_ones(base)
            candidate_path = HERE / f"shave_all_task{task:03d}.onnx"
            onnx.save(candidate, candidate_path)
            structural = {"checker": False, "strict_shape": False, "error": None}
            try:
                onnx.checker.check_model(candidate, full_check=True)
                structural["checker"] = True
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
                structural["strict_shape"] = True
            except Exception as exc:  # candidate remains audit evidence
                structural["error"] = f"{type(exc).__name__}: {exc}"

            result = None
            if structural["checker"] and structural["strict_shape"]:
                result = score_and_verify(
                    candidate,
                    task,
                    str(HERE / "tmp"),
                    label="allones",
                    require_correct=True,
                )
            rows.append(
                {
                    "task": task,
                    "baseline": str(base_path.relative_to(ROOT)),
                    "baseline_sha256": sha(data),
                    "candidate": str(candidate_path.relative_to(ROOT)),
                    "candidate_sha256": sha(candidate_path.read_bytes()),
                    "protobuf_identical_after_clearing_value_info": (
                        clear_value_info(base) == clear_value_info(candidate)
                    ),
                    "changed_value_info": sum(
                        describe_value_info(base).get(name)
                        != describe_value_info(candidate).get(name)
                        for name in set(describe_value_info(base))
                        | set(describe_value_info(candidate))
                    ),
                    "structural": structural,
                    "score": result,
                }
            )
            print(f"task{task:03d}: {result}", flush=True)

    (HERE / "shape_shave_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
