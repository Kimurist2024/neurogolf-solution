#!/usr/bin/env python3
"""Build and known-score the sound task349 lookup-table crop.

The task generator uses square factors 2..6.  A maroon core has width 2*r
with r in 1..5, so the normalized contiguous-run masks are
2**(2*r)-1.  Modulo 11 these are exactly {3, 4, 8, 2, 0}.  Therefore the
four tables indexed by ``radius_code`` never use entries 9 or 10 and can be
cropped from length 11 to length 9 without changing generator-domain output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    convert_to_numpy,
    load_examples,
    masks_equal_with_margin,
    model_margin_stable,
    sanitize_model,
    score_and_verify,
)


BASE_ZIP = ROOT / "submission_base_7999.13.zip"
BASE_MEMBER = "task349.onnx"
BASE_PATH = HERE / "baseline_task349.onnx"
CAND_PATH = HERE / "candidates" / "task349_radius_tables_len9.onnx"
REPORT_PATH = HERE / "build_and_known_report.json"

TABLES = {
    "shift_by_mod": np.asarray([32, 1, 16, 2, 4, 1, 1, 1, 8, 1, 1], dtype=np.int32),
    "top_offset_by_mod_i8": np.asarray([-9, 1, -7, -1, -3, 1, 1, 1, -5, 1, 1], dtype=np.int8),
    "hstart_offset_by_mod_i8": np.asarray([-14, 1, -11, -2, -5, 1, 1, 1, -8, 1, 1], dtype=np.int8),
    "hend_offset_by_mod_i8": np.asarray([6, 1, 5, 2, 3, 1, 1, 1, 4, 1, 1], dtype=np.int8),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_report(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    domains = sorted({op.domain for op in inferred.opset_import})
    bad_dims: list[dict[str, object]] = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tt = value.type.tensor_type
        if not tt.HasField("shape"):
            bad_dims.append({"name": value.name, "reason": "missing_shape"})
            continue
        for dim in tt.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                bad_dims.append({"name": value.name, "reason": "dynamic_or_nonpositive"})
                break
    return {
        "checker_full": True,
        "strict_shape_inference": True,
        "domains": domains,
        "functions": len(inferred.functions),
        "inputs": len(inferred.graph.input),
        "outputs": len(inferred.graph.output),
        "bad_dims": bad_dims,
    }


def default_ort_known(model: onnx.ModelProto) -> dict[str, object]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    # Deliberately leave graph_optimization_level at the ORT default.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = load_examples(349)
    right = wrong = errors = 0
    for example in examples["train"] + examples["test"] + examples["arc-gen"]:
        benchmark = convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            got = session.run(["output"], {"input": benchmark["input"]})[0] > 0.0
            if np.array_equal(got, benchmark["output"].astype(bool)):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {"right": right, "wrong": wrong, "errors": errors}


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    CAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        raw = archive.read(BASE_MEMBER)
    BASE_PATH.write_bytes(raw)
    base = onnx.load_from_string(raw)
    cand = copy.deepcopy(base)

    found: set[str] = set()
    for initializer in cand.graph.initializer:
        if initializer.name not in TABLES:
            continue
        actual = numpy_helper.to_array(initializer)
        expected = TABLES[initializer.name]
        if not np.array_equal(actual, expected):
            raise AssertionError(f"unexpected baseline contents for {initializer.name}: {actual}")
        cropped = np.ascontiguousarray(actual[:9])
        initializer.CopyFrom(numpy_helper.from_array(cropped, name=initializer.name))
        found.add(initializer.name)
    if found != set(TABLES):
        raise AssertionError(f"missing tables: {sorted(set(TABLES) - found)}")

    onnx.checker.check_model(cand, full_check=True)
    onnx.shape_inference.infer_shapes(cand, strict_mode=True)
    onnx.save(cand, CAND_PATH)

    work = HERE / "work"
    base_score = score_and_verify(copy.deepcopy(base), 349, str(work), label="base349", require_correct=True)
    cand_score = score_and_verify(copy.deepcopy(cand), 349, str(work), label="crop349", require_correct=True)
    if base_score is None or cand_score is None:
        raise RuntimeError(f"known score failed: base={base_score}, candidate={cand_score}")

    margin_ok, min_abs = model_margin_stable(copy.deepcopy(cand), 349, margin=0.25)
    report = {
        "task": 349,
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_member": BASE_MEMBER,
        "generator_proof": {
            "factor_range": [2, 6],
            "radius_range": [1, 5],
            "normalized_masks": [(1 << (2 * radius)) - 1 for radius in range(1, 6)],
            "radius_codes_mod11": [((1 << (2 * radius)) - 1) % 11 for radius in range(1, 6)],
            "reachable_codes": sorted({((1 << (2 * radius)) - 1) % 11 for radius in range(1, 6)}),
            "cropped_unused_indices": [9, 10],
        },
        "baseline": {
            "path": str(BASE_PATH.relative_to(ROOT)),
            "sha256": sha256(BASE_PATH),
            "bytes": BASE_PATH.stat().st_size,
            **base_score,
        },
        "candidate": {
            "path": str(CAND_PATH.relative_to(ROOT)),
            "sha256": sha256(CAND_PATH),
            "bytes": CAND_PATH.stat().st_size,
            **cand_score,
        },
        "cost_delta": cand_score["cost"] - base_score["cost"],
        "projected_score_gain": math.log(base_score["cost"] / cand_score["cost"]),
        "known_mask_equal_with_margin": masks_equal_with_margin(
            copy.deepcopy(base), copy.deepcopy(cand), 349, margin=0.25
        ),
        "candidate_margin": {"stable": margin_ok, "min_nonzero_abs": min_abs},
        "default_ort_known": default_ort_known(copy.deepcopy(cand)),
        "structure": structural_report(copy.deepcopy(cand)),
        "table_lengths": {
            init.name: int(numpy_helper.to_array(init).size)
            for init in cand.graph.initializer
            if init.name in TABLES
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    ok = (
        cand_score["cost"] < base_score["cost"]
        and report["known_mask_equal_with_margin"]
        and margin_ok
        and report["default_ort_known"] == {"right": report["default_ort_known"]["right"], "wrong": 0, "errors": 0}
        and not report["structure"]["bad_dims"]
        and report["structure"]["domains"] == [""]
        and report["structure"]["functions"] == 0
        and report["structure"]["inputs"] == 1
        and report["structure"]["outputs"] == 1
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
