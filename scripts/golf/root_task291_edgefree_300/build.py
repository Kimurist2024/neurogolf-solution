#!/usr/bin/env python3
"""Replace task291's two 30-element edge vectors with size-one batch axes."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.lib import scoring  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8011.05.zip"
TASK = 291


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        baseline = onnx.load_model_from_string(archive.read("task291.onnx"))

    candidate = onnx.ModelProto()
    candidate.CopyFrom(baseline)
    node = candidate.graph.node[0]
    assert node.op_type == "Einsum"
    assert list(node.input) == [
        "input", "input", "input", "sign", "sign", "sign", "edge", "edge"
    ]
    del node.input[:]
    node.input.extend(["input", "input", "input", "sign", "sign", "sign", "input", "input"])
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    equation.s = b"abrd,abec,afrc,b,f,g,xhij,yklm->abxy"
    kept = [tensor for tensor in candidate.graph.initializer if tensor.name != "edge"]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)

    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        candidate, strict_mode=True, data_prop=True
    )
    out_dims = [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim]
    if out_dims != [1, 10, 1, 1]:
        raise RuntimeError(f"unexpected inferred output shape: {out_dims}")

    path = HERE / "task291.onnx"
    onnx.save(candidate, path)
    base_profile = scoring.score_and_verify(
        baseline, TASK, str(HERE / "profiles"), "baseline", require_correct=True
    )
    cand_profile = scoring.score_and_verify(
        candidate, TASK, str(HERE / "profiles"), "candidate", require_correct=True
    )
    margin_ok, margin_min = scoring.model_margin_stable(candidate, TASK)
    result = {
        "authority_zip": BASE.name,
        "authority_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "candidate": str(path.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_profile": base_profile,
        "candidate_profile": cand_profile,
        "checker_full": True,
        "strict_data_prop": True,
        "inferred_output_shape": out_dims,
        "known_correct": bool(cand_profile and cand_profile["correct"]),
        "margin_stable": margin_ok,
        "margin_min": margin_min,
    }
    (HERE / "build_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
