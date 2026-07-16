#!/usr/bin/env python3
"""Build task226 cost-370 by comparing the exact 0/1 row probes directly."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "submission_base_8009.46.zip"
BASELINE = HERE / "baseline/task226.onnx"
CANDIDATE = HERE / "candidates/task226_greater_cost370.onnx"
EXPECTED_ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_MEMBER_SHA256 = "852b6091385d97df6899e21304bf194440fb5cd3343385693093c24be0cb8203"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    assert sha256(ARCHIVE.read_bytes()) == EXPECTED_ARCHIVE_SHA256
    with zipfile.ZipFile(ARCHIVE) as archive:
        authority_bytes = archive.read("task226.onnx")
    assert sha256(authority_bytes) == EXPECTED_MEMBER_SHA256
    BASELINE.write_bytes(authority_bytes)
    model = onnx.load_model_from_string(authority_bytes)

    rewritten = []
    for node in model.graph.node:
        output = node.output[0]
        if output in {"nr8", "nr1"}:
            continue
        if output == "r3_and_nr8":
            # Official/generator carriers are one-hot, hence the gathered
            # background-channel values are exactly 0.0 or 1.0.  Therefore
            # bool(r3_f) AND NOT bool(r8_f) == (r3_f > r8_f).
            rewritten.append(helper.make_node("Greater", ["r3_f", "r8_f"], [output]))
            continue
        if output == "r6_and_nr1":
            rewritten.append(helper.make_node("Greater", ["r6_f", "r1_f"], [output]))
            continue
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.producer_name = "codex-task226-regolf187-greater"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, CANDIDATE)

    result = {
        "authority": str(BASELINE.relative_to(ROOT)),
        "authority_sha256": sha256(BASELINE.read_bytes()),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": sha256(CANDIDATE.read_bytes()),
        "rewrite": {
            "removed": ["Not(nr8)", "And(r3_and_nr8)", "Not(nr1)", "And(r6_and_nr1)"],
            "added": ["Greater(r3_f,r8_f)", "Greater(r6_f,r1_f)"],
            "domain_identity": "For a,b in {0.0,1.0}: bool(a) AND NOT bool(b) == (a > b).",
        },
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
    }
    (HERE / "audit/build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
