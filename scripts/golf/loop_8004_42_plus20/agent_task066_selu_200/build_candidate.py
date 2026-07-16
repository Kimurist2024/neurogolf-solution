#!/usr/bin/env python3
"""Build the task066 Div->Selu candidate from immutable 8009.46 authority."""

from __future__ import annotations

import copy
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
OUTPUT = HERE / "task066_selu_cost561.onnx"
EXPECTED_ZIP_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_MEMBER_SHA = "bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    zip_data = AUTHORITY_ZIP.read_bytes()
    if sha256(zip_data) != EXPECTED_ZIP_SHA:
        raise RuntimeError("immutable authority ZIP hash changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        base_data = archive.read("task066.onnx")
    if sha256(base_data) != EXPECTED_MEMBER_SHA:
        raise RuntimeError("immutable task066 authority member hash changed")

    base = onnx.load_model_from_string(base_data)
    candidate = copy.deepcopy(base)
    div_nodes = [node for node in candidate.graph.node if list(node.output) == ["selQ"]]
    if len(div_nodes) != 1 or div_nodes[0].op_type != "Div" or list(div_nodes[0].input) != ["selLog", "ln2"]:
        raise RuntimeError("unexpected authority selQ producer")

    # ln2 is stored as float16 in the authority.  ORT's float16 Selu kernel
    # takes a float attribute; use the float32 reciprocal of that exact stored
    # value.  The audit proves equivalence after the immediately following
    # uint8 Cast over the entire reachable integer bound.
    ln2 = next(item for item in candidate.graph.initializer if item.name == "ln2")
    stored_ln2 = float(onnx.numpy_helper.to_array(ln2))
    gamma = float(np.float32(1.0 / stored_ln2))
    node = div_nodes[0]
    node.op_type = "Selu"
    del node.input[:]
    node.input.extend(["selLog"])
    del node.attribute[:]
    node.attribute.extend(
        [helper.make_attribute("alpha", 1.0), helper.make_attribute("gamma", gamma)]
    )

    initializers = list(candidate.graph.initializer)
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(item for item in initializers if item.name != "ln2")
    onnx.checker.check_model(candidate, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUTPUT)
    print(f"authority_member_sha256={sha256(base_data)}")
    print(f"stored_ln2={stored_ln2!r}")
    print(f"gamma_float32={gamma!r}")
    print(f"candidate_sha256={sha256(OUTPUT.read_bytes())}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
