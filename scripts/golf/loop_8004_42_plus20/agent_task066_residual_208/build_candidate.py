#!/usr/bin/env python3
"""Build the task066 residual exact-support regolf from staged cost-561 parent."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/task066.onnx"
OUTPUT = HERE / "task066_residual_cost551.onnx"
REJECTED = HERE / "REJECTED_DO_NOT_MERGE"
EXPECTED_SOURCE_SHA = "2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    source_data = SOURCE.read_bytes()
    if sha256(source_data) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("staged task066 parent changed")
    model = onnx.load_model_from_string(source_data)

    # The opaque coordinate/color algebra was independently contracted in
    # review206.  Reconstruct the green selector e3 from existing initializer
    # rows, eliminating greenhalf10 without adding a runtime tensor:
    #   (U[0] * V[2]) * dot(U[2], V[2]) = (-e3) * (-1) = e3.
    # Trow/Tcol/z1 contractions select rows 0 and 2 inside the same Einsum.
    green_inputs = [
        "Uchan",
        "Vchan",
        "Trow",
        "Tcol",
        "Tcol",
        "z1",
        "Uchan",
        "Vchan",
        "Tcol",
        "z1",
    ]
    for output, second_term, power_label in (("Gv", "bdrw", "h"), ("Gh", "bdhr", "w")):
        node = next(item for item in model.graph.node if list(item.output) == [output])
        expected_inputs = ["input", "input", "Uchan", "Trow", "z1", "greenhalf10", "pow2"]
        if node.op_type != "Einsum" or list(node.input) != expected_inputs:
            raise RuntimeError(f"unexpected {output} parent")
        del node.input[:]
        node.input.extend(["input", "input", "Uchan", "Trow", "z1", *green_inputs, "pow2"])
        equation = (
            f"bchw,{second_term},qc,qz,z,ad,ed,aj,aj,ek,k,fl,fl,fm,m,{power_label}->b"
        )
        attr = next(item for item in node.attribute if item.name == "equation")
        attr.s = equation.encode("ascii")

    initializers = list(model.graph.initializer)
    if [item.name for item in initializers].count("greenhalf10") != 1:
        raise RuntimeError("greenhalf10 initializer mismatch")
    del model.graph.initializer[:]
    model.graph.initializer.extend(item for item in initializers if item.name != "greenhalf10")

    onnx.checker.check_model(model, full_check=True)
    OUTPUT.write_bytes(model.SerializeToString())

    # Probe the ONNX schema's advertised smaller numeric index types.  These
    # are lane-only experiments; audit code must require an actual ORT kernel.
    probe_types = {
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
        "int16": TensorProto.INT16,
        "uint16": TensorProto.UINT16,
        "float16": TensorProto.FLOAT16,
        "uint32": TensorProto.UINT32,
        "float32": TensorProto.FLOAT,
        "int64": TensorProto.INT64,
    }
    REJECTED.mkdir(parents=True, exist_ok=True)
    for label, dtype in probe_types.items():
        probe = copy.deepcopy(model)
        cast = next(item for item in probe.graph.node if list(item.output) == ["cOut64"])
        attr = next(item for item in cast.attribute if item.name == "to")
        attr.i = dtype
        path = REJECTED / f"task066_residual_cast_{label}.onnx"
        onnx.checker.check_model(probe, full_check=True)
        path.write_bytes(probe.SerializeToString())
    print(f"source_sha256={sha256(source_data)}")
    print(f"candidate_sha256={sha256(OUTPUT.read_bytes())}")
    print(f"nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")


if __name__ == "__main__":
    main()
