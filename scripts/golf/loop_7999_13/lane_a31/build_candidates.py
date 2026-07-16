#!/usr/bin/env python3
"""Build bounded task273/task306 probes from the pinned Wave16 members."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE_ZIP = HERE.parent / "submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_ZIP_SHA = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"


def _save_member(task: int) -> onnx.ModelProto:
    assert hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest() == EXPECTED_ZIP_SHA
    with zipfile.ZipFile(BASE_ZIP) as archive:
        payload = archive.read(f"task{task:03d}.onnx")
    path = HERE / f"task{task:03d}_base.onnx"
    path.write_bytes(payload)
    return onnx.load_model_from_string(payload)


def _replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array.copy(), name=name)
            )
            return
    raise KeyError(name)


def _save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, HERE / name)


def task306_state_slices(base: onnx.ModelProto) -> None:
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    for keep in ((0, 1), (0, 2), (1, 2)):
        model = onnx.ModelProto()
        model.CopyFrom(base)
        for name in ("Dp0", "Dp1"):
            _replace(model, name, arrays[name][list(keep), :])
        _save(model, f"task306_state_{keep[0]}{keep[1]}.onnx")


def task306_bond_slices(base: onnx.ModelProto) -> None:
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    for keep in ((0, 1), (0, 2), (1, 2)):
        model = onnx.ModelProto()
        model.CopyFrom(base)
        for name in ("Dp0", "Dp1"):
            _replace(model, name, arrays[name][:, list(keep)])
        _replace(model, "X", arrays["X"][list(keep), :])
        _replace(model, "S", arrays["S"][list(keep)])
        _save(model, f"task306_bond_{keep[0]}{keep[1]}.onnx")


def task306_single_state(base: onnx.ModelProto) -> None:
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    for keep in range(3):
        model = onnx.ModelProto()
        model.CopyFrom(base)
        for name in ("Dp0", "Dp1"):
            _replace(model, name, arrays[name][keep : keep + 1, :])
        _save(model, f"task306_state_{keep}.onnx")


def task306_single_bond(base: onnx.ModelProto) -> None:
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    for keep in range(3):
        model = onnx.ModelProto()
        model.CopyFrom(base)
        for name in ("Dp0", "Dp1"):
            _replace(model, name, arrays[name][:, keep : keep + 1])
        _replace(model, "X", arrays["X"][keep : keep + 1, :])
        _replace(model, "S", arrays["S"][keep : keep + 1])
        _save(model, f"task306_bond_{keep}.onnx")


def task306_reuse_diagonal_for_s(base: onnx.ModelProto, source: str) -> None:
    """Gauge columns so diag(source)==S, then delete S and use that diagonal.

    Scaling both Dp columns by c and the corresponding X rows by 1/c leaves
    every Dp-X contraction exactly unchanged.  Both matrices have nonzero
    diagonals, so either can be made equal to S on its diagonal.
    """
    model = onnx.ModelProto()
    model.CopyFrom(base)
    arrays = {item.name: numpy_helper.to_array(item) for item in base.graph.initializer}
    s = arrays["S"]
    diagonal = np.diag(arrays[source])
    scale = s / diagonal
    assert np.all(np.isfinite(scale)) and np.all(scale != 0)
    for name in ("Dp0", "Dp1"):
        _replace(model, name, arrays[name] * scale[None, :])
    _replace(model, "X", arrays["X"] / scale[:, None])
    transformed = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    assert np.array_equal(np.diag(transformed[source]), s)

    node = model.graph.node[0]
    attribute = next(item for item in node.attribute if item.name == "equation")
    terms = attribute.s.decode("ascii").split("->")[0].split(",")
    rhs = attribute.s.decode("ascii").split("->")[1]
    for index, name in enumerate(node.input):
        if name == "S":
            label = terms[index]
            assert len(label) == 1
            node.input[index] = source
            terms[index] = label * 2
    attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")
    kept = [item for item in model.graph.initializer if item.name != "S"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    _save(model, f"task306_reuse_{source.lower()}_diag_for_s.onnx")


def task306_share_dp(base: onnx.ModelProto, retained: str) -> None:
    model = onnx.ModelProto()
    model.CopyFrom(base)
    removed = "Dp1" if retained == "Dp0" else "Dp0"
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == removed:
                node.input[index] = retained
    kept = [item for item in model.graph.initializer if item.name != removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    _save(model, f"task306_share_{retained.lower()}.onnx")


def main() -> None:
    _save_member(273)
    task306 = _save_member(306)
    task306_state_slices(task306)
    task306_bond_slices(task306)
    task306_single_state(task306)
    task306_single_bond(task306)
    task306_reuse_diagonal_for_s(task306, "Dp0")
    task306_reuse_diagonal_for_s(task306, "Dp1")
    task306_share_dp(task306, "Dp0")
    task306_share_dp(task306, "Dp1")


if __name__ == "__main__":
    main()
