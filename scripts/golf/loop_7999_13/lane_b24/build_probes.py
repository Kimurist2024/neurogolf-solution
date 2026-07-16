#!/usr/bin/env python3
"""Build conservative parameter/memory probes for task256 and task257.

These probes retain the incumbent terminal contractions and never add an
Einsum operand.  They test whether two apparently explicit basis/mask tensors
are semantically redundant on the true generator domain.  They are candidates
for screening, not presumed winners.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_SHA256 = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"


def save(
    model: onnx.ModelProto, task: int, name: str, description: str
) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = HERE / name
    onnx.save(model, path)
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "description": description,
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "params": sum(
            int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
            for initializer in model.graph.initializer
        ),
    }


def task256_p_broadcast(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    if [node.op_type for node in model.graph.node] != ["Einsum", "Concat", "Einsum"]:
        raise RuntimeError("unexpected task256 graph")
    final = model.graph.node[2]
    for index, name in enumerate(final.input):
        if name == "XP":
            final.input[index] = "P"
    del model.graph.node[1]
    kept = [initializer for initializer in model.graph.initializer if initializer.name != "one"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    kept_vi = [value for value in model.graph.value_info if value.name != "XP"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    return model


def task256_one_broadcast(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    final = copy.deepcopy(model.graph.node[2])
    for index, name in enumerate(final.input):
        if name == "XP":
            final.input[index] = "one"
    del model.graph.node[:]
    model.graph.node.extend([final])
    del model.graph.value_info[:]
    return model


def task257_mask_broadcast(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    for initializer in model.graph.initializer:
        if initializer.name == "mask":
            replacement = numpy_helper.from_array(np.ones((1,), dtype=np.float32), "mask")
            initializer.CopyFrom(replacement)
            break
    else:
        raise RuntimeError("task257 mask initializer missing")
    return model


def singleton_initializer_axis(
    base: onnx.ModelProto, initializer_name: str, axis: int, index: int
) -> onnx.ModelProto:
    """Keep one basis slice and rely only on legal Einsum singleton broadcast."""
    model = copy.deepcopy(base)
    for initializer in model.graph.initializer:
        if initializer.name != initializer_name:
            continue
        array = np.asarray(numpy_helper.to_array(initializer))
        if array.shape[axis] != 2:
            raise ValueError(f"{initializer_name} axis {axis} is not binary")
        replacement = numpy_helper.from_array(
            np.take(array, [index], axis=axis), initializer_name
        )
        initializer.CopyFrom(replacement)
        return model
    raise RuntimeError(f"initializer missing: {initializer_name}")


def main() -> int:
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"baseline SHA mismatch: {digest}")
    with zipfile.ZipFile(BASELINE) as archive:
        base256 = onnx.load_model_from_string(archive.read("task256.onnx"))
        base257 = onnx.load_model_from_string(archive.read("task257.onnx"))
    rows = [
        save(
            task256_p_broadcast(base256),
            256,
            "task256_p_broadcast.onnx",
            "remove Concat/one and broadcast the dynamic scalar P over each old XP axis",
        ),
        save(
            task256_one_broadcast(base256),
            256,
            "task256_one_broadcast.onnx",
            "remove the P/Concat stage and broadcast constant one over each old XP axis",
        ),
        save(
            task257_mask_broadcast(base257),
            257,
            "task257_mask_broadcast.onnx",
            "replace the 30-element output-domain mask by a broadcast scalar one",
        ),
    ]
    axes256 = {
        "B": (0,),
        "M": (0, 1, 2),
        "QX": (0,),
        "C": (0, 1, 2, 3),
        "S_XE": (0,),
    }
    axes257 = {"feat": (0,), "proj_a": (0, 1), "color": (0,)}
    for initializer_name, axes in axes256.items():
        for axis in axes:
            for index in (0, 1):
                rows.append(
                    save(
                        singleton_initializer_axis(base256, initializer_name, axis, index),
                        256,
                        f"task256_{initializer_name.lower()}_axis{axis}_keep{index}.onnx",
                        f"singleton-broadcast {initializer_name} axis {axis}, retaining basis slice {index}",
                    )
                )
    for initializer_name, axes in axes257.items():
        for axis in axes:
            for index in (0, 1):
                rows.append(
                    save(
                        singleton_initializer_axis(base257, initializer_name, axis, index),
                        257,
                        f"task257_{initializer_name.lower()}_axis{axis}_keep{index}.onnx",
                        f"singleton-broadcast {initializer_name} axis {axis}, retaining basis slice {index}",
                    )
                )
    payload = {"baseline_sha256": digest, "rows": rows}
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
