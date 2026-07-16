#!/usr/bin/env python3
"""Build strictly smaller, non-giant latent-axis pruning probes for task232."""

from __future__ import annotations

import copy
import hashlib
import itertools
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


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(array, name))
            return
    raise RuntimeError(f"missing initializer: {name}")


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def latent_probe(
    base: onnx.ModelProto,
    r_keep: tuple[int, ...],
    binary_keep: tuple[int, int, int] | None,
) -> onnx.ModelProto:
    """Prune shared r and optionally all binary contraction dimensions.

    The single V initializer occurs as uvr/wzr/abr, F as xu/xv/pw/pz,
    and G as or/cr.  Reducing r therefore requires matching V axis 2 and G
    axis 1.  Reducing binary labels requires singleton V axes 0/1 and F
    axis 1; the three retained indices independently choose the stored V/F
    slices while preserving legal dimension-one contractions.
    """
    model = copy.deepcopy(base)
    data = arrays(model)
    v = np.take(data["V"], r_keep, axis=2)
    g = np.take(data["G"], r_keep, axis=1)
    f = data["F"]
    if binary_keep is not None:
        v0, v1, f1 = binary_keep
        v = np.take(np.take(v, [v0], axis=0), [v1], axis=1)
        f = np.take(f, [f1], axis=1)
    replace(model, "V", v)
    replace(model, "G", g)
    replace(model, "F", f)
    return model


def save(model: onnx.ModelProto, name: str, description: str) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = HERE / name
    onnx.save(model, path)
    params = sum(
        int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1
        for item in model.graph.initializer
    )
    return {
        "task": 232,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "description": description,
        "params": params,
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    }


def main() -> int:
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"baseline SHA mismatch: {digest}")
    with zipfile.ZipFile(BASELINE) as archive:
        base = onnx.load_model_from_string(archive.read("task232.onnx"))
    if [node.op_type for node in base.graph.node] != ["Einsum"]:
        raise RuntimeError("unexpected task232 graph")
    node = base.graph.node[0]
    equation = next(attr.s.decode() for attr in node.attribute if attr.name == "equation")
    if equation != "uvr,wzr,abr,xu,xv,pw,pz,ndhx,nchp,or,cr->nohx":
        raise RuntimeError(f"unexpected task232 equation: {equation}")

    rows: list[dict[str, object]] = []
    for width in (1, 2, 3):
        for keep in itertools.combinations(range(4), width):
            rows.append(
                save(
                    latent_probe(base, keep, None),
                    f"task232_r_{''.join(map(str, keep))}.onnx",
                    f"retain shared r components {keep} in both V and G",
                )
            )
    for v0, v1, f1 in itertools.product(range(2), repeat=3):
        rows.append(
            save(
                latent_probe(base, (0, 1, 2, 3), (v0, v1, f1)),
                f"task232_bin_v{v0}{v1}_f{f1}.onnx",
                f"singleton all binary contractions using V[{v0},{v1},:] and F[:,{f1}]",
            )
        )
        for width in (1, 2, 3):
            for keep in itertools.combinations(range(4), width):
                rows.append(
                    save(
                        latent_probe(base, keep, (v0, v1, f1)),
                        f"task232_bin_v{v0}{v1}_f{f1}_r_{''.join(map(str, keep))}.onnx",
                        f"singleton binary contractions and retain shared r components {keep}",
                    )
                )
    payload = {"baseline_sha256": digest, "candidate_count": len(rows), "rows": rows}
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "params": sorted({r['params'] for r in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
