#!/usr/bin/env python3
"""Build the lane SOUND artifacts without touching repository incumbents."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"

SOURCE_344 = ROOT / "scripts/golf/scratch_codex/task344/candidate_rank4_boundary.onnx"
SOURCE_344_SHA256 = "159b06e536ec43c97aa34ff1e1d51211ebafcde5630d793d8dd6904984c49e50"
SOURCE_168 = ROOT / "others/7903/task168_improved.onnx"
SOURCE_168_SHA256 = "dcf6a0cc845c4363197195dcf72e64f89e45116eacdc08ac767cfc0076f845f4"
SOURCE_192 = ROOT / "scripts/golf/scratch_codex_plus10/wave2_sound/task192_sound_bitset.onnx"
SOURCE_192_SHA256 = "16f59d172be152d14e087e54085d6ef2cb6ee188528e70e9549c1a3fac391193"

ACTIVE_INPUT_COLORS = (0, 2, 3, 5)
NEW_EQUATION_344 = (
    "sl,ld,bdpq,ap,ah,ep,eh,fp,fh,gp,gh,iq,iw,jq,jw,mq,mw,nq,nw,"
    "bchw,kc,sx,xk,ky,yo->bohw"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_baselines() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(BASE) as archive:
        for task in (168, 192, 344):
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            output = HERE / "baseline" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            result[str(task)] = {
                "path": str(output.relative_to(ROOT)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    return result


def build_task344() -> dict[str, object]:
    if digest(SOURCE_344) != SOURCE_344_SHA256:
        raise RuntimeError("task344 rank-4 source hash changed")
    model = onnx.load(SOURCE_344)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("unexpected task344 source graph")
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }
    expected = {"H": (3, 4), "V": (4, 10), "B": (4, 30), "S": (3, 3), "U": (4, 10)}
    if {name: value.shape for name, value in arrays.items()} != expected:
        raise RuntimeError("unexpected task344 source tensors")

    active = list(ACTIVE_INPUT_COLORS)
    inactive = [color for color in range(10) if color not in active]
    v_active = arrays["V"][:, active]
    transform = arrays["U"][:, active] @ np.linalg.inv(v_active)
    new_v = arrays["V"].copy()
    new_v[:, inactive] = np.linalg.solve(transform, arrays["U"][:, inactive])
    transform32 = transform.astype(np.float32)
    new_v32 = new_v.astype(np.float32)
    max_error = float(np.max(np.abs(transform32 @ new_v32 - arrays["U"].astype(np.float32))))
    if max_error > 2e-4:
        raise RuntimeError(f"task344 factor reconstruction error {max_error}")
    if not np.array_equal(new_v32[:, active], arrays["V"][:, active].astype(np.float32)):
        raise RuntimeError("task344 active input columns changed")

    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "V":
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(new_v32, "V"))
        elif initializer.name == "U":
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(transform32, "M"))
    node = model.graph.node[0]
    if list(node.input[-5:]) != ["input", "V", "S", "H", "U"]:
        raise RuntimeError("unexpected task344 Einsum operands")
    node.input[-1] = "M"
    node.input.append("V")
    next(attribute for attribute in node.attribute if attribute.name == "equation").s = (
        NEW_EQUATION_344.encode("ascii")
    )
    model.graph.name = "task344_sound_rank4_reuse_v"
    model.producer_name = "codex-sound-lane-7999.13"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    params = sum(int(np.prod(item.dims)) for item in model.graph.initializer)
    if params != 197:
        raise RuntimeError(f"task344 expected cost/params 197, got {params}")
    output = HERE / "task344_sound_cost197.onnx"
    onnx.save(model, output)
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": digest(output),
        "params": params,
        "factor_reconstruction_max_error": max_error,
    }


def retain_verified_source(source: Path, expected_sha: str, output_name: str) -> dict[str, object]:
    actual = digest(source)
    if actual != expected_sha:
        raise RuntimeError(f"source hash changed for {source}: {actual}")
    output = HERE / output_name
    output.write_bytes(source.read_bytes())
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": digest(output),
        "source": str(source.relative_to(ROOT)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(BASE) != BASE_SHA256:
        raise RuntimeError("7999.13 baseline hash changed")
    manifest = {
        "baseline": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": BASE_SHA256,
            "members": extract_baselines(),
        },
        "built": {
            "task344": build_task344(),
            "task168": retain_verified_source(
                SOURCE_168, SOURCE_168_SHA256, "task168_sound_cost416.onnx"
            ),
            "task192_control": retain_verified_source(
                SOURCE_192, SOURCE_192_SHA256, "task192_sound_control_cost3325.onnx"
            ),
        },
    }
    (HERE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
