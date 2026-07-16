#!/usr/bin/env python3
"""Build isolated exact-base variants for A29 tasks 275 and 308."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import onnx
import numpy as np
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WAVE14 = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave14_candidate_meta.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(model: onnx.ModelProto, name: str) -> dict[str, object]:
    path = HERE / name
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def main() -> None:
    manifest: dict[str, object] = {"wave14_sha256": sha(WAVE14), "models": {}}
    with zipfile.ZipFile(WAVE14) as zf:
        bases = {task: onnx.load_from_string(zf.read(f"task{task:03d}.onnx")) for task in (275, 308)}

    for task, model in bases.items():
        manifest["models"][f"task{task:03d}_base"] = save(model, f"task{task:03d}_base.onnx")

    base = bases[275]
    final = base.graph.node[-1]
    equation = next(attr for attr in final.attribute if attr.name == "equation")
    text = equation.s.decode("ascii")

    # Share the two independent 3x3 latent maps without changing graph shape or
    # operand count.  Both orientations are screened; these are semantic probes,
    # not retained unless complete differential validation passes.
    for source, removed, replace_index in (("T", "W", 40), ("W", "T", 14)):
        for transpose in (False, True):
            candidate = copy.deepcopy(base)
            node = candidate.graph.node[-1]
            node.input[replace_index] = source
            remove_initializer(candidate, removed)
            if transpose:
                attr = next(item for item in node.attribute if item.name == "equation")
                equation_text = text
                parts, output = equation_text.split("->")
                terms = parts.split(",")
                terms[replace_index] = terms[replace_index][::-1]
                attr.s = (",".join(terms) + "->" + output).encode("ascii")
            name = f"task275_share_{removed}_from_{source}_{'transpose' if transpose else 'direct'}.onnx"
            manifest["models"][name.removesuffix(".onnx")] = save(candidate, name)

    # The size gate only ever sees total=18 (N=3) or total=32 (N=4).  The base
    # realizes its 2x2 router as
    #   (total-25) * [1,-1] outer [-1,1] + [1,1] outer [7,7].
    # An exactly equal router on both generator-reachable totals is
    #   (25-total) * [1,-1] outer [1,-1] + 7 * [1,1] outer [1,1].
    # Therefore GU can serve both latent factors and GV's four parameters can
    # be deleted, while graph shape, operand count, and runtime tensor sizes stay
    # unchanged.
    candidate = copy.deepcopy(base)
    arrays = {init.name: init for init in candidate.graph.initializer}
    arrays["GW"].CopyFrom(numpy_helper.from_array(np.array([[[[-1.0]]], [[[0.0]]]], dtype=np.float32), "GW"))
    arrays["GB"].CopyFrom(numpy_helper.from_array(np.array([25.0, 7.0], dtype=np.float32), "GB"))
    for node in candidate.graph.node:
        for index, value in enumerate(node.input):
            if value == "GV":
                node.input[index] = "GU"
    remove_initializer(candidate, "GV")
    manifest["models"]["task275_shared_gate_router"] = save(
        candidate, "task275_shared_gate_router.onnx"
    )

    # Exact task308 probe: make the dtype-only CastLike use the existing fp16
    # coordinate initializer.  This alone does not remove four_f (it remains
    # required by the two variance contractions), but establishes that there is
    # no hidden parameter win from that anchor use.
    candidate = copy.deepcopy(bases[308])
    candidate.graph.node[3].input[1] = "idx30"
    manifest["models"]["task308_castlike_idx_anchor"] = save(
        candidate, "task308_castlike_idx_anchor.onnx"
    )

    # Shape(out_shape4_const) and Shape(out_shape4) are both the same rank
    # vector [4].  Reuse the first node output for TopK K and its two crop
    # targets, deleting the second Shape without changing the dynamic target
    # construction used by the incumbent.
    candidate = copy.deepcopy(bases[308])
    for node in candidate.graph.node:
        for index, value in enumerate(node.input):
            if value == "topk_k_dyn":
                node.input[index] = "out_shape4_len"
    kept_nodes = [node for node in candidate.graph.node if "topk_k_dyn" not in node.output]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)
    kept_vi = [item for item in candidate.graph.value_info if item.name != "topk_k_dyn"]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_vi)
    manifest["models"]["task308_reuse_rank_shape"] = save(
        candidate, "task308_reuse_rank_shape.onnx"
    )

    (HERE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
