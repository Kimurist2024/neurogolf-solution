#!/usr/bin/env python3
"""Build algebraic, non-expanding Wave17 candidates for task328/task358."""

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
BASELINE_ZIP = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"
EXPECTED_ZIP_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"
EXPECTED_MEMBERS = {
    328: "08ba1aa525d67f290c13e7b79aef339aeb5912bf0d1b0b379ff6ab8792cf576a",
    358: "3ea6a4be62e0ee5e50ae94c1e89db81e2eb12f98b97317e28f0fb27ff81105a9",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def einsum_equation(node: onnx.NodeProto) -> tuple[onnx.AttributeProto, list[str], str]:
    attr = next(item for item in node.attribute if item.name == "equation")
    lhs, rhs = attr.s.decode("ascii").split("->")
    return attr, lhs.split(","), rhs


def max_einsum(model: onnx.ModelProto) -> int:
    return max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)


def params(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1 for item in model.graph.initializer)


def save(model: onnx.ModelProto, task: int, name: str, proof: dict[str, object]) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    path = HERE / name
    onnx.save(model, path)
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path.read_bytes()),
        "serialized_bytes": path.stat().st_size,
        "initializer_params": params(model),
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max_einsum(model),
        "proof": proof,
    }


def task328_reuse_j_diagonal(base: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    """Replace e=[1,0,0,0] by the exact diagonal sum of existing J.

    J[0] has only J[0,0,0]=1 on its diagonal and J[1] has a zero
    diagonal, hence sum_t J[t,c,c] == e[c].  Each of the four uses gets
    its own private summation label, so no contraction is coupled.
    """
    model = copy.deepcopy(base)
    node = model.graph.node[-1]
    if node.op_type != "Einsum" or len(node.input) != 58:
        raise RuntimeError("unexpected task328 terminal Einsum")
    attr, subs, rhs = einsum_equation(node)
    expected = [(44, "C", "tCC"), (45, "F", "PFF"), (56, "I", "RII"), (57, "L", "SLL")]
    for index, old_sub, new_sub in expected:
        if node.input[index] != "e" or subs[index] != old_sub:
            raise RuntimeError(f"unexpected e operand at {index}: {node.input[index]} {subs[index]}")
        node.input[index] = "J"
        subs[index] = new_sub
    attr.s = (",".join(subs) + "->" + rhs).encode("ascii")

    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    derived = np.einsum("taa->a", arrays["J"])
    if not np.array_equal(derived, arrays["e"]):
        raise RuntimeError(f"J diagonal identity failed: {derived} != {arrays['e']}")
    kept = [item for item in model.graph.initializer if item.name != "e"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model, {
        "identity": "e[a] = sum_t J[t,a,a]",
        "exact_float32": True,
        "removed_initializers": ["e"],
        "added_initializers": [],
        "baseline_max_einsum_inputs": 58,
        "candidate_max_einsum_inputs": 58,
    }


def task358_combine_r2_r3(base: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    """Collapse the exact (x-2)(x+2)=x^2-4 factor pair.

    For l=0 the selected coordinate feature is F[:,1], and for l=1 it
    is F[:,2].  R2 and R3 contribute (x-2) and (x+2), respectively.  Their
    product therefore equals a single diagonal contraction with
    R23=[[-4,1,0],[-4,0,1]].  The rewrite is applied independently to the
    row and column polynomial groups, reducing each six-operand block to five.
    """
    model = copy.deepcopy(base)
    node = model.graph.node[-1]
    if node.op_type != "Einsum" or len(node.input) != 44:
        raise RuntimeError("unexpected task358 terminal Einsum")
    attr, subs, rhs = einsum_equation(node)
    pairs = list(zip(node.input, subs))
    row_old = [("F", "rb"), ("F", "hb"), ("R2", "lb"), ("F", "rd"), ("F", "hd"), ("R3", "ld")]
    row_new = [("F", "rb"), ("F", "hb"), ("F", "rb"), ("F", "hb"), ("R23", "lb")]
    col_old = [("F", "cp"), ("F", "wp"), ("R2", "lp"), ("F", "cz"), ("F", "wz"), ("R3", "lz")]
    col_new = [("F", "cp"), ("F", "wp"), ("F", "cp"), ("F", "wp"), ("R23", "lp")]

    def replace_once(items: list[tuple[str, str]], old: list[tuple[str, str]], new: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for start in range(len(items) - len(old) + 1):
            if items[start : start + len(old)] == old:
                return items[:start] + new + items[start + len(old) :]
        raise RuntimeError(f"task358 factor block not found: {old}")

    pairs = replace_once(pairs, row_old, row_new)
    pairs = replace_once(pairs, col_old, col_new)
    del node.input[:]
    node.input.extend(name for name, _ in pairs)
    attr.s = (",".join(sub for _, sub in pairs) + "->" + rhs).encode("ascii")

    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    r23 = np.array([[-4.0, 1.0, 0.0], [-4.0, 0.0, 1.0]], dtype=np.float32)
    # Exhaust all F coordinate pairs and both l values to prove the local
    # factor identity in the exact dtype used by the model.
    for l in range(2):
        for row in range(30):
            for col in range(30):
                old = np.dot(arrays["F"][row] * arrays["F"][col], arrays["R2"][l])
                old *= np.dot(arrays["F"][row] * arrays["F"][col], arrays["R3"][l])
                new = np.dot((arrays["F"][row] * arrays["F"][col]) ** 2, r23[l])
                if old != new:
                    raise RuntimeError((l, row, col, old, new))

    kept = [item for item in model.graph.initializer if item.name not in {"R2", "R3"}]
    kept.append(numpy_helper.from_array(r23, name="R23"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model, {
        "identity": "(x-2)*(x+2) = x^2-4 for both selected coordinate features",
        "exhaustive_local_float32_checks": 1800,
        "removed_initializers": ["R2", "R3"],
        "added_initializers": ["R23"],
        "baseline_max_einsum_inputs": 44,
        "candidate_max_einsum_inputs": 42,
    }


def main() -> int:
    zip_digest = digest(BASELINE_ZIP.read_bytes())
    if zip_digest != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"Wave17 SHA mismatch: {zip_digest}")
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        bases: dict[int, onnx.ModelProto] = {}
        baseline_rows: list[dict[str, object]] = []
        for task in (328, 358):
            payload = archive.read(f"task{task:03d}.onnx")
            member_digest = digest(payload)
            if member_digest != EXPECTED_MEMBERS[task]:
                raise RuntimeError(f"task{task} SHA mismatch: {member_digest}")
            path = HERE / f"baseline_task{task:03d}.onnx"
            path.write_bytes(payload)
            model = onnx.load_model_from_string(payload)
            bases[task] = model
            baseline_rows.append({
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": member_digest,
                "initializer_params": params(model),
                "nodes": len(model.graph.node),
                "max_einsum_inputs": max_einsum(model),
            })

    candidate328, proof328 = task328_reuse_j_diagonal(bases[328])
    candidate358, proof358 = task358_combine_r2_r3(bases[358])
    candidates = [
        save(candidate328, 328, "task328_reuse_j_diagonal.onnx", proof328),
        save(candidate358, 358, "task358_combine_r2_r3.onnx", proof358),
    ]
    payload = {
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": zip_digest,
        "baselines": baseline_rows,
        "candidates": candidates,
    }
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
