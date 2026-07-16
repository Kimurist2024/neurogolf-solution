#!/usr/bin/env python3
"""Build semantics-preserving low-cost candidates from the 8011.05 authority."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib import scoring  # noqa: E402


BASE_ZIP = ROOT / "submission_base_8011.05.zip"
OUT = Path(__file__).resolve().parent


def load_member(task: int) -> onnx.ModelProto:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        return onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def remove_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def eliminate_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    kept = []
    for node in candidate.graph.node:
        if node.op_type == "Identity" and len(node.input) == len(node.output) == 1:
            replace_all_inputs(candidate, node.output[0], node.input[0])
        else:
            kept.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    remove_unused_initializers(candidate)
    return candidate


def fold_shrink_scalar(model: onnx.ModelProto) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    initializers = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in candidate.graph.initializer
    }
    kept = []
    additions = []
    for node in candidate.graph.node:
        if node.op_type != "Shrink" or node.input[0] not in initializers:
            kept.append(node)
            continue
        value = initializers[node.input[0]]
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        lambd = float(attrs.get("lambd", 0.5))
        bias = float(attrs.get("bias", 0.0))
        folded = np.where(value < -lambd, value + bias, np.where(value > lambd, value - bias, 0))
        folded = folded.astype(value.dtype, copy=False)
        additions.append(numpy_helper.from_array(folded, name=node.output[0]))
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    candidate.graph.initializer.extend(additions)
    remove_unused_initializers(candidate)
    return candidate


def fold_constant_of_shape(model: onnx.ModelProto) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    initializers = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in candidate.graph.initializer
    }
    kept = []
    additions = []
    for node in candidate.graph.node:
        if node.op_type != "ConstantOfShape" or node.input[0] not in initializers:
            kept.append(node)
            continue
        shape = tuple(int(x) for x in initializers[node.input[0]].reshape(-1))
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        fill_tensor = attrs.get("value")
        if fill_tensor is None:
            fill = np.asarray(0, dtype=np.float32)
        else:
            fill = numpy_helper.to_array(fill_tensor).reshape(-1)[0]
        folded = np.full(shape, fill, dtype=np.asarray(fill).dtype)
        additions.append(numpy_helper.from_array(folded, name=node.output[0]))
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    candidate.graph.initializer.extend(additions)
    remove_unused_initializers(candidate)
    return candidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        150: fold_shrink_scalar,
        155: fold_shrink_scalar,
        171: fold_constant_of_shape,
        262: eliminate_identity,
        269: eliminate_identity,
        289: eliminate_identity,
    }
    evidence = []
    for task, transform in jobs.items():
        baseline = load_member(task)
        candidate = transform(baseline)
        path = OUT / f"task{task:03d}.onnx"
        onnx.save(candidate, path)
        try:
            onnx.checker.check_model(candidate, full_check=True)
            checker = {"ok": True, "error": None}
        except Exception as exc:  # fail closed but retain the diagnostic artifact
            checker = {"ok": False, "error": str(exc)}
            evidence.append(
                {
                    "task": task,
                    "candidate": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "checker": checker,
                    "baseline_profile": None,
                    "candidate_profile": None,
                    "known_raw_bit_identical": False,
                    "strict_lower": False,
                }
            )
            print(json.dumps(evidence[-1], ensure_ascii=False))
            continue
        base_profile = scoring.score_and_verify(
            baseline, task, str(OUT / "profiles"), f"base{task}", require_correct=False
        )
        cand_profile = scoring.score_and_verify(
            candidate, task, str(OUT / "profiles"), f"cand{task}", require_correct=False
        )
        bit_identical = scoring.outputs_bit_identical(baseline, candidate, task)
        evidence.append(
            {
                "task": task,
                "candidate": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "checker": checker,
                "baseline_profile": base_profile,
                "candidate_profile": cand_profile,
                "known_raw_bit_identical": bit_identical,
                "strict_lower": bool(
                    base_profile
                    and cand_profile
                    and cand_profile["cost"] < base_profile["cost"]
                ),
            }
        )
        print(json.dumps(evidence[-1], ensure_ascii=False))
    (OUT / "screen.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
