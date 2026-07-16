#!/usr/bin/env python3
"""Apply the exact nonnegative Selu memshave to staged SOUND task158."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "others/71407/task158.onnx"
BASE_SHA256 = "127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd"
OUTPUT = HERE / "task158_extension/task158.onnx"
TASK = 158
SEEDS = (158_127_001, 158_127_002)
COUNT = 1000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def load_audit():
    path = HERE / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("selu127_audit_for158", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(base: onnx.ModelProto) -> onnx.ModelProto:
    result = copy.deepcopy(base)
    values = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in result.graph.initializer
    }
    gamma_array = values["coord_pack"]
    if gamma_array.size != 1 or gamma_array.dtype != np.float16 or float(gamma_array) != 32.0:
        raise RuntimeError("unexpected coord_pack")
    sources = {"p_row", "q_row"}
    replacements = {}
    uses = 0
    for index, node in enumerate(result.graph.node):
        if "coord_pack" not in node.input:
            continue
        uses += 1
        if node.op_type != "Mul" or len(node.input) != 2 or len(node.output) != 1:
            raise RuntimeError(f"unexpected coord_pack use: {node.op_type}")
        source = node.input[0] if node.input[1] == "coord_pack" else node.input[1]
        if source not in sources:
            raise RuntimeError(source)
        sources.remove(source)
        replacements[index] = helper.make_node(
            "Selu",
            [source],
            list(node.output),
            name=f"exact_nonnegative_{source}_pack32",
            alpha=1.0,
            gamma=32.0,
        )
    if uses != 2 or sources:
        raise RuntimeError(f"coord_pack use mismatch: {uses}, {sources}")
    nodes = [replacements.get(index, node) for index, node in enumerate(result.graph.node)]
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    initializers = [item for item in result.graph.initializer if item.name != "coord_pack"]
    if len(initializers) + 1 != len(result.graph.initializer):
        raise RuntimeError("coord_pack removal failed")
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(initializers)
    return result


def official(model: onnx.ModelProto) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="task158_selu_", dir="/tmp") as work:
        return scoring.score_and_verify(model, TASK, work, label="task158_selu", require_correct=True)


def main() -> int:
    if digest(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("staged task158 parent changed")
    base = onnx.load(BASE)
    candidate = build(base)
    onnx.checker.check_model(candidate, full_check=True)
    shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    findings = check_conv_bias(candidate)
    if findings:
        raise RuntimeError(findings)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUTPUT)
    base_bytes = BASE.read_bytes()
    candidate_bytes = OUTPUT.read_bytes()
    known_cases = AUDIT.known(TASK)
    known = {}
    for disable, threads, label in AUDIT.CONFIGS:
        known[label] = AUDIT.evaluate_cases(
            base_bytes, candidate_bytes, known_cases, disable, threads
        )
    fresh = []
    for seed in SEEDS:
        cases, attempts = AUDIT.generate(TASK, seed, COUNT)
        stream = {"seed": seed, "attempts": attempts, "modes": {}}
        for disable, label in ((True, "disable_all"), (False, "default")):
            stream["modes"][label] = AUDIT.evaluate_cases(
                base_bytes, candidate_bytes, cases, disable, 1
            )
        fresh.append(stream)
        print(f"fresh seed={seed} valid={len(cases)}", flush=True)
    report = {
        "task": TASK,
        "parent_sha256": BASE_SHA256,
        "candidate_sha256": digest(candidate_bytes),
        "nonnegative_proof": (
            "p_row and q_row gather anchor_rows=2*floor(nonnegative TopK index/15) "
            "+ Cast(boolean phase), hence are finite nonnegative float16 and never -0"
        ),
        "full_check": True,
        "strict_data_prop": True,
        "conv_bias_ub0": True,
        "runtime_shape_truth": AUDIT.runtime_shape_truth(TASK, candidate_bytes),
        "zero_profile_parent": dict(zip(("memory", "params", "cost"), cost_of(str(BASE)))),
        "zero_profile_candidate": dict(zip(("memory", "params", "cost"), cost_of(str(OUTPUT)))),
        "official": official(candidate),
        "known_four_configs": known,
        "fresh": fresh,
    }
    report["accepted"] = bool(
        report["runtime_shape_truth"].get("truthful", False)
        and report["official"]
        and report["official"]["cost"] == 7524
        and report["official"]["correct"]
        and all(item.get("exact_equivalent", False) for item in known.values())
        and all(
            mode.get("exact_equivalent", False)
            for stream in fresh
            for mode in stream["modes"].values()
        )
    )
    (HERE / "task158_extension/audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": report["accepted"], "candidate_sha256": report["candidate_sha256"], "official": report["official"]}))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
