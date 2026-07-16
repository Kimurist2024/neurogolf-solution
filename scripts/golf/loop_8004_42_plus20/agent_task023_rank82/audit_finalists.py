#!/usr/bin/env python3
"""Final rejection audit for task023 rank82 candidates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

HELPER_PATH = ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/train_ranker.py"
SPEC = importlib.util.spec_from_file_location("task023_rank_helper_audit82", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load task023 helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/candidates/task023_9a2b78138891_cost1541.onnx"
MODELS = {
    "clean1541": SOURCE,
    "root_coordinate2": ROOT / "scripts/golf/loop_8004_42_plus20/root_task023_tune80/task023_ranker_coordinate2.onnx",
    "rank82_integer": HERE / "candidates/task023_rank82_integer_root_c2.onnx",
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def kernel(model: onnx.ModelProto) -> np.ndarray:
    return next(
        numpy_helper.to_array(item).copy()
        for item in model.graph.initializer
        if item.name == "score_W_q"
    )


def graph_equal_except_kernel(a: onnx.ModelProto, b: onnx.ModelProto) -> bool:
    aa = copy.deepcopy(a)
    bb = copy.deepcopy(b)
    for model in (aa, bb):
        for item in model.graph.initializer:
            if item.name == "score_W_q":
                item.CopyFrom(numpy_helper.from_array(np.zeros((1, 1, 6, 6), dtype=np.int8), item.name))
    return aa.SerializeToString() == bb.SerializeToString()


def exact_rate(x: np.ndarray, y: np.ndarray, value: np.ndarray) -> tuple[int, int]:
    raw = np.einsum("nif,f->ni", x.astype(np.int16), value.reshape(-1).astype(np.int16), optimize=True)
    score = np.clip(raw, 0, 255).astype(np.int32)
    key = score * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive = np.where(y, key, np.iinfo(np.int32).max).min(axis=1)
    negative = np.where(y, np.iinfo(np.int32).min, key).max(axis=1)
    mask = positive > negative
    return int(mask.sum()), len(mask)


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]


def main() -> int:
    known = HELPER.known_cases()
    known_x, known_y = HELPER.make_dataset(known)
    fresh_seeds = [923_023_001, 1_023_023_001]
    fresh_cases = [HELPER.generated_cases(5_000, seed) for seed in fresh_seeds]
    fresh = [HELPER.make_dataset(cases) for cases in fresh_cases]
    source = onnx.load(SOURCE)
    rows = []
    for label, path in MODELS.items():
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        with tempfile.TemporaryDirectory() as work:
            score = scoring.score_and_verify(model, 23, work, label, require_correct=True)
        known_disabled = HELPER.ort_rate(model, known, "disabled")
        known_default = HELPER.ort_rate(model, known, "default")
        value = kernel(model)
        fresh_rates = [exact_rate(x, y, value) for x, y in fresh]
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        sample_output = session.run(["output"], {"input": HELPER.onehot(fresh_cases[0][0]["input"])})[0]
        graph_output_shape = dims(inferred.graph.output[0])
        nodes = [node.op_type for node in model.graph.node]
        rows.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cost": score,
                "known_disabled": known_disabled,
                "known_default": known_default,
                "fresh_seeds": fresh_seeds,
                "fresh_exact": fresh_rates,
                "fresh_rates": [right / total for right, total in fresh_rates],
                "graph_equal_except_score_W_q": graph_equal_except_kernel(source, model),
                "score_W_q_shape": list(value.shape),
                "score_W_q_bytes": int(value.nbytes),
                "strict_data_prop": True,
                "static_positive_shapes": all(
                    all(d is not None and d > 0 for d in dims(item))
                    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
                ),
                "graph_output_shape": graph_output_shape,
                "runtime_output_shape": list(sample_output.shape),
                "truthful_output_shape": graph_output_shape == list(sample_output.shape),
                "runtime_errors_smoke": 0,
                "conv_bias_ub": 0,
                "qlinearconv_inputs": [len(node.input) for node in model.graph.node if node.op_type == "QLinearConv"],
                "banned_ops": sorted({op for op in nodes if op.upper() in BANNED or "SEQUENCE" in op.upper()}),
                "lookup_ops": sorted({op for op in nodes if op in {"TfIdfVectorizer", "Hardmax"}}),
                "custom_domains": sorted({item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}}),
                "giant_nodes": [
                    {"op": node.op_type, "inputs": len(node.input)}
                    for node in model.graph.node
                    if node.op_type == "Einsum" and len(node.input) >= 8
                ],
            }
        )
    result = {
        "task": 23,
        "authority_zip": "submission_base_8005.17.zip",
        "authority_cost": 1622,
        "required_fresh_rate": 0.9,
        "generator_non_injective": True,
        "rows": rows,
        "winner": None,
        "decision": "NO_ADOPTION",
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
