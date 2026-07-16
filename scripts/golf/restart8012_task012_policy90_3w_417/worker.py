#!/usr/bin/env python3
"""Three bounded task012 searches used by the wave417 orchestrator.

The lanes deliberately cover disjoint terminal families.  A sub-650 model
cannot afford a full-grid intermediate, so the useful standard-ONNX boundary
is a single output-producing spatial operator.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task012.json"
PARENT = ROOT / (
    "scripts/golf/root_task012_h8w8_policy90_272/candidates/"
    "task012_h8w8_policy90.onnx"
)
EXPECTED_PARENT_SHA = (
    "9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947"
)
EXPECTED = (1, 10, 30, 30)
CONFIGS = ((True, 1), (True, 4), (False, 1), (False, 4))

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def known_cases() -> list[dict[str, Any]]:
    payload = json.loads(KNOWN_PATH.read_text())
    return [case for split in ("train", "test", "arc-gen") for case in payload[split]]


def domain_cases() -> list[dict[str, Any]]:
    return [
        GEN.generate(colors=[1, 2], cols=[left, right], gravity=gravity)
        for left in range(3, 10)
        for right in range(3, 10)
        for gravity in range(4)
    ]


def fresh_cases(seed: int, count: int) -> list[dict[str, Any]]:
    random.seed(seed)
    return [GEN.generate() for _ in range(count)]


def session(model: onnx.ModelProto, disabled: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def evaluate(
    model: onnx.ModelProto,
    corpora: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    runs = []
    for disabled, threads in CONFIGS:
        runner = session(model, disabled, threads)
        rows = []
        for label, cases in corpora:
            correct = errors = nonfinite = shape_mismatch = small_positive = 0
            prediction = hashlib.sha256()
            for case in cases:
                converted = scoring.convert_to_numpy(case)
                if converted is None:
                    errors += 1
                    continue
                try:
                    raw = runner.run(["output"], {"input": converted["input"]})[0]
                except Exception:
                    errors += 1
                    continue
                shape_mismatch += int(tuple(raw.shape) != EXPECTED)
                nonfinite += int(not np.isfinite(raw).all())
                small_positive += int(np.any((raw > 0.0) & (raw < 0.25)))
                signs = np.asarray(raw > 0.0, dtype=np.uint8)
                prediction.update(np.packbits(signs).tobytes())
                correct += int(np.array_equal(signs, converted["output"] > 0.0))
            rows.append(
                {
                    "label": label,
                    "correct": correct,
                    "total": len(cases),
                    "rate": correct / len(cases),
                    "errors": errors,
                    "nonfinite": nonfinite,
                    "shape_mismatch": shape_mismatch,
                    "small_positive_cases": small_positive,
                    "prediction_sha256": prediction.hexdigest(),
                }
            )
        runs.append(
            {
                "config": f"{'disabled' if disabled else 'default'}_t{threads}",
                "corpora": rows,
            }
        )
    return {
        "runs": runs,
        "prediction_stable": all(
            run["corpora"][index]["prediction_sha256"]
            == runs[0]["corpora"][index]["prediction_sha256"]
            for run in runs
            for index in range(len(corpora))
        ),
    }


def build_model(
    weights: np.ndarray,
    bias: np.ndarray | None,
    pads: tuple[int, int, int, int],
    output: Path,
    producer: str,
) -> onnx.ModelProto:
    inputs = ["input", "w"] + (["b"] if bias is not None else [])
    initializers = [numpy_helper.from_array(weights.astype(np.float32), "w")]
    if bias is not None:
        initializers.append(numpy_helper.from_array(bias.astype(np.float32), "b"))
    node = helper.make_node(
        "Conv",
        inputs,
        ["output"],
        group=10,
        kernel_shape=list(weights.shape[-2:]),
        pads=list(pads),
    )
    graph = helper.make_graph(
        [node],
        producer,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(EXPECTED))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(EXPECTED))],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name=producer,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, output)
    return model


def worker_dense() -> dict[str, Any]:
    """Audit every contiguous crop of the retained 8x8 classifier."""
    if sha256(PARENT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("parent SHA mismatch")
    parent = onnx.load(PARENT)
    arrays = {item.name: numpy_helper.to_array(item) for item in parent.graph.initializer}
    weights = arrays["w"]
    bias = arrays["b"]
    domain = domain_cases()
    crops = []
    candidate_dir = HERE / "candidates" / "dense_crops"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    # Keep only contiguous rectangular receptive fields.  Removing from an
    # edge adjusts the corresponding pad so every surviving coefficient keeps
    # exactly the same spatial offset.
    for top in range(0, 4):
        for bottom in range(0, 5):
            for left in range(0, 4):
                for right in range(0, 5):
                    kh = 8 - top - bottom
                    kw = 8 - left - right
                    if kh <= 0 or kw <= 0 or kh * kw >= 64:
                        continue
                    pt, pl, pb, pr = 3 - top, 3 - left, 4 - bottom, 4 - right
                    if min(pt, pl, pb, pr) < 0:
                        continue
                    w = weights[:, :, top : 8 - bottom, left : 8 - right]
                    name = f"crop_t{top}b{bottom}l{left}r{right}.onnx"
                    path = candidate_dir / name
                    model = build_model(w, bias, (pt, pl, pb, pr), path, "task012_crop417")
                    runner = session(model, True, 1)
                    right_cases = errors = 0
                    for case in domain:
                        converted = scoring.convert_to_numpy(case)
                        try:
                            raw = runner.run(["output"], {"input": converted["input"]})[0]
                        except Exception:
                            errors += 1
                            continue
                        right_cases += int(np.array_equal(raw > 0.0, converted["output"] > 0.0))
                    profile = cost_of(str(path))
                    crops.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "crop": [top, bottom, left, right],
                            "kernel": [kh, kw],
                            "profile": {"memory": profile[0], "params": profile[1], "cost": profile[2]},
                            "domain_correct": right_cases,
                            "domain_total": len(domain),
                            "domain_rate": right_cases / len(domain),
                            "errors": errors,
                            "sha256": sha256(path),
                        }
                    )
    crops.sort(key=lambda row: (-row["domain_correct"], row["profile"]["cost"]))
    # Only retain the compact top diagnostic models; all others are generated
    # artifacts, not admitted candidates.
    kept = {row["path"] for row in crops[:12]}
    for path in candidate_dir.glob("*.onnx"):
        if str(path.relative_to(ROOT)) not in kept:
            path.unlink()
    result = {
        "worker": 0,
        "family": "contiguous crops of cost650 8x8 biased depthwise Conv",
        "attempts": len(crops),
        "best": crops[:12],
        "policy90_found": any(row["domain_rate"] >= 0.90 for row in crops),
    }
    (HERE / "worker0_dense.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def role_constraints(module, example, kh: int, kw: int, pt: int, pl: int):
    return module.role_state_constraints(example, kh, kw, pt, pl)


def worker_nobias() -> dict[str, Any]:
    """Check homogeneous (no-bias) dense families at the sub-650 boundary."""
    base_search = load_module(
        "task012_base_search_417",
        ROOT / "scripts/golf/root_task012_sub710_retrain_250/search.py",
    )
    no_bias_search = load_module(
        "task012_nobias_search_417",
        ROOT / "scripts/golf/root_task012_h8w8_nobias_274/search.py",
    )
    cases = base_search.domain_cases()
    layouts = [
        (7, 7, 3, 3),
        (7, 8, 3, 3),
        (7, 8, 3, 4),
        (8, 7, 3, 3),
        (8, 7, 4, 3),
        (7, 9, 3, 3),
        (7, 9, 3, 4),
        (7, 9, 3, 5),
        (9, 7, 3, 3),
        (9, 7, 4, 3),
        (9, 7, 5, 3),
        (8, 8, 3, 3),
    ]
    rows = []
    for kh, kw, pt, pl in layouts:
        feasible = contradictory = 0
        for case in cases:
            try:
                x, y = role_constraints(base_search, case, kh, kw, pt, pl)
            except RuntimeError:
                contradictory += 1
                continue
            feasible += int(no_bias_search.homogeneous_feasible(x, y)["success"])
        rows.append(
            {
                "kernel": [kh, kw],
                "padding_top_left": [pt, pl],
                "parameter_cost_without_bias": kh * kw * 10,
                "individually_homogeneous_feasible_states": feasible,
                "internally_contradictory_states": contradictory,
                "domain_states": len(cases),
            }
        )

    parent = onnx.load(PARENT)
    parent_arrays = {item.name: numpy_helper.to_array(item) for item in parent.graph.initializer}
    probe_path = HERE / "candidates" / "task012_h8w8_nobias_REJECTED.onnx"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe = build_model(
        parent_arrays["w"], None, (3, 3, 4, 4), probe_path, "task012_nobias417"
    )
    runtime = evaluate(probe, [("domain196", domain_cases())])
    profile = cost_of(str(probe_path))
    result = {
        "worker": 1,
        "family": "single output-only homogeneous depthwise Conv",
        "layouts": rows,
        "runtime_probe": {
            "path": str(probe_path.relative_to(ROOT)),
            "sha256": sha256(probe_path),
            "profile": {"memory": profile[0], "params": profile[1], "cost": profile[2]},
            **runtime,
        },
        "policy90_found": any(
            row["individually_homogeneous_feasible_states"] >= 177 for row in rows
        ),
    }
    (HERE / "worker1_nobias.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker_alternatives() -> dict[str, Any]:
    """Screen parameter-free single-op morphology and terminal-cost bounds."""
    # MaxPool/AveragePool/LpPool have the same positive support on nonnegative
    # one-hot inputs.  Around an isolated center, any rectangular pool emits a
    # Cartesian product R x C.  task012's center-color component instead has
    # the nine X offsets below.  Its row and column projections both have five
    # elements, so a Cartesian product containing it has at least 25 elements;
    # equality with the nine-cell X is impossible for every kernel/dilation/
    # padding, without a finite enumeration.
    target_offsets = sorted(
        [(0, 0)]
        + [(dr * distance, dc * distance)
           for dr in (-1, 1)
           for dc in (-1, 1)
           for distance in (1, 2)]
    )
    row_projection = sorted({row for row, _column in target_offsets})
    column_projection = sorted({column for _row, column in target_offsets})

    # Runtime witnesses for representative parameter-free terminals.
    witnesses = []
    witness_specs = [
        ("identity_pool", [1, 1], [1, 1], [0, 0, 0, 0]),
        ("box3_pool", [3, 3], [1, 1], [1, 1, 1, 1]),
        ("dilated3_pool", [3, 3], [2, 2], [2, 2, 2, 2]),
        ("line5_pool", [1, 5], [1, 1], [0, 2, 0, 2]),
    ]
    for name, kernel, dilations, pads in witness_specs:
        node = helper.make_node(
            "MaxPool", ["input"], ["output"], kernel_shape=kernel,
            dilations=dilations, pads=pads, strides=[1, 1]
        )
        graph = helper.make_graph(
            [node], name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(EXPECTED))],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(EXPECTED))],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
            producer_name="task012_pool417"
        )
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        path = HERE / "candidates" / f"task012_{name}_REJECTED.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, path)
        audit = evaluate(model, [("domain196", domain_cases())])
        profile = cost_of(str(path))
        witnesses.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "profile": {"memory": profile[0], "params": profile[1], "cost": profile[2]},
                **audit,
            }
        )

    # Any legal multi-node replacement with a full [1,10,30,30] intermediate
    # already spends >=9000 bytes even at one-byte dtype, far over cost650.
    # QuantizeLinear+QLinearConv and reshape-to-shared-kernel Conv are thus
    # excluded by cost, independently of accuracy.
    result = {
        "worker": 2,
        "family": "single parameter-free pooling terminals and multi-node cost floor",
        "pool_semantics": (
            "MaxPool, AveragePool, and LpPool have identical >0 support on canonical "
            "nonnegative one-hot input for these windows"
        ),
        "center_role_support_proof": {
            "target_offsets": target_offsets,
            "row_projection": row_projection,
            "column_projection": column_projection,
            "target_cardinality": len(target_offsets),
            "minimum_cartesian_product_cardinality_containing_target": (
                len(row_projection) * len(column_projection)
            ),
            "conclusion": "no rectangular pooling support can equal the center-color X",
        },
        "runtime_witnesses": witnesses,
        "policy90_found": False,
        "full_grid_intermediate_floor_bytes": 9000,
        "cost_target": 650,
        "excluded_by_cost": [
            "QuantizeLinear -> QLinearConv",
            "Reshape/Transpose -> shared-kernel Conv -> terminal reshape",
            "Conv -> Add/BatchNormalization terminal bias synthesis",
        ],
    }
    (HERE / "worker2_alternatives.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"0", "1", "2"}:
        raise SystemExit("usage: worker.py {0|1|2}")
    worker = int(sys.argv[1])
    result = (worker_dense, worker_nobias, worker_alternatives)[worker]()
    print(json.dumps({"worker": worker, "policy90_found": result["policy90_found"]}, indent=2))


if __name__ == "__main__":
    main()
