#!/usr/bin/env python3
"""Independent strict audit for C28 (tasks 190 and 195).

This lane deliberately does not promote or mutate shared artifacts.  It proves
the generator rules, audits the exact Wave15 members, and summarizes the
already-deduplicated repository history under the current strict structure
gate.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASKS))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref190(grid: list[list[int]]) -> np.ndarray:
    """Generator rule for 7ddcd7ec: extend selected corner markers."""
    arr = np.asarray(grid, dtype=np.uint8)
    color = int(arr.max())
    out = arr.copy()
    block: tuple[int, int] | None = None
    for row in range(arr.shape[0] - 1):
        for col in range(arr.shape[1] - 1):
            if np.all(arr[row : row + 2, col : col + 2] == color):
                block = row, col
                break
        if block is not None:
            break
    if block is None:
        raise AssertionError("task190 generator case has no 2x2 block")
    row, col = block
    for dr, dc in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        r = row + (2 if dr == 1 else -1)
        c = col + (2 if dc == 1 else -1)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1] and arr[r, c] == color:
            while 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                out[r, c] = color
                r += dr
                c += dc
    return out


def ref195(grid: list[list[int]]) -> np.ndarray:
    """Generator rule for 80af3007: 3x3 sprite Kronecker self-product."""
    gray = np.asarray(grid, dtype=np.uint8) == 5
    rows = np.flatnonzero(gray.any(axis=1))
    cols = np.flatnonzero(gray.any(axis=0))
    r0, c0 = int(rows[0]), int(cols[0])
    sprite = gray[r0 + np.arange(3) * 3][:, c0 + np.arange(3) * 3]
    return np.where(np.kron(sprite, sprite), 5, 0).astype(np.uint8)


def verify_reference(task: int, fresh: int = 5000) -> dict[str, int]:
    reference = ref190 if task == 190 else ref195
    dataset = json.loads((ROOT / "inputs" / "neurogolf-2026" / f"task{task:03d}.json").read_text())
    known = 0
    for cases in dataset.values():
        for case in cases:
            got = reference(case["input"])
            want = np.asarray(case["output"], dtype=np.uint8)
            if not np.array_equal(got, want):
                raise AssertionError(f"task{task:03d} reference known mismatch {known}")
            known += 1

    module = importlib.import_module("task_7ddcd7ec" if task == 190 else "task_80af3007")
    random.seed(28_000_000 + task)
    for index in range(fresh):
        case = module.generate()
        got = reference(case["input"])
        want = np.asarray(case["output"], dtype=np.uint8)
        if not np.array_equal(got, want):
            raise AssertionError(f"task{task:03d} reference fresh mismatch {index}")
    return {"known_right": known, "known_wrong": 0, "fresh_right": fresh, "fresh_wrong": 0}


def encode(case: list[list[int]]) -> np.ndarray:
    arr = np.asarray(case, dtype=np.uint8)
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    height, width = arr.shape
    for color in range(10):
        x[0, color, :height, :width] = arr == color
    return x


def shape_tuple(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)


def task195_runtime_shapes(model: onnx.ModelProto) -> dict[str, object]:
    """Expose intermediates and compare ORT runtime shapes with declarations."""
    names = ["gn", "q_u8", "q2", "q3", "q1", "a_u", "z_u"]
    declared = {value.name: shape_tuple(value) for value in model.graph.value_info if value.name in names}
    elem_types = {
        value.name: value.type.tensor_type.elem_type
        for value in model.graph.value_info
        if value.name in names
    }
    probe = copy.deepcopy(model)
    # Give the probe truthful shapes so ORT returns the real intermediates.  The
    # original declarations are retained separately in ``declared`` above.
    truthful = {
        "gn": (1, 10, 30, 30),
        "q_u8": (1, 10, 30, 30),
        "q2": (1, 2, 30, 30),
        "q3": (1, 3, 30, 30),
        "q1": (1, 1, 30, 30),
        "a_u": (1, 1, 3, 3),
        "z_u": (1, 1, 9, 9),
    }
    for name in names:
        probe.graph.output.append(
            helper.make_tensor_value_info(name, elem_types[name], list(truthful[name]))
        )
    dataset = json.loads((ROOT / "inputs" / "neurogolf-2026" / "task195.json").read_text())
    feed = encode(dataset["train"][0]["input"])
    modes: dict[str, object] = {}
    for label, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(probe.SerializeToString(), options, providers=["CPUExecutionProvider"])
        values = session.run(names, {"input": feed})
        actual = {name: tuple(value.shape) for name, value in zip(names, values, strict=True)}
        mismatches = {
            name: {"declared": declared[name], "runtime": actual[name]}
            for name in names
            if declared[name] != actual[name]
        }
        modes[label] = {
            "actual": {name: list(shape) for name, shape in actual.items()},
            "mismatches": {
                name: {key: list(shape) for key, shape in item.items()}
                for name, item in mismatches.items()
            },
        }
    return modes


def structural(task: int, model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    einsums = [len(node.input) for node in model.graph.node if node.op_type == "Einsum"]
    lookup_ops = [node.op_type for node in model.graph.node if node.op_type in {"TfIdfVectorizer", "Hardmax"}]
    used = {name for node in model.graph.node for name in node.input if name}
    unused = sorted(initializer.name for initializer in model.graph.initializer if initializer.name not in used)
    result: dict[str, object] = {
        "full_check": True,
        "strict_shape_inference": True,
        "node_count": len(model.graph.node),
        "initializer_params": int(sum(np.prod(initializer.dims) for initializer in model.graph.initializer)),
        "einsum_input_counts": einsums,
        "lookup_ops": lookup_ops,
        "unused_initializers": unused,
    }
    if task == 190:
        result["strict_gate"] = "reject_lookup_and_giant_einsum_25"
    else:
        result["runtime_shapes"] = task195_runtime_shapes(model)
        result["strict_gate"] = "reject_shape_cloak"
    return result


def history_floor(task: int) -> dict[str, object]:
    source = (
        ROOT / "scripts/golf/loop_7999_13/lane_a14/loose_history_scan.json"
        if task == 190
        else ROOT / "scripts/golf/loop_7999_13/lane_a17/loose_history_scan.json"
    )
    data = json.loads(source.read_text())
    rows = [row for row in data["rows"] if int(row.get("task", -1)) == task]
    passing = [row for row in rows if row.get("structure_gate") == "pass" and row.get("actual_screen_cost") is not None]
    cheaper = [row for row in rows if int(row.get("actual_screen_cost", 10**18)) < (153 if task == 190 else 150)]
    return {
        "source": str(source.relative_to(ROOT)),
        "unique_models": len(rows),
        "strict_structure_pass_min_cost": min(int(row["actual_screen_cost"]) for row in passing),
        "cheaper_screen_models": len(cheaper),
        "cheaper_screen_paths": [row["path"] for row in cheaper],
    }


def main() -> int:
    exact = {
        190: {"cost": 153, "sha256": "7f5f1cd6e9bb3158db6a4f15d25327c904e38a406710c7a28e7de58c1272a56e"},
        195: {"cost": 150, "sha256": "02ea0c97c9f63f58c7099c94a7bc2634eea9ea21bf69df9936a2b0f5f3e2d56c"},
    }
    rows: list[dict[str, object]] = []
    for task in (190, 195):
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        digest = sha256(path)
        if digest != exact[task]["sha256"]:
            raise AssertionError(f"task{task:03d} exact SHA mismatch: {digest}")
        model = onnx.load(path)
        rows.append(
            {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "exact_cost": exact[task]["cost"],
                "reference": verify_reference(task),
                "structure": structural(task, model),
                "history": history_floor(task),
                "winner": None,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    (HERE / "audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
