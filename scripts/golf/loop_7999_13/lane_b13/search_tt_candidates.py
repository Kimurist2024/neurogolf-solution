from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "scripts" / "golf" / "loop_7999_13" / "lane_b13"
SOURCE = (
    ROOT
    / "scripts"
    / "golf"
    / "loop_7999_13"
    / "lane_archive_all400"
    / "task254_r03_static68.onnx"
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


FEATURE_NAMES = "ABCDEj"
FEATURE_EQUATIONS = [
    "akmn",
    "Ak",
    "almd",
    "Bl",
    "aopu",
    "Co",
    "aqrd",
    "Dq",
    "atru",
    "Et",
    "aicd",
    "ji",
]
FEATURE_INPUTS = [
    "input",
    "V",
    "input",
    "V",
    "input",
    "V",
    "input",
    "V",
    "input",
    "V",
    "input",
    "V",
]


def coefficient_tensor() -> tuple[np.ndarray, np.ndarray]:
    model = onnx.load(SOURCE)
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }
    v = arrays["V"]
    u = np.einsum("PZ,Pb->b", v, v)
    f = np.einsum("efXY,Xb,Yb->efb", arrays["W"], v, v)
    t = np.einsum("efj,b,efb->efjb", arrays["P"], u, f)
    k = np.einsum("ABe,CDEf,efjb->ABCDEjb", arrays["CA"], arrays["CB"], t)
    return v, k


def tt_decompose(
    array: np.ndarray, group_dims: list[list[int]], ranks: tuple[int, int, int]
) -> list[np.ndarray]:
    current = array
    previous_rank = 1
    cores: list[np.ndarray] = []
    for index, dims in enumerate(group_dims[:-1]):
        width = int(np.prod(dims))
        matrix = current.reshape(previous_rank * width, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = min(ranks[index], singular.size)
        cores.append(u[:, :rank].reshape(previous_rank, *dims, rank))
        current = singular[:rank, None] * vh[:rank]
        previous_rank = rank
    cores.append(current.reshape(previous_rank, *group_dims[-1]))
    return cores


def reconstruct(cores: list[np.ndarray]) -> np.ndarray:
    value = cores[0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=([-1], [0]))
    return np.squeeze(value, axis=0)


def enumerate_configs(k: np.ndarray, keep: int = 60) -> list[dict[str, object]]:
    dimensions = [2] * 6 + [10]
    norm = float(np.linalg.norm(k))
    rows: list[dict[str, object]] = []
    for permutation in itertools.permutations(range(6)):
        order = [*permutation, 6]
        arranged = np.transpose(k, order)
        ordered_dimensions = [dimensions[index] for index in order]
        ordered_names = "".join(FEATURE_NAMES[index] for index in permutation) + "b"
        for cuts in itertools.combinations(range(1, 7), 3):
            groups: list[list[int]] = []
            start = 0
            for end in (*cuts, 7):
                groups.append(ordered_dimensions[start:end])
                start = end
            widths = [int(np.prod(group)) for group in groups]
            for ranks in itertools.product(range(1, 5), repeat=3):
                params = (
                    widths[0] * ranks[0]
                    + ranks[0] * widths[1] * ranks[1]
                    + ranks[1] * widths[2] * ranks[2]
                    + ranks[2] * widths[3]
                )
                if params > 55:
                    continue
                cores = tt_decompose(arranged, groups, ranks)
                rebuilt = reconstruct(cores)
                difference = rebuilt - arranged
                rows.append(
                    {
                        "order": ordered_names,
                        "cuts": list(cuts),
                        "ranks": list(ranks),
                        "core_params": params,
                        "total_params": 20 + params,
                        "relative_l2_error": float(np.linalg.norm(difference) / norm),
                        "max_abs_error": float(np.max(np.abs(difference))),
                    }
                )
    rows.sort(
        key=lambda row: (
            float(row["relative_l2_error"]),
            int(row["total_params"]),
            str(row["order"]),
        )
    )
    # Keep some parameter diversity instead of only symmetry-equivalent top rows.
    retained: list[dict[str, object]] = []
    seen_signatures: set[tuple[object, ...]] = set()
    for row in rows:
        signature = (
            round(float(row["relative_l2_error"]), 12),
            int(row["core_params"]),
            tuple(row["cuts"]),
            tuple(row["ranks"]),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        retained.append(row)
        if len(retained) >= keep:
            break
    return retained


def make_model(v: np.ndarray, k: np.ndarray, config: dict[str, object]) -> onnx.ModelProto:
    order_names = str(config["order"])
    name_to_axis = {name: index for index, name in enumerate(FEATURE_NAMES + "b")}
    order = [name_to_axis[name] for name in order_names]
    arranged = np.transpose(k, order)
    cuts = [int(value) for value in config["cuts"]]
    dimensions = [arranged.shape[index] for index in range(arranged.ndim)]
    groups: list[list[int]] = []
    name_groups: list[str] = []
    start = 0
    for end in (*cuts, 7):
        groups.append(dimensions[start:end])
        name_groups.append(order_names[start:end])
        start = end
    ranks = tuple(int(value) for value in config["ranks"])
    cores = tt_decompose(arranged, groups, ranks)
    cores[0] = cores[0][0]
    bond_names = "xyz"
    core_equations = [
        name_groups[0] + bond_names[0],
        bond_names[0] + name_groups[1] + bond_names[1],
        bond_names[1] + name_groups[2] + bond_names[2],
        bond_names[2] + name_groups[3],
    ]
    initializers = [numpy_helper.from_array(v.astype(np.float32), name="V")]
    core_inputs = []
    for index, core in enumerate(cores):
        name = f"K{index}"
        core_inputs.append(name)
        initializers.append(numpy_helper.from_array(core.astype(np.float32), name=name))
    equation = ",".join([*FEATURE_EQUATIONS, *core_equations]) + "->abcd"
    node = helper.make_node(
        "Einsum", [*FEATURE_INPUTS, *core_inputs], ["output"], equation=equation
    )
    graph = helper.make_graph(
        [node],
        "task254_tt16",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_result(model: onnx.ModelProto, disable_all: bool) -> dict[str, object]:
    session = make_session(model, disable_all)
    right = wrong = errors = nonfinite = 0
    min_positive = float("inf")
    max_nonpositive = -float("inf")
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(254)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
                correct = np.array_equal(raw > 0, benchmark["output"] > 0)
                right += int(correct)
                wrong += int(not correct)
                finite = np.isfinite(raw)
                nonfinite += int(np.count_nonzero(~finite))
                positive = raw[(raw > 0) & finite]
                nonpositive = raw[(raw <= 0) & finite]
                if positive.size:
                    min_positive = min(min_positive, float(positive.min()))
                if nonpositive.size:
                    max_nonpositive = max(max_nonpositive, float(nonpositive.max()))
            except Exception:  # noqa: BLE001
                errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "runtime_errors": errors,
        "nonfinite_elements": nonfinite,
        "min_positive": min_positive if np.isfinite(min_positive) else None,
        "max_nonpositive": max_nonpositive if np.isfinite(max_nonpositive) else None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    v, k = coefficient_tensor()
    configs = enumerate_configs(k)
    attempts_dir = LANE / "tt_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    winners = []
    for index, config in enumerate(configs):
        model = make_model(v, k, config)
        path = attempts_dir / f"task254_tt_{index:02d}.onnx"
        onnx.save(model, path)
        row = dict(config)
        row.update(
            {
                "index": index,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "einsum_operands": 16,
                "disable_all": known_result(model, True),
                "default": known_result(model, False),
            }
        )
        row["known_pass"] = all(
            row[mode]["right"] == 265
            and row[mode]["wrong"] == 0
            and row[mode]["runtime_errors"] == 0
            and row[mode]["nonfinite_elements"] == 0
            for mode in ("disable_all", "default")
        )
        rows.append(row)
        if row["known_pass"]:
            winners.append(row)
        print(
            index,
            config["total_params"],
            config["relative_l2_error"],
            row["disable_all"]["right"],
            row["default"]["right"],
            flush=True,
        )
    output = {
        "source": str(SOURCE.relative_to(ROOT)),
        "operand_cap": 16,
        "incumbent_cost": 76,
        "attempt_count": len(rows),
        "rows": rows,
        "known_winners": winners,
    }
    (LANE / "tt_search.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0 if winners else 2


if __name__ == "__main__":
    raise SystemExit(main())
