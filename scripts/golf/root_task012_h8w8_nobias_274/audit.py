#!/usr/bin/env python3
"""Independent exact-certificate audit for task012 no-bias search 274."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEARCH_PATH = HERE / "search.json"
CANDIDATES_PATH = HERE / "candidates.json"
PARENT = ROOT / (
    "scripts/golf/root_task012_h8w8_policy90_272/candidates/"
    "task012_h8w8_policy90.onnx"
)
PARENT_SHA256 = "9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KERNEL = 8
PAD_TOP = PAD_LEFT = 3

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.rank_dir import cost_of  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_channels(example: dict[str, Any]) -> tuple[int, int]:
    values = np.asarray(example["input"], dtype=np.int8)
    colors, counts = np.unique(values[values > 0], return_counts=True)
    return int(colors[np.argmin(counts)]), int(colors[np.argmax(counts)])


def encode(grid: list[list[int]], channel: int) -> np.ndarray:
    source = np.asarray(grid, dtype=np.int8)
    target = np.zeros((30, 30), dtype=np.uint8)
    target[: source.shape[0], : source.shape[1]] = source == channel
    return target


def patches(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((3, 4), (3, 4)))
    return np.lib.stride_tricks.sliding_window_view(padded, (8, 8)).reshape(900, 64)


def independent_constraints(
    example: dict[str, Any], channels: tuple[int, ...]
) -> tuple[list[bytes], np.ndarray, np.ndarray]:
    mapping: dict[bytes, int] = {}
    for channel in channels:
        inputs = patches(encode(example["input"], channel))
        outputs = encode(example["output"], channel).reshape(-1)
        for patch, label in zip(inputs, outputs, strict=True):
            key = np.packbits(patch, bitorder="big").tobytes()
            value = int(label)
            if key in mapping and mapping[key] != value:
                raise AssertionError("internal contradictory patch")
            mapping[key] = value
    keys = sorted(mapping)
    inputs = np.unpackbits(
        np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1),
        axis=1,
        bitorder="big",
    )[:, :64].astype(np.int8)
    labels = np.asarray([mapping[key] for key in keys], dtype=np.int8)
    return keys, inputs, labels


def background_feasible(inputs: np.ndarray, labels: np.ndarray) -> bool:
    matrix = np.where(labels[:, None] > 0, -inputs, inputs).astype(np.float64)
    bounds = np.where(labels > 0, -1.0, 0.0)
    result = linprog(
        np.zeros(64),
        A_ub=matrix,
        b_ub=bounds,
        bounds=[(None, None)] * 64,
        method="highs",
    )
    return bool(result.success)


def verify_certificate(
    stored: dict[str, Any], keys: list[bytes], inputs: np.ndarray, labels: np.ndarray
) -> bool:
    if not stored or not stored["exact_rational"]:
        return False
    signed = np.where(labels[:, None] > 0, -inputs, inputs).astype(np.int8)
    total = [Fraction(0) for _ in range(64)]
    positive_mass = Fraction(0)
    for item in stored["support"]:
        index = int(item["constraint_index"])
        coefficient = Fraction(item["coefficient"])
        if coefficient < 0 or keys[index].hex() != item["patch_hex"]:
            return False
        if int(labels[index]) != int(item["label"]):
            return False
        for column in range(64):
            total[column] += coefficient * int(signed[index, column])
        if labels[index] > 0:
            positive_mass += coefficient
    return bool(
        all(value == 0 for value in total)
        and positive_mass == 1
        and stored["positive_coefficient_mass"] == "1"
    )


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def no_bias_profile(parent: onnx.ModelProto) -> dict[str, int]:
    model = onnx.ModelProto()
    model.CopyFrom(parent)
    del model.graph.node[0].input[2:]
    weights = [item for item in model.graph.initializer if item.name != "b"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    with tempfile.TemporaryDirectory(prefix="audit274_", dir="/tmp") as directory:
        path = Path(directory) / "nobias.onnx"
        onnx.save(model, path)
        return profile(path)


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    search = json.loads(SEARCH_PATH.read_text())
    candidates = json.loads(CANDIDATES_PATH.read_text())
    checks: list[str] = []

    require(sha256(PARENT) == PARENT_SHA256, "cost650 parent SHA matches", checks)
    require(profile(PARENT) == {"memory": 0, "params": 650, "cost": 650}, "parent official profile is cost650", checks)
    parent = onnx.load(PARENT)
    require(no_bias_profile(parent) == {"memory": 0, "params": 640, "cost": 640}, "biasless family profile is cost640", checks)
    node = parent.graph.node[0]
    attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in parent.graph.initializer}
    require(
        node.op_type == "Conv"
        and attributes["group"] == 10
        and attributes["pads"] == [3, 3, 4, 4]
        and list(arrays["w"].shape) == [10, 1, 8, 8],
        "parent architecture is the requested 8x8 group10 Conv",
        checks,
    )
    require(
        all(arrays["w"][index].tobytes() == arrays["w"][1].tobytes() for index in range(2, 10)),
        "nonzero channels 1..9 use one byte-identical kernel",
        checks,
    )

    state_rows = {tuple(row["state"]): row for row in search["state_rows"]}
    require(len(state_rows) == 196, "search contains 196 unique states", checks)
    background_count = certificate_count = 0
    regenerated_states = []
    support_histogram: dict[int, int] = {}
    for left in range(3, 10):
        for right in range(3, 10):
            for gravity in range(4):
                state = (left, right, gravity)
                regenerated_states.append(state)
                example = GEN.generate(colors=[1, 2], cols=[left, right], gravity=gravity)
                center, arm = role_channels(example)
                background_keys, background_x, background_y = independent_constraints(example, (0,))
                foreground_keys, foreground_x, foreground_y = independent_constraints(example, (center, arm))
                row = state_rows[state]
                require(
                    row["background_constraint_digest"]
                    == hashlib.sha256(b"".join(background_keys) + background_y.tobytes()).hexdigest()
                    and row["foreground_constraint_digest"]
                    == hashlib.sha256(b"".join(foreground_keys) + foreground_y.tobytes()).hexdigest(),
                    f"state{state}: independently regenerated constraint digests match",
                    checks,
                )
                background_count += int(background_feasible(background_x, background_y))
                certificate = row["foreground_farkas_certificate"]
                exact = verify_certificate(certificate, foreground_keys, foreground_x, foreground_y)
                certificate_count += int(exact)
                size = int(certificate["support_size"])
                support_histogram[size] = support_histogram.get(size, 0) + 1

    state_digest = hashlib.sha256(np.asarray(regenerated_states, dtype=np.int16).tobytes()).hexdigest()
    require(
        state_digest == search["complete_domain"]["state_tuple_sha256"],
        "state tuple digest independently matches",
        checks,
    )
    require(background_count == 196, "background kernel is individually feasible for all 196 states", checks)
    require(certificate_count == 196, "all 196 foreground infeasibilities have exact rational certificates", checks)
    require(
        search["complete_domain"]["case_level_optimal_upper_bound"] == 0,
        "exact case-level optimum upper bound is 0/196",
        checks,
    )
    require(
        math.ceil(0.90 * 196) == search["complete_domain"]["policy90_required_cases"] == 177,
        "POLICY90 requires 177/196 cases",
        checks,
    )
    require(search["candidate"] is None and search["winner"] is None, "search has no candidate or winner", checks)
    require(candidates["candidates"] == [] and candidates["winner"] is None, "candidate ledger is empty", checks)
    require(not search["candidate_runtime_gate"]["run"], "runtime candidate gate is correctly skipped", checks)
    require(not any(search["policy"].values()), "policy rejects lookup/cloak/private-zero/protected writes", checks)

    payload = {
        "task": 12,
        "lane": "root_task012_h8w8_nobias_274",
        "pass": True,
        "decision": "NO_POLICY90_NOBIAS_CANDIDATE",
        "search_sha256": sha256(SEARCH_PATH),
        "searcher_sha256": sha256(HERE / "search.py"),
        "candidates_sha256": sha256(CANDIDATES_PATH),
        "parent_sha256": sha256(PARENT),
        "checks": checks,
        "exact_certificate_support_histogram": support_histogram,
        "optimal_upper_bound": {"right": 0, "total": 196, "rate": 0.0},
        "candidate_execution_gate": {
            "required": False,
            "reason": search["candidate_runtime_gate"]["skip_reason"],
        },
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
